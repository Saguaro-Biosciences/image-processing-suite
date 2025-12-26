import os
import argparse
import logging
import time
import tifffile
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
import scipy.fft
import scipy.ndimage
import scipy.stats
from tqdm import tqdm
from queue import Empty

# --- 1. QC Math Functions (Tri-State Diagnostic) ---

def calculate_diagnostic_slopes(image, channel_name):
    """
    Calculates 3 variations of the slope to identify the CellProfiler match.
    """
    results = {}
    
    # --- Percent Maximal ---
    # Calc on raw image
    if image.dtype == np.uint8: max_val = 255
    else: max_val = 65535
    pct_max = (np.sum(image >= max_val) / image.size) * 100
    results[f'IQ_PctMax_{channel_name}'] = pct_max

    try:
        h, w = image.shape
        max_r = int(min(h, w) / 2)
        
        # --- PREP 1: RAW SCALING (Current Method) ---
        if image.dtype == np.uint16:
            img_raw = image.astype(float) / 65535.0
        else:
            img_raw = image.astype(float) / 255.0
            
        # --- PREP 2: MIN-MAX STRETCH (CellProfiler "Rescale" Method) ---
        # Stretches the dynamic range to 0-1, amplifying noise floor
        img_min = np.min(image)
        img_max = np.max(image)
        if img_max > img_min:
            img_stretched = (image.astype(float) - img_min) / (img_max - img_min)
        else:
            img_stretched = np.zeros_like(image, dtype=float)

        # Helper to calc slope
        def get_slope(img_input, use_magnitude=False):
            F = scipy.fft.fft2(img_input)
            F_shifted = scipy.fft.fftshift(F)
            
            if use_magnitude:
                spectrum = np.abs(F_shifted) # Magnitude (|F|)
            else:
                spectrum = np.abs(F_shifted) ** 2 # Power (|F|^2)
            
            # Radial Sum
            y, x = np.indices(img_input.shape)
            center_y, center_x = img_input.shape[0] // 2, img_input.shape[1] // 2
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
            
            r_flat = r.ravel()
            p_flat = spectrum.ravel()
            radial_sum = np.bincount(r_flat, weights=p_flat)
            
            # Slice valid range
            if len(radial_sum) > max_r:
                radial_sum = radial_sum[1:max_r+1]
                freqs = np.arange(1, max_r + 1)
            else:
                radial_sum = radial_sum[1:]
                freqs = np.arange(1, len(radial_sum) + 1)
            
            # Regression
            valid = radial_sum > 0
            if np.sum(valid) > 2:
                s, _, _, _, _ = scipy.stats.linregress(np.log(freqs[valid]), np.log(radial_sum[valid]))
                return s
            return np.nan

        # --- METRIC 1: RAW SUM (Baseline: -2.5) ---
        results[f'Slope_Raw_{channel_name}'] = get_slope(img_raw, use_magnitude=False)

        # --- METRIC 2: STRETCHED SUM (Hypothesis: -1.5) ---
        results[f'Slope_Stretched_{channel_name}'] = get_slope(img_stretched, use_magnitude=False)

        # --- METRIC 3: MAGNITUDE (Hypothesis: -1.25) ---
        # (Calculating on raw image, as log-log slope is scale invariant, 
        # but magnitude vs power changes slope by factor of 0.5)
        results[f'Slope_Magnitude_{channel_name}'] = get_slope(img_raw, use_magnitude=True)

    except Exception:
        results[f'Slope_Raw_{channel_name}'] = np.nan
        results[f'Slope_Stretched_{channel_name}'] = np.nan
        results[f'Slope_Magnitude_{channel_name}'] = np.nan

    return results

# --- 2. Worker ---

def qc_producer_worker(task_queue, results_dict, worker_id, channels, illum_path):
    if worker_id == 0:
        print("--- WORKER STARTED: QC_V6_Diagnostic (Raw vs Stretched vs Mag) ---")

    corrections = None
    if illum_path:
        try:
            corrections = []
            for c in channels:
                c_path = os.path.join(illum_path, f"{c}_illum.npy")
                if os.path.exists(c_path):
                    corrections.append(np.load(c_path))
                else:
                    corrections.append(None)
        except Exception:
            pass

    while True:
        try:
            task = task_queue.get(timeout=2)
        except Empty:
            break
        if task is None: break

        index, paths = task
        site_results = {}

        try:
            for i, (path, ch_name) in enumerate(zip(paths, channels)):
                if not os.path.exists(path):
                    site_results[f"QC_Error_{ch_name}"] = "File Not Found"
                    continue

                img = tifffile.imread(path)
                
                # Apply Illumination (Raw Division)
                # Note: We apply stretch inside the calc function to test both
                if corrections and corrections[i] is not None:
                     if img.shape == corrections[i].shape:
                        img = img / corrections[i]
                
                metrics = calculate_diagnostic_slopes(img, ch_name)
                site_results.update(metrics)
            
            results_dict[index] = site_results

        except Exception as e:
            results_dict[index] = {f"QC_Error": str(e)}

# --- 3. Orchestrator ---

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(f"Reading CSV: {args.load_data}")
    
    df = pd.read_csv(args.load_data)
    
    # SAFETY: Drop any existing test columns to ensure fresh data
    cols_to_drop = [c for c in df.columns if 'Slope_' in c or 'IQ_' in c]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    channel_cols = [f'FileName_{c}' for c in args.channels]
    
    tasks = [
        (idx, [os.path.join(args.data_path, row[col]) for col in channel_cols])
        for idx, row in df.iterrows()
    ]

    manager = mp.Manager()
    results_dict = manager.dict()
    task_queue = mp.Queue()

    for t in tasks: task_queue.put(t)
    for _ in range(args.threads): task_queue.put(None)

    logging.info(f"Starting QC V6 on {len(tasks)} sites...")
    
    processes = []
    for i in range(args.threads):
        p = mp.Process(target=qc_producer_worker, 
                       args=(task_queue, results_dict, i, args.channels, args.illum_path))
        p.start()
        processes.append(p)

    with tqdm(total=len(tasks)) as pbar:
        while True:
            completed = len(results_dict)
            pbar.n = completed
            pbar.refresh()
            if completed >= len(tasks): break
            if not any(p.is_alive() for p in processes) and completed < len(tasks): break
            time.sleep(1)

    for p in processes: p.join()

    qc_df = pd.DataFrame.from_dict(results_dict, orient='index').sort_index()
    final_df = pd.concat([df, qc_df], axis=1)
    
    final_df.to_csv(args.output, index=False)
    logging.info(f"Saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-data', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--illum-path', type=str, default=None)
    parser.add_argument('--channels', nargs='+', required=True)
    parser.add_argument('--output', type=str, default='QC_Results_V6.csv')
    parser.add_argument('--threads', type=int, default=24)
    args = parser.parse_args()

    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    main(args)