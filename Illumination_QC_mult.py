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

# --- 1. QC Math Functions ---

def calculate_cp_identical_slope(image, channel_name):
    """
    Replicates CellProfiler MeasureImageQuality PowerLogLogSlope.
    
    ADJUSTMENT V4:
    - Logic: Radial MEAN (PSD), not Sum.
    - Range: FULL SPECTRUM (Includes Corners).
    
    Why? 
    - Slope -2.5 (Previous) = Structural limit (Low Freq).
    - Slope -1.4 (Target) = Structure + Noise (High Freq).
    - We must include the corners (high freq noise) to flatten the slope.
    """
    results = {}
    
    # --- Percent Maximal (Saturation) ---
    if image.dtype == np.uint8: max_val = 255
    else: max_val = 65535
        
    pct_max = (np.sum(image >= max_val) / image.size) * 100
    results[f'ImageQuality_PercentMaximal_{channel_name}'] = pct_max

    # --- PowerLogLogSlope (Mean + Full Range) ---
    try:
        # 1. FFT & Power Spectrum
        image_float = image.astype(float)
        F = scipy.fft.fft2(image_float)
        F_shifted = scipy.fft.fftshift(F)
        power_spectrum = np.abs(F_shifted) ** 2
        
        # 2. Radial Map
        h, w = image.shape
        center_y, center_x = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # 3. Define Max Radius (INCLUDE CORNERS)
        # Previous code stopped at min(h,w)//2.
        # We now go to the full diagonal to capture high-freq noise.
        r_int = r.astype(int)
        max_r = int(np.max(r_int))
        
        # 4. Radial MEAN (PSD)
        # We calculate Sum and Count, then divide.
        r_flat = r_int.ravel()
        p_flat = power_spectrum.ravel()
        
        # Sum of power per ring
        radial_sum = np.bincount(r_flat, weights=p_flat)
        # Count of pixels per ring
        pixel_count = np.bincount(r_flat)
        
        # Handle cases where bincount is smaller than max_r (rare)
        if len(radial_sum) < max_r + 1:
            max_r = len(radial_sum) - 1
            
        # 5. Compute Mean (PSD)
        # Avoid division by zero
        valid_bins = pixel_count > 0
        radial_mean = np.zeros_like(radial_sum)
        radial_mean[valid_bins] = radial_sum[valid_bins] / pixel_count[valid_bins]
        
        # 6. Select Range (Ignore DC at 0)
        # We use the full valid range up to max_r
        indices = np.arange(len(radial_mean))
        
        # Filter: Start at 1, go to max, ensure mean > 0
        mask = (indices >= 1) & (indices <= max_r) & (radial_mean > 0)
        
        if np.sum(mask) > 10: # Ensure enough points for fit
            freq_log = np.log(indices[mask])
            power_log = np.log(radial_mean[mask])
            
            slope, _, _, _, _ = scipy.stats.linregress(freq_log, power_log)
            results[f'ImageQuality_PowerLogLogSlope_{channel_name}'] = slope
        else:
            results[f'ImageQuality_PowerLogLogSlope_{channel_name}'] = np.nan
            
    except Exception:
        results[f'ImageQuality_PowerLogLogSlope_{channel_name}'] = np.nan

    return results

# --- 2. Parallel Worker ---

def qc_producer_worker(task_queue, results_dict, worker_id, channels, illum_path):
    if worker_id == 0:
        print("--- DEBUG: WORKER RUNNING V4 (MEAN + CORNERS) LOGIC ---")
        
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
                
                if corrections and corrections[i] is not None:
                    if img.shape == corrections[i].shape:
                        img = img / corrections[i]
                
                metrics = calculate_cp_identical_slope(img, ch_name)
                site_results.update(metrics)
            
            results_dict[index] = site_results

        except Exception as e:
            results_dict[index] = {f"QC_Error": str(e)}

# --- 3. Orchestrator ---

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(f"Reading CSV: {args.load_data}")
    logging.info("!!! VERSION CHECK: RUNNING V4 (MEAN + CORNERS) !!!")
    
    df = pd.read_csv(args.load_data)
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

    logging.info(f"Starting QC on {len(tasks)} sites...")
    
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
    parser.add_argument('--output', type=str, default='QC_Results.csv')
    parser.add_argument('--threads', type=int, default=24)
    args = parser.parse_args()

    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    main(args)