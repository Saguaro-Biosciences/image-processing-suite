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

# --- 1. QC Math Functions (V7: Quantization & Mag Logic) ---

def calculate_v7_metrics(image, channel_name):
    """
    V7 Diagnostic:
    1. Slope_Quantized_8bit: Forces image into 256 bins (adds noise).
    2. Slope_Theoretical_Power: Calculates (Slope_Magnitude * 2).
    """
    results = {}
    
    # --- Percent Maximal ---
    if image.dtype == np.uint8: max_val = 255
    else: max_val = 65535
    pct_max = (np.sum(image >= max_val) / image.size) * 100
    results[f'ImageQuality_PercentMaximal_{channel_name}'] = pct_max

    try:
        # Common Radial Setup
        h, w = image.shape
        max_r = int(min(h, w) / 2)
        y, x = np.indices((h, w))
        center_y, center_x = h // 2, w // 2
        r_int = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        r_flat = r_int.ravel()

        # --- CANDIDATE 1: 8-BIT QUANTIZATION NOISE ---
        # Convert 16-bit (0-65535) -> 8-bit (0-255) -> Float
        # This adds "steps" that look like high-frequency noise
        if image.dtype == np.uint16:
            img_8bit = (image / 256).astype(np.uint8)
            img_input = img_8bit.astype(float) / 255.0
        elif image.dtype == np.uint8:
            img_input = image.astype(float) / 255.0
        else:
            # Force quantization on float inputs too
            img_input = ((image / image.max()) * 255).astype(np.uint8).astype(float) / 255.0

        F = scipy.fft.fft2(img_input)
        F_shifted = scipy.fft.fftshift(F)
        power_spectrum = np.abs(F_shifted) ** 2
        
        # Radial Sum (Power)
        p_flat = power_spectrum.ravel()
        radial_sum = np.bincount(r_flat, weights=p_flat)
        
        # Slice & Regress
        if len(radial_sum) > max_r:
            vals = radial_sum[1:max_r+1]
            freqs = np.arange(1, max_r + 1)
        else:
            vals = radial_sum[1:]
            freqs = np.arange(1, len(radial_sum) + 1)
            
        valid = vals > 0
        if np.sum(valid) > 2:
            s_quant, _, _, _, _ = scipy.stats.linregress(np.log(freqs[valid]), np.log(vals[valid]))
            results[f'Slope_Quant8bit_{channel_name}'] = s_quant
        else:
            results[f'Slope_Quant8bit_{channel_name}'] = np.nan

        # --- CANDIDATE 2: THEORETICAL POWER (Mag * 2) ---
        # Based on your V6 result: Mag Slope (-0.75) * 2 = -1.5
        # We calculate Magnitude Slope on the HIGH PRECISION image
        img_raw = image.astype(float)
        F_raw = scipy.fft.fft2(img_raw)
        F_shifted_raw = scipy.fft.fftshift(F_raw)
        mag_spectrum = np.abs(F_shifted_raw) # Note: NOT SQUARED
        
        m_flat = mag_spectrum.ravel()
        radial_sum_mag = np.bincount(r_flat, weights=m_flat)
        
        if len(radial_sum_mag) > max_r:
            vals_mag = radial_sum_mag[1:max_r+1]
        else:
            vals_mag = radial_sum_mag[1:]
            
        valid_mag = vals_mag > 0
        if np.sum(valid_mag) > 2:
            s_mag, _, _, _, _ = scipy.stats.linregress(np.log(freqs[valid_mag]), np.log(vals_mag[valid_mag]))
            # MULTIPLY BY 2 to get "Theoretical Power Slope"
            results[f'Slope_Theoretical_{channel_name}'] = s_mag * 2.0
        else:
            results[f'Slope_Theoretical_{channel_name}'] = np.nan

    except Exception:
        results[f'Slope_Quant8bit_{channel_name}'] = np.nan
        results[f'Slope_Theoretical_{channel_name}'] = np.nan

    return results

# --- 2. Worker ---

def qc_producer_worker(task_queue, results_dict, worker_id, channels, illum_path):
    if worker_id == 0:
        print("--- WORKER STARTED: V7 (8-bit Quantization vs Theoretical Power) ---")

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
                
                # Apply Illumination
                if corrections and corrections[i] is not None:
                     if img.shape == corrections[i].shape:
                        img = img / corrections[i]
                
                metrics = calculate_v7_metrics(img, ch_name)
                site_results.update(metrics)
            
            results_dict[index] = site_results

        except Exception as e:
            results_dict[index] = {f"QC_Error": str(e)}

# --- 3. Orchestrator ---

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(f"Reading CSV: {args.load_data}")
    
    df = pd.read_csv(args.load_data)
    
    # SAFETY: Drop existing test columns
    cols_to_drop = [c for c in df.columns if 'Slope_' in c]
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

    logging.info(f"Starting QC V7 on {len(tasks)} sites...")
    
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
    parser.add_argument('--output', type=str, default='QC_Results_V7.csv')
    parser.add_argument('--threads', type=int, default=24)
    args = parser.parse_args()

    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    main(args)