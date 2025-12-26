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

# --- 1. QC Math Functions (Renamed to FORCE worker update) ---

def calculate_cp_metrics_v2(image, channel_name):
    """
    V2 Update: Matches CellProfiler logic by using RADIAL SUM (not Mean).
    
    Logic:
    1. FFT -> Shift -> Power Spectrum (|F|^2).
    2. Radial Sum: Sum of power in each ring.
       (Slope of Sum = Slope of Mean + 1.0).
       This shifts the slope from ~ -2.5 to ~ -1.5.
    """
    results = {}
    
    # --- Percent Maximal (Saturation) ---
    if image.dtype == np.uint8:
        max_val = 255
    else:
        max_val = 65535 # Default to 16-bit
        
    num_saturated = np.sum(image >= max_val)
    pct_max = (num_saturated / image.size) * 100
    results[f'ImageQuality_PercentMaximal_{channel_name}'] = pct_max

    # --- PowerLogLogSlope (Radial Sum) ---
    try:
        # 1. FFT
        image_float = image.astype(float)
        F = scipy.fft.fft2(image_float)
        F_shifted = scipy.fft.fftshift(F)
        
        # 2. Power Spectrum (Squared Magnitude)
        power_spectrum = np.abs(F_shifted) ** 2
        
        # 3. Radial Map
        h, w = image.shape
        center_y, center_x = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r_int = r.astype(int)
        
        # 4. Define Max Radius (Nyquist / Inscribed Circle)
        # We must stop before the corners, or the sum drops artifically.
        max_r = int(min(h, w) / 2)
        
        # 5. Radial Summation
        # Use simple bincount with weights (This calculates SUM, not Mean)
        r_flat = r_int.ravel()
        p_flat = power_spectrum.ravel()
        
        radial_sum = np.bincount(r_flat, weights=p_flat)
        
        # Slice to valid range [1, max_r] (Ignore DC at 0)
        if len(radial_sum) > max_r:
            radial_sum = radial_sum[1:max_r+1]
            freqs = np.arange(1, max_r + 1)
        else:
            radial_sum = radial_sum[1:]
            freqs = np.arange(1, len(radial_sum) + 1)
        
        # 6. Log-Log Slope
        valid_mask = radial_sum > 0
        if np.sum(valid_mask) > 2:
            freq_log = np.log(freqs[valid_mask])
            power_log = np.log(radial_sum[valid_mask])
            
            slope, _, _, _, _ = scipy.stats.linregress(freq_log, power_log)
            results[f'ImageQuality_PowerLogLogSlope_{channel_name}'] = slope
        else:
            results[f'ImageQuality_PowerLogLogSlope_{channel_name}'] = np.nan
            
    except Exception as e:
        results[f'ImageQuality_PowerLogLogSlope_{channel_name}'] = np.nan

    return results

# --- 2. Parallel Worker (Renamed) ---

def qc_worker_v2(task_queue, results_dict, worker_id, channels, illum_path):
    """
    Worker V2: Uses calculate_cp_metrics_v2
    """
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
            
        if task is None:
            break

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
                
                # CALL THE NEW V2 FUNCTION
                metrics = calculate_cp_metrics_v2(img, ch_name)
                site_results.update(metrics)
            
            results_dict[index] = site_results

        except Exception as e:
            results_dict[index] = {f"QC_Error": str(e)}

# --- 3. Orchestrator ---

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    logging.info(f"Reading CSV: {args.load_data}")
    df = pd.read_csv(args.load_data)
    
    channel_cols = [f'FileName_{c}' for c in args.channels]
    
    # 1. Setup Tasks
    tasks = [
        (idx, [os.path.join(args.data_path, row[col]) for col in channel_cols])
        for idx, row in df.iterrows()
    ]

    # 2. Setup Multiprocessing
    manager = mp.Manager()
    results_dict = manager.dict()
    task_queue = mp.Queue()

    for t in tasks:
        task_queue.put(t)
    for _ in range(args.threads):
        task_queue.put(None)

    logging.info(f"Starting QC V2 (Force Update) on {len(tasks)} sites...")
    
    start_time = time.time()
    processes = []
    
    # 3. Launch NEW Workers
    for i in range(args.threads):
        p = mp.Process(target=qc_worker_v2, 
                       args=(task_queue, results_dict, i, args.channels, args.illum_path))
        p.start()
        processes.append(p)

    # 4. Monitor
    with tqdm(total=len(tasks)) as pbar:
        while True:
            completed = len(results_dict)
            pbar.n = completed
            pbar.refresh()
            if completed >= len(tasks):
                break
            if not any(p.is_alive() for p in processes) and completed < len(tasks):
                logging.error("Workers died unexpectedly.")
                break
            time.sleep(1)

    for p in processes:
        p.join()

    # 5. Save
    logging.info("Merging results...")
    qc_df = pd.DataFrame.from_dict(results_dict, orient='index')
    qc_df = qc_df.sort_index()
    final_df = pd.concat([df, qc_df], axis=1)
    
    final_df.to_csv(args.output, index=False)
    logging.info(f"Complete. Saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-data', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--illum-path', type=str, default=None)
    parser.add_argument('--channels', nargs='+', required=True)
    parser.add_argument('--output', type=str, default='QC_Results_V2.csv')
    parser.add_argument('--threads', type=int, default=24)
    args = parser.parse_args()

    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    main(args)