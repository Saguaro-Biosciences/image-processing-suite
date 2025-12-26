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
    
    CRITICAL MATH:
    - Uses RADIAL SUM (not Mean).
    - Slope(Sum) = Slope(Mean) + 1.0
    - If Mean slope is -2.5, Sum slope will be -1.5 (Matching CellProfiler).
    """
    results = {}
    
    # --- Percent Maximal (Saturation) ---
    if image.dtype == np.uint8: max_val = 255
    else: max_val = 65535
        
    pct_max = (np.sum(image >= max_val) / image.size) * 100
    results[f'ImageQuality_PercentMaximal_{channel_name}'] = pct_max

    # --- PowerLogLogSlope (Sum + No Corners) ---
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
        r_int = r.astype(int)
        
        # 3. Define Max Radius (EXCLUDE CORNERS)
        # Matches CellProfiler: Stop at the edge of the circle.
        max_r = int(min(h, w) / 2)
        
        # 4. Radial SUMMATION
        # bincount with weights calculates the SUM.
        # DO NOT divide by pixel counts. This is the +1.0 slope difference.
        r_flat = r_int.ravel()
        p_flat = power_spectrum.ravel()
        
        radial_sum = np.bincount(r_flat, weights=p_flat)
        
        # 5. Filter Valid Range
        if len(radial_sum) > max_r:
            radial_sum = radial_sum[1:max_r+1]
            freqs = np.arange(1, max_r + 1)
        else:
            radial_sum = radial_sum[1:]
            freqs = np.arange(1, len(radial_sum) + 1)
        
        # 6. Log-Log Slope Calculation
        valid_mask = radial_sum > 0
        if np.sum(valid_mask) > 2:
            freq_log = np.log(freqs[valid_mask])
            power_log = np.log(radial_sum[valid_mask])
            
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
        print(f"--- WORKER STARTED: Running FIXED Summation Logic ---")

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
    logging.info("!!! VERIFICATION: Running 'Sum' Logic (Expect Slope ~ -1.5) !!!")
    
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