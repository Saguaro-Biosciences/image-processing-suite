import os
import argparse
import logging
import time
import tifffile
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
from scipy import fftpack
from tqdm import tqdm
from queue import Empty

# --- 1. QC Math Functions (CellProfiler Matched) ---

def calculate_cp_metrics(image, channel_name, bit_depth=16):
    """
    Calculates QC metrics matching CellProfiler's ImageQuality module.
    
    ADJUSTMENTS FOR -1.4 RANGE:
    1. No Hanning Window (matches CP's raw FFT).
    2. Magnitude Spectrum (matches the slope scale).
    """
    results = {}
    
    # --- A. Percent Maximal (Saturation) ---
    max_val = (2**bit_depth) - 1
    pct_max = (np.sum(image >= max_val) / image.size) * 100
    # Matching the naming convention for Corrected images
    results[f'ImageQuality_PercentMaximal_Corr{channel_name}'] = pct_max

    # --- B. PowerLogLog Slope (Focus/Blur) ---
    try:
        # 1. Convert to float and remove DC component (mean intensity)
        # We do NOT apply a window here, to match CellProfiler's spectral leakage
        img_float = image.astype(np.float32)
        img_float -= np.mean(img_float)
        
        # 2. FFT & Magnitude Calculation
        # Use Magnitude (Abs) instead of Power (Abs^2) to get the -1.0 to -1.5 range
        F = fftpack.fft2(img_float)
        F_shifted = fftpack.fftshift(F)
        magnitude_spectrum = np.abs(F_shifted)
        
        # 3. Radial Averaging
        h, w = img_float.shape
        y, x = np.indices(magnitude_spectrum.shape)
        center = (h // 2, w // 2)
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2).astype(int)

        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        # Avoid division by zero
        radial_profile = tbin / (nr + 1e-10)
        
        # 4. Select Frequency Range (Standard CP behavior)
        # Avoid the very bottom (structure) and very top (camera noise)
        max_r = min(center)
        start_idx = max(1, int(max_r * 0.1)) 
        end_idx = int(max_r * 0.5)
        
        if end_idx <= start_idx:
            end_idx = len(radial_profile) - 1

        # 5. Log-Log Fit
        indices = np.arange(start_idx, end_idx)
        normalized_freqs = indices / max_r
        
        log_x = np.log(normalized_freqs)
        log_y = np.log(radial_profile[start_idx:end_idx] + 1e-10)
        
        if len(log_x) > 2:
            slope, _ = np.polyfit(log_x, log_y, 1)
            results[f'ImageQuality_PowerLogLogSlope_Corr{channel_name}'] = slope
        else:
            results[f'ImageQuality_PowerLogLogSlope_Corr{channel_name}'] = np.nan
            
    except Exception:
        results[f'ImageQuality_PowerLogLogSlope_Corr{channel_name}'] = np.nan

    return results

# --- 2. Parallel Worker ---

def qc_producer_worker(task_queue, results_dict, worker_id, channels, illum_path):
    # Load illumination corrections once
    corrections = None
    if illum_path:
        try:
            # Loads "DNA_illum.npy", etc.
            corrections = [np.load(os.path.join(illum_path, f"{c}_illum.npy")) for c in channels]
        except Exception as e:
            logging.error(f"Worker-{worker_id} could not load illum files: {e}")

    while True:
        task = task_queue.get()
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
                
                # Apply Illumination Correction
                if corrections is not None:
                    img = img / corrections[i]
                
                # Run QC
                metrics = calculate_cp_metrics(img, ch_name)
                site_results.update(metrics)
            
            results_dict[index] = site_results

        except Exception as e:
            logging.error(f"Worker-{worker_id} failed on index {index}: {e}")
            results_dict[index] = {f"QC_Error": str(e)}

# --- 3. Orchestrator ---

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(f"Reading input CSV: {args.load_data}")
    df = pd.read_csv(args.load_data)
    
    channel_cols = [f'FileName_{c}' for c in args.channels]
    
    # Validate columns exist
    for col in channel_cols:
        if col not in df.columns:
            logging.error(f"Column {col} not found in CSV. Check --channels argument.")
            return

    tasks = [
        (idx, [os.path.join(args.data_path, row[col]) for col in channel_cols])
        for idx, row in df.iterrows()
    ]

    manager = mp.Manager()
    results_dict = manager.dict()
    task_queue = mp.Queue()

    for t in tasks:
        task_queue.put(t)
    for _ in range(args.threads):
        task_queue.put(None)

    logging.info(f"Starting QC on {len(tasks)} sites using {args.threads} threads...")
    
    start_time = time.time()
    processes = []
    
    for i in range(args.threads):
        p = mp.Process(target=qc_producer_worker, 
                       args=(task_queue, results_dict, i, args.channels, args.illum_path))
        p.start()
        processes.append(p)

    with tqdm(total=len(tasks)) as pbar:
        while len(results_dict) < len(tasks):
            pbar.n = len(results_dict)
            pbar.refresh()
            time.sleep(1)

    for p in processes:
        p.join()

    logging.info("Merging results...")
    qc_df = pd.DataFrame.from_dict(results_dict, orient='index')
    qc_df = qc_df.sort_index()
    
    final_df = pd.concat([df, qc_df], axis=1)
    
    final_df.to_csv(args.output, index=False)
    logging.info(f"Complete! Processed {len(tasks)} sites in {time.time()-start_time:.2f}s.")
    logging.info(f"Results saved to: {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run High-Speed QC matching CellProfiler Metrics.")
    parser.add_argument('--load-data', type=str, required=True, help='Path to Load_data.csv')
    parser.add_argument('--data-path', type=str, required=True, help='Base directory for images')
    parser.add_argument('--illum-path', type=str, default=None, help='Folder containing _illum.npy files')
    parser.add_argument('--channels', nargs='+', required=True, help='List of channel names (e.g. DNA CL488R)')
    parser.add_argument('--output', type=str, default='QC_Results.csv', help='Output CSV path')
    parser.add_argument('--threads', type=int, default=24, help='Number of threads (default: 24)')
    
    args = parser.parse_args()

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main(args)