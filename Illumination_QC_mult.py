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

# --- 1. QC Math Functions (CP Matched: Magnitude + Full Range) ---

def calculate_cp_identical_slope(image, channel_name, bit_depth=16):
    # Auto-detect bit depth max if needed
    if image.dtype == np.uint8: max_val = 255
    elif image.dtype == np.uint16: max_val = 65535
    else: max_val = (2**bit_depth) - 1

    results = {}
    
    # 1. Percent Maximal
    pct_max = (np.sum(image >= max_val) / image.size) * 100
    results[f'ImageQuality_PercentMaximal_Corr{channel_name}'] = pct_max

    # 2. PowerLogLog Slope
    try:
        img_float = image.astype(np.float32)
        F = fftpack.fft2(img_float)
        F_shifted = fftpack.fftshift(F)
        magnitude_spectrum = np.abs(F_shifted)
        
        h, w = img_float.shape
        y, x = np.indices(magnitude_spectrum.shape)
        center = (h // 2, w // 2)
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2).astype(int)

        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / (nr + 1e-10)
        
        # FULL RANGE (1 to Nyquist)
        max_r = min(center)
        start_idx = 1
        end_idx = int(max_r)
        
        if end_idx > start_idx:
            freqs = np.arange(start_idx, end_idx)
            log_x = np.log(freqs)
            log_y = np.log(radial_profile[start_idx:end_idx] + 1e-10)
            slope, _ = np.polyfit(log_x, log_y, 1)
            results[f'ImageQuality_PowerLogLogSlope_Corr{channel_name}'] = slope
        else:
            results[f'ImageQuality_PowerLogLogSlope_Corr{channel_name}'] = np.nan
            
    except Exception:
        results[f'ImageQuality_PowerLogLogSlope_Corr{channel_name}'] = np.nan

    return results

# --- 2. Parallel Worker ---

def qc_producer_worker(task_queue, results_dict, worker_id, channels, illum_path):
    corrections = None
    if illum_path:
        try:
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
                
                if corrections is not None:
                    img = img / corrections[i]
                
                metrics = calculate_cp_identical_slope(img, ch_name)
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
    logging.info(f"Complete! Processed {len(tasks)} sites in {time.time()-start_time:.2f}s. Saved to {args.output}")

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