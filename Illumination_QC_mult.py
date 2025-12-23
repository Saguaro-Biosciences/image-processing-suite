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

# --- 1. QC Math Functions (Vectorized for Speed) ---

def calculate_qc_metrics(image, bit_depth=16):
    """
    Calculates PowerLogLog Slope and Percent Maximal.
    Uses FFT for frequency analysis and bincount for fast radial averaging.
    """
    results = {}
    
    # 1. Percent Maximal (Saturation)
    max_val = (2**bit_depth) - 1
    results['PercentMaximal'] = (np.sum(image >= max_val) / image.size) * 100

    # 2. PowerLogLog Slope (Blur/Focus)
    try:
        h, w = image.shape
        # Use fast FFT
        F = fftpack.fft2(image.astype(np.float32))
        F_shifted = fftpack.fftshift(F)
        psd2D = np.abs(F_shifted)**2
        
        # Radial averaging
        y, x = np.indices(psd2D.shape)
        center = (h // 2, w // 2)
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2).astype(int)

        tbin = np.bincount(r.ravel(), psd2D.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / nr
        
        # Linear fit in log-log space
        max_r = min(center)
        indices = np.arange(1, max_r)
        log_r = np.log(indices)
        log_power = np.log(radial_profile[1:max_r] + 1e-10)
        
        slope, _ = np.polyfit(log_r, log_power, 1)
        results['PowerLogLogSlope'] = slope
    except:
        results['PowerLogLogSlope'] = np.nan

    return results

# --- 2. Parallel Worker ---

def qc_producer_worker(task_queue, results_dict, worker_id, channels, illum_path):
    """
    Worker designed for high-thread-count CPUs.
    Loads -> Corrects -> QCs -> Stores.
    """
    # Load illumination corrections once per worker to save RAM
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
                img = tifffile.imread(path)
                
                # Apply Illumination Correction
                if corrections is not None:
                    img = img / corrections[i]
                
                # Run QC
                metrics = calculate_qc_metrics(img)
                
                # Store with channel-specific keys
                for m_name, val in metrics.items():
                    site_results[f"QC_{m_name}_{ch_name}"] = val
            
            results_dict[index] = site_results

        except Exception as e:
            logging.error(f"Worker-{worker_id} failed on index {index}: {e}")
            results_dict[index] = {f"QC_Error": str(e)}

# --- 3. Orchestrator ---

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load the Load_data.csv
    df = pd.read_csv(args.load_data)
    channel_cols = [f'FileName_{c}' for c in args.channels]
    
    tasks = [
        (idx, [os.path.join(args.data_path, row[col]) for col in channel_cols])
        for idx, row in df.iterrows()
    ]

    # Multiprocessing setup
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

    # Progress bar monitoring
    with tqdm(total=len(tasks)) as pbar:
        while len(results_dict) < len(tasks):
            pbar.n = len(results_dict)
            pbar.refresh()
            time.sleep(1)

    for p in processes:
        p.join()

    # Merge results back to dataframe
    qc_df = pd.DataFrame.from_dict(results_dict, orient='index')
    final_df = pd.concat([df, qc_df], axis=1)
    
    final_df.to_csv(args.output, index=False)
    logging.info(f"Complete! Processed {len(tasks)} sites in {time.time()-start_time:.2f}s. Saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-data', type=str, required=True, help='Path to Load_data.csv')
    parser.add_argument('--data-path', type=str, required=True, help='Base directory for images')
    parser.add_argument('--illum-path', type=str, default=None, help='Folder containing _illum.npy files')
    parser.add_argument('--channels', nargs='+', required=True, help='List of channel names (e.g. DNA Mito ER)')
    parser.add_argument('--output', type=str, default='QC_Results.csv')
    parser.add_argument('--threads', type=int, default=24)
    
    # Required for Threadripper/Unix systems to handle memory cleanly
    mp.set_start_method('spawn', force=True)
    main(parser.parse_args())