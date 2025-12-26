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

# --- 1. QC Math Functions (CP Matched: PowerLogLogSlope + PercentMaximal) ---

def calculate_cp_identical_slope(image, channel_name):
    """
    Calculates QC metrics replicating CellProfiler's MeasureImageQuality logic.
    
    1. Percent Maximal: Checks for saturation (specifically for 16-bit images).
    2. PowerLogLogSlope: FFT -> Power Spectrum -> Radial Sum -> Log-Log Slope.
    """
    results = {}
    
    # --- 1. Percent Maximal (Saturation for 16-bit) ---
    # CellProfiler MeasureImageQuality "PercentMaximal" is the percentage of pixels 
    # at the maximum possible value of the bit-depth.
    
    # Determine max value based on dtype, defaulting to 16-bit (65535) if unsure
    if image.dtype == np.uint8:
        max_val = 255
    elif image.dtype == np.uint16:
        max_val = 65535
    else:
        # Fallback for floating point or other types: assume standard 16-bit range or 1.0
        max_val = 65535 if image.max() > 1.0 else 1.0
        
    num_saturated = np.sum(image >= max_val)
    pct_max = (num_saturated / image.size) * 100
    results[f'ImageQuality_PercentMaximal_{channel_name}'] = pct_max

    # --- 2. PowerLogLogSlope (CellProfiler Logic) ---
    try:
        # Ensure image is float for FFT
        image_float = image.astype(float)
        
        # A. Compute the 2D Fast Fourier Transform
        f_transform = scipy.fft.fft2(image_float)
        
        # B. Shift zero-frequency component to center
        f_shifted = scipy.fft.fftshift(f_transform)
        
        # C. Calculate Power Spectrum (Magnitude squared)
        # Note: CellProfiler calculates Power = |F|^2
        power_spectrum = np.abs(f_shifted) ** 2
        
        # D. Radial Integration Setup
        h, w = image.shape
        center_y, center_x = h // 2, w // 2
        
        y, x = np.indices((h, w))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Convert radii to integer indices for binning
        r_int = r.astype(int)
        
        # Define max radius (Nyquist limit / half-width of smallest dimension)
        # This matches standard CellProfiler behavior which avoids corner artifacts.
        max_r = min(h, w) // 2
        
        # E. Sum power spectrum intensity within each ring
        # We integrate from radius 1 to max_r (ignoring DC component at 0)
        tbin = np.arange(1, max_r + 1)
        
        # Efficient labeled sum using scipy.ndimage
        radial_power = scipy.ndimage.sum(power_spectrum, r_int, index=tbin)
        
        # F. Filter valid values for Log-Log
        # Ensure we don't take log of zero
        valid_mask = radial_power > 0
        if np.sum(valid_mask) > 2: # Need at least a few points for regression
            freq_log = np.log(tbin[valid_mask])
            power_log = np.log(radial_power[valid_mask])
            
            # G. Linear Regression
            # Returns: slope, intercept, r_value, p_value, std_err
            slope, _, _, _, _ = scipy.stats.linregress(freq_log, power_log)
            results[f'ImageQuality_PowerLogLogSlope_{channel_name}'] = slope
        else:
            results[f'ImageQuality_PowerLogLogSlope_{channel_name}'] = np.nan
            
    except Exception as e:
        # logging.error(f"Error calculating slope for {channel_name}: {e}") # Optional debug
        results[f'ImageQuality_PowerLogLogSlope_{channel_name}'] = np.nan

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
                
                # Apply Illumination Correction if loaded
                if corrections is not None:
                    # Ensure dimensions match before division
                    if img.shape == corrections[i].shape:
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
    
    # Construct expected column names for file paths
    channel_cols = [f'FileName_{c}' for c in args.channels]
    
    # verify columns exist
    for col in channel_cols:
        if col not in df.columns:
            logging.error(f"Column {col} not found in CSV. Check your --channels argument.")
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
    logging.info(f"Complete! Processed {len(tasks)} sites in {time.time()-start_time:.2f}s. Saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-data', type=str, required=True, help='Path to CSV containing image filenames')
    parser.add_argument('--data-path', type=str, required=True, help='Base path where images are stored')
    parser.add_argument('--illum-path', type=str, default=None, help='Path to folder containing numpy illumination functions')
    parser.add_argument('--channels', nargs='+', required=True, help='List of channel names (e.g. DAPI GFP)')
    parser.add_argument('--output', type=str, default='QC_Results.csv')
    parser.add_argument('--threads', type=int, default=24)
    args = parser.parse_args()

    # Safety for multiprocessing in certain environments
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    main(args)