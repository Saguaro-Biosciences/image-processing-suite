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

# --- 1. QC Math Functions (V5 Sum + CP Clipping) ---

def calculate_cp_metrics_exact(image, channel_name):
    """
    Replicates CellProfiler MeasureImageQuality EXACTLY.
    
    CRITICAL STEPS FROM PIPELINE:
    1. INPUT: Image must be scaled 0-1 and CLIPPED at 1.0 before FFT.
       (The clipping adds high-freq artifacts that flatten the slope).
    2. MATH: Radial SUM (Not Mean).
    3. RANGE: Nyquist Limit (No corners).
    """
    results = {}
    
    # --- Percent Maximal (Saturation) ---
    # Since we are now working in 0-1 float space due to the pipeline logic:
    # "PercentMaximal" in CP on a scaled image checks for values = 1.0
    pct_max = (np.sum(image >= 1.0) / image.size) * 100
    results[f'ImageQuality_PercentMaximal_{channel_name}'] = pct_max

    # --- PowerLogLogSlope ---
    try:
        # 1. FFT & Power Spectrum (On the CLIPPED image)
        F = scipy.fft.fft2(image)
        F_shifted = scipy.fft.fftshift(F)
        power_spectrum = np.abs(F_shifted) ** 2
        
        # 2. Radial Map
        h, w = image.shape
        center_y, center_x = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r_int = r.astype(int)
        
        # 3. Define Max Radius (Nyquist)
        max_r = int(min(h, w) / 2)
        
        # 4. Radial SUMMATION
        r_flat = r_int.ravel()
        p_flat = power_spectrum.ravel()
        radial_sum = np.bincount(r_flat, weights=p_flat)
        
        # 5. Filter Valid Range [1, max_r]
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
            
    except Exception:
        results[f'ImageQuality_PowerLogLogSlope_{channel_name}'] = np.nan

    return results

# --- 2. Parallel Worker (With Scaling & Clipping) ---

def qc_producer_worker(task_queue, results_dict, worker_id, channels, illum_path):
    if worker_id == 0:
        print("--- WORKER STARTED: Simulating CP Pipeline (Scale -> Illum -> Clip -> Sum) ---")

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

                # A. Load Raw Image
                img_raw = tifffile.imread(path)
                
                # B. REPLICATE "LoadData" SCALING
                # CellProfiler scales 16-bit images to 0-1 automatically
                if img_raw.dtype == np.uint16:
                    img_float = img_raw.astype(np.float32) / 65535.0
                elif img_raw.dtype == np.uint8:
                    img_float = img_raw.astype(np.float32) / 255.0
                else:
                    img_float = img_raw.astype(np.float32)

                # C. REPLICATE "CorrectIlluminationApply"
                if corrections and corrections[i] is not None:
                    # Resize illum if needed (CP handles this, we must ensure safety)
                    illum = corrections[i]
                    if img_float.shape == illum.shape:
                        img_float = img_float / illum
                
                # D. CRITICAL: REPLICATE CLIPPING
                # "Set output image values greater than 1 equal to 1?: Yes"
                # This hard clip introduces the high-freq harmonics that flatten the slope.
                img_float = np.clip(img_float, 0.0, 1.0)
                
                # E. Measure
                metrics = calculate_cp_metrics_exact(img_float, ch_name)
                site_results.update(metrics)
            
            results_dict[index] = site_results

        except Exception as e:
            results_dict[index] = {f"QC_Error": str(e)}

# --- 3. Orchestrator ---

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(f"Reading CSV: {args.load_data}")
    logging.info("!!! REPLICATING CP PIPELINE: SCALING & CLIPPING ENABLED !!!")
    
    df = pd.read_csv(args.load_data)
    
    # RENAME KEYS TEMPORARILY TO ENSURE FRESH COLUMNS (Optional but recommended)
    # The worker function above uses the standard names, but you can change them if needed.
    
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
    parser.add_argument('--output', type=str, default='QC_Results_PipelineMatch.csv')
    parser.add_argument('--threads', type=int, default=24)
    args = parser.parse_args()

    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    main(args)