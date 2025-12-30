import os
import argparse
import logging
import tifffile
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
import scipy.fft
import scipy.stats
from tqdm import tqdm
from queue import Empty

# --- 1. QC Math Functions ---

def calculate_final_metrics(image, channel_name):
    """
    Calculates the QC metrics to match CellProfiler.
    
    Metrics:
    1. Slope_Theoretical_Power: (Magnitude Slope * 2). 
       Expected: ~ -1.5 (Matches CP's -1.4).
    2. Slope_Quantized_8bit: Force 8-bit degradation.
       Expected: ~ -1.4 (Noise floor flattens the slope).
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
        # Use inscribed circle (Nyquist)
        max_r = int(min(h, w) / 2)
        y, x = np.indices((h, w))
        center_y, center_x = h // 2, w // 2
        r_int = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        r_flat = r_int.ravel()

        # --- METRIC A: THEORETICAL POWER (Mag * 2) ---
        # This aligns with the math: log(Power) = 2 * log(Mag)
        # Your previous run showed Mag Slope = -0.75, so this will be -1.5.
        
        # 1. FFT on high-precision float
        img_float = image.astype(float)
        F = scipy.fft.fft2(img_float)
        F_shifted = scipy.fft.fftshift(F)
        
        # 2. Magnitude (Not Squared yet)
        magnitude_spectrum = np.abs(F_shifted)
        m_flat = magnitude_spectrum.ravel()
        
        # 3. Radial Sum of Magnitude
        radial_sum_mag = np.bincount(r_flat, weights=m_flat)
        
        # 4. Slice Valid Range
        if len(radial_sum_mag) > max_r:
            vals_mag = radial_sum_mag[1:max_r+1]
            freqs = np.arange(1, max_r + 1)
        else:
            vals_mag = radial_sum_mag[1:]
            freqs = np.arange(1, len(radial_sum_mag) + 1)
            
        # 5. Calculate Slope & Multiply by 2
        valid_mag = vals_mag > 0
        if np.sum(valid_mag) > 2:
            s_mag, _, _, _, _ = scipy.stats.linregress(np.log(freqs[valid_mag]), np.log(vals_mag[valid_mag]))
            results[f'Slope_Theoretical_Power_{channel_name}'] = s_mag * 2.0
        else:
            results[f'Slope_Theoretical_Power_{channel_name}'] = np.nan

        # --- METRIC B: 8-BIT QUANTIZATION ---
        # This tests if CP is converting to 8-bit, which adds white noise (slope 0)
        # and flattens the result from -2.5 to -1.4.
        
        if image.dtype == np.uint16:
            # Scale 0-65535 down to 0-255
            img_8bit = (image / 256).astype(np.uint8)
            img_quant = img_8bit.astype(float)
        elif image.dtype == np.float32 or image.dtype == np.float64:
            # Scale float min-max to 0-255
            img_norm = (image - image.min()) / (image.max() - image.min())
            img_quant = (img_norm * 255).astype(np.uint8).astype(float)
        else:
            img_quant = image.astype(float)

        # Power Slope on Quantized Image
        F_q = scipy.fft.fft2(img_quant)
        F_q_shift = scipy.fft.fftshift(F_q)
        power_q = np.abs(F_q_shift) ** 2
        
        p_q_flat = power_q.ravel()
        radial_sum_q = np.bincount(r_flat, weights=p_q_flat)
        
        if len(radial_sum_q) > max_r:
            vals_q = radial_sum_q[1:max_r+1]
        else:
            vals_q = radial_sum_q[1:]
            
        valid_q = vals_q > 0
        if np.sum(valid_q) > 2:
            s_q, _, _, _, _ = scipy.stats.linregress(np.log(freqs[valid_q]), np.log(vals_q[valid_q]))
            results[f'Slope_Quantized_8bit_{channel_name}'] = s_q
        else:
            results[f'Slope_Quantized_8bit_{channel_name}'] = np.nan

    except Exception:
        results[f'Slope_Theoretical_Power_{channel_name}'] = np.nan
        results[f'Slope_Quantized_8bit_{channel_name}'] = np.nan

    return results

# --- 2. Worker ---

def qc_producer_worker(task_queue, results_dict, worker_id, channels, illum_path):
    # Print only once
    if worker_id == 0:
        print("--- WORKER STARTED: Calculating Theoretical (Mag*2) & Quantized Slopes ---")

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
                
                metrics = calculate_final_metrics(img, ch_name)
                site_results.update(metrics)
            
            results_dict[index] = site_results

        except Exception as e:
            results_dict[index] = {f"QC_Error": str(e)}

# --- 3. Orchestrator ---

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(f"Reading CSV: {args.load_data}")
    
    df = pd.read_csv(args.load_data)
    
    # Drop existing QC columns to avoid pollution
    cols_to_drop = [c for c in df.columns if 'Slope_' in c or 'ImageQuality_' in c]
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
            import time
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
    parser.add_argument('--output', type=str, default='QC_Results_Final.csv')
    parser.add_argument('--threads', type=int, default=24)
    args = parser.parse_args()

    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    main(args)