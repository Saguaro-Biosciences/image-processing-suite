import os
import argparse
import logging
import tifffile
import numpy as np
import pandas as pd
import scipy.fftpack
import scipy.ndimage
import scipy.stats
import concurrent.futures
from tqdm import tqdm

# ==========================================
# 1. ARGUMENT PARSING
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="CellProfiler-Matched Image QC (Production)")
    parser.add_argument('--load-data', type=str, required=True, help="Path to input CSV (LoadData format)")
    parser.add_argument('--data-path', type=str, required=True, help="Base path for image files")
    parser.add_argument('--illum-path', type=str, default=None, help="Folder containing .npy illumination functions")
    parser.add_argument('--channels', nargs='+', required=True, help="List of channel names (e.g. CL488 CL568)")
    parser.add_argument('--output', type=str, default='QC_Results.csv', help="Path for output CSV")
    parser.add_argument('--threads', type=int, default=24, help="Number of threads for parallel processing")
    return parser.parse_args()

# ==========================================
# 2. EXACT MATH FUNCTIONS (FROM SOURCE)
# ==========================================

def rps(img):
    """
    Exact implementation of centrosome.radial_power_spectrum.rps
    Source: CellProfiler/Centrosome
    """
    assert img.ndim == 2
    
    # 1. Quadrant Folding (Radii calculation for unshifted FFT)
    radii2 = (np.arange(img.shape[0]).reshape((img.shape[0], 1)) ** 2) + (
        np.arange(img.shape[1]) ** 2
    )
    radii2 = np.minimum(radii2, np.flipud(radii2))
    radii2 = np.minimum(radii2, np.fliplr(radii2))
    
    # 2. Truncation (The Critical Difference from standard PSD)
    # Only analyzes low frequencies (up to width/8).
    # For a 256px image, maxwidth is 32. This excludes high-freq noise.
    maxwidth = (min(img.shape[0], img.shape[1]) / 8.0)
    
    # 3. Intensity Invariant Normalization
    # Updated for NumPy 2.0 compatibility (np.ptp instead of img.ptp)
    if np.ptp(img) > 0:
        img = img / np.median(np.abs(img - np.mean(img)))
        
    # 4. FFT (DC removed)
    # fft2 returns unshifted array (DC at 0,0)
    mag = np.abs(scipy.fftpack.fft2(img - np.mean(img)))
    power = mag ** 2
    
    # 5. Binning
    radii = np.floor(np.sqrt(radii2)).astype(int) + 1
    labels = np.arange(2, np.floor(maxwidth)).astype(int).tolist() # Skip DC (0) and Freq 1
    
    if len(labels) > 0:
        # Sum power in each frequency ring
        magsum = scipy.ndimage.sum(mag, radii, labels)
        powersum = scipy.ndimage.sum(power, radii, labels)
        return np.array(labels), np.array(magsum), np.array(powersum)
    
    return [2], [0], [0]


def calculate_saturation_cp_exact(image, mask=None):
    """
    Exact implementation of CellProfiler saturation logic.
    Calculates PercentMaximal based on the image's own maximum value.
    """
    # Handle Masking if present
    if mask is not None:
        pixel_data = image[mask]
    else:
        pixel_data = image

    pixel_count = pixel_data.size
    
    if pixel_count == 0:
        return 0.0
    
    # Logic: Count pixels equal to the maximum value found in the array
    max_val = np.max(pixel_data)
    number_pixels_maximal = np.sum(pixel_data == max_val)
    
    percent_maximal = (100.0 * float(number_pixels_maximal) / float(pixel_count))
    
    return percent_maximal


def calculate_qc_metrics(image, channel_name):
    """
    Wrapper to run both RPS and Saturation checks.
    """
    results = {}
    
    # --- 1. PowerLogLogSlope (Using RPS) ---
    try:
        radii, magsum, powersum = rps(image)
        
        valid = powersum > 0
        if np.sum(valid) > 2:
            # CellProfiler uses Least Squares (lstsq), equivalent to linregress for 1D
            slope, _, _, _, _ = scipy.stats.linregress(np.log(radii[valid]), np.log(powersum[valid]))
            results[f'ImageQuality_PowerLogLogSlope_{channel_name}'] = slope
        else:
            results[f'ImageQuality_PowerLogLogSlope_{channel_name}'] = 0.0
    except Exception:
        results[f'ImageQuality_PowerLogLogSlope_{channel_name}'] = np.nan

    # --- 2. PercentMaximal (Using CP Logic) ---
    try:
        pct_max = calculate_saturation_cp_exact(image)
        results[f'ImageQuality_PercentMaximal_{channel_name}'] = pct_max
    except Exception:
        results[f'ImageQuality_PercentMaximal_{channel_name}'] = np.nan

    return results

# ==========================================
# 3. WORKER & ORCHESTRATOR
# ==========================================

def process_site(site_data):
    """
    Worker function to process a single site (row).
    """
    index, paths, channels, illum_cache = site_data
    site_results = {}

    for i, (path, ch_name) in enumerate(zip(paths, channels)):
        try:
            if not os.path.exists(path):
                site_results[f"QC_Error_{ch_name}"] = "File Not Found"
                continue

            # Load Image
            img = tifffile.imread(path).astype(float)
            
            # Apply Illumination Correction
            if illum_cache and illum_cache[i] is not None:
                if img.shape == illum_cache[i].shape:
                    img = img / illum_cache[i]
                else:
                    # Fallback to raw if shape mismatch
                    pass 
            
            # Calculate Metrics
            metrics = calculate_qc_metrics(img, ch_name)
            site_results.update(metrics)
            
        except Exception as e:
            site_results[f"QC_Error_{ch_name}"] = str(e)
            
    return index, site_results


def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(f"Reading CSV: {args.load_data}")
    
    df = pd.read_csv(args.load_data)
    
    # Clean up old QC columns
    cols_to_drop = [c for c in df.columns if 'ImageQuality_' in c or 'QC_Error' in c]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    channel_cols = [f'FileName_{c}' for c in args.channels]
    
    # --- PRE-LOAD ILLUMINATION ---
    logging.info("Loading Illumination Correction Files...")
    illum_cache = []
    if args.illum_path:
        for c in args.channels:
            # Try standard naming patterns
            p1 = os.path.join(args.illum_path, f"{c}_illum.npy")
            p2 = os.path.join(args.illum_path, f"Illum{c}.npy")
            
            if os.path.exists(p1):
                illum_cache.append(np.load(p1))
                logging.info(f"  Loaded {c}_illum.npy")
            elif os.path.exists(p2):
                illum_cache.append(np.load(p2))
                logging.info(f"  Loaded Illum{c}.npy")
            else:
                illum_cache.append(None)
                logging.warning(f"  Warning: No illumination file found for {c}")
    else:
        illum_cache = [None] * len(args.channels)

    # --- PREPARE TASKS ---
    tasks = []
    for idx, row in df.iterrows():
        paths = [os.path.join(args.data_path, row[col]) for col in channel_cols]
        tasks.append((idx, paths, args.channels, illum_cache))

    logging.info(f"Starting processing on {len(tasks)} sites with {args.threads} threads...")
    
    # --- EXECUTE PARALLEL ---
    results_dict = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_site, t): t[0] for t in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            idx, res = future.result()
            results_dict[idx] = res

    # --- MERGE & SAVE ---
    logging.info("Merging results...")
    qc_df = pd.DataFrame.from_dict(results_dict, orient='index').sort_index()
    
    final_df = pd.concat([df, qc_df], axis=1)
    
    final_df.to_csv(args.output, index=False)
    logging.info(f"Done! Saved to {args.output}")

if __name__ == '__main__':
    main()