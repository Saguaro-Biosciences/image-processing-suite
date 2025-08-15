import os
import argparse
import logging
import tifffile
import pandas as pd
import numpy as np
import torch
import s3fs
import multiprocessing
import concurrent.futures
from PIL import Image
from tqdm import tqdm

# Import scientific libraries
from skimage.measure import regionprops
from cellpose import models
from transformers import AutoImageProcessor, AutoModel

# --- 1. Setup Logging ---
# Configure logging to show timestamps and informational messages.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Define function for nornalization and 8bit scaling
def scale_to_8bit(image_16bit):
    """
    Intelligently scales a 16-bit image to 8-bit by stretching
    the actual dynamic range of the image to the full 0-255 range.
    """
    min_val, max_val = image_16bit.min(), image_16bit.max()
    if max_val == min_val:
        return np.zeros(image_16bit.shape, dtype=np.uint8)
    
    # Stretch the relevant intensity range to 0-255
    scaled_image = 255.0 * (image_16bit.astype(np.float32) - min_val) / (max_val - min_val)
    
    return scaled_image.astype(np.uint8)

# --- 2. Define Worker Functions for Multiprocessing ---
# These functions must be defined at the top level of the script for multiprocessing to work.

def init_worker(model_name, device_name, cellpose_model_path):
    """
    Initializes models and processor. The S3 filesystem is no longer needed here.
    """
    global worker_cell_model, worker_model, worker_processor, worker_device

    logging.info(f"Initializing worker on process ID: {os.getpid()}")
    
    # Setup device (GPU or CPU)
    worker_device = torch.device(device_name)
    
    # Load Cellpose model
    worker_cell_model = models.CellposeModel(gpu=(worker_device.type == 'cuda'), pretrained_model=cellpose_model_path)
    
    # Load Hugging Face model and processor
    worker_processor = AutoImageProcessor.from_pretrained(model_name)
    worker_model = AutoModel.from_pretrained(model_name).to(worker_device)
    worker_model.eval()

def process_site(site_image_paths, box_size, feature_length):
    """
    Processes a set of 4 channel images for a single site.
    This version includes the correct 16-bit to 8-bit scaling workflow.
    """
    # Access the globally initialized models and objects for this worker
    global worker_cell_model, worker_model, worker_processor, worker_device, worker_s3fs

    try:

        all_channels = [tifffile.imread(local_path) for local_path in site_image_paths]
        
        image_4ch = np.stack(all_channels, axis=-1)


        # Run Cellpose to get masks
        masks, _, _ = worker_cell_model.eval([image_4ch], diameter=100, channels=None)
        props = regionprops(masks[0])
        
        if not props:
            return np.zeros((4, feature_length))

        # Process each cell found in the site
        all_cell_crops = []
        h, w, _ = image_4ch.shape
        half_box = box_size // 2

        for prop in props:
            # ... (cropping logic is unchanged)
            y_center, x_center = map(int, prop.centroid)
            target_id = prop.label
            y1, y2 = max(0, y_center - half_box), min(h, y_center + half_box)
            x1, x2 = max(0, x_center - half_box), min(w, x_center + half_box)
            
            mask_crop = masks[0][y1:y2, x1:x2]
            binary_mask = (mask_crop == target_id)[:, :, np.newaxis]
            
            cell_crop_4ch = image_4ch[y1:y2, x1:x2, :]
            masked_cell_crop = cell_crop_4ch * binary_mask
            
            pad_h = box_size - masked_cell_crop.shape[0]
            pad_w = box_size - masked_cell_crop.shape[1]
            padded_crop = np.pad(masked_cell_crop, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
            all_cell_crops.append(padded_crop)

        # Extract features for each cell
        all_stacked_features = []
        for cell_crop in all_cell_crops:
            channel_features = []
            for ch in range(cell_crop.shape[2]):
                
                ### --- START: UPDATED WORKFLOW --- ###

                # 1. Get the raw 16-bit data for this single channel
                raw_16bit_channel = cell_crop[:, :, ch]
                
                if np.all(raw_16bit_channel == 0):
                    channel_features.append(np.zeros(feature_length, dtype=np.float32))
                    continue

                # 2. Perform intensity scaling to convert to a scaled rich 8-bit image
                scaled_8bit_channel = scale_to_8bit(raw_16bit_channel)
                
                # 3. Convert to a 3-channel RGB PIL Image for the processor
                pil_image = Image.fromarray(scaled_8bit_channel).convert("RGB")
                inputs = worker_processor(images=pil_image,return_tensors="pt").to(worker_device)
                
                ### --- END: UPDATED WORKFLOW --- ###

                with torch.no_grad():
                    outputs = worker_model(**inputs)
                features = outputs.pooler_output.cpu().numpy().squeeze()
                channel_features.append(features)
                
            all_stacked_features.append(np.stack(channel_features, axis=0))

        if not all_stacked_features:
            return np.zeros((4, feature_length))

        # Return the mean feature profile for all cells in this site
        return np.mean(np.stack(all_stacked_features, axis=0), axis=0)

    except Exception as e:
        logging.error(f"Error processing site {site_image_paths[0]}: {e}")
        return np.zeros((4, feature_length))

# --- 3. Main Execution Block ---
def main(args):
    """
    Main function to orchestrate the data loading, processing, and saving.
    """
    logging.info(f"Starting analysis with parameters: {args}")

    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
        logging.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        logging.warning("Multiprocessing start method already set.")

    # --- Load Data ---
    s3_input_path = f"s3://{args.bucket_input}/{args.load_data_key}"
    logging.info(f"Reading load_data CSV from {s3_input_path}")
    try:
        load_data = pd.read_csv(s3_input_path)
    except Exception as e:
        logging.error(f"Failed to read input CSV from S3. Error: {e}")
        return # Exit if we can't load the main file
    s3_input_path = f"s3://{args.bucket_input}/{args.meta_data_key}"
    logging.info(f"Reading load_data CSV from {s3_input_path}")
    try:
        meta_data = pd.read_csv(s3_input_path)
    except Exception as e:
        logging.error(f"Failed to read meta-data CSV from S3. Error: {e}")
        return # Exit if we can't load the main file


    # --- Prepare Tasks for Parallel Processing ---
    # These columns should contain the relative paths to the TIFF images
    channel_columns = ['FileName_CL488Y', 'FileName_CL640', 'FileName_DNA', 'FileName_CL488R']
    
    # Create a list of tasks. Each task is a list of full S3 paths for one site.
    image_dir = os.path.dirname(args.load_data_key) # Assumes images are in the same S3 "folder"
    tasks = [
        [f"/home/ubuntu/data/{row[c]}" for c in channel_columns]
        for _, row in load_data.iterrows()
    ]
    logging.info(f"Prepared {len(tasks)} sites for processing.")

    # --- Run Parallel Processing ---
    logging.info(f"Starting parallel feature extraction with {args.max_workers} workers...")
    
    # Constants for models
    MODEL_NAME = "timm/tf_efficientnetv2_xl.in21k"
    CELLPOSE_MODEL = 'nuclei'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FEATURE_LENGTH = 1280
    
    site_results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=init_worker,
        initargs=(MODEL_NAME, DEVICE, CELLPOSE_MODEL)
    ) as executor:
        # Create a list of arguments for each task
        # The map function requires an iterable of arguments for each parameter
        task_args = [(task, args.box_size, FEATURE_LENGTH) for task in tasks]
        
        # Use executor.map to apply the function to all tasks
        # The *zip(*task_args) correctly unpacks the arguments for the map function
        results_iterator = executor.map(process_site, *zip(*task_args))
        
        # Use tqdm to show a progress bar in the console
        site_results = list(tqdm(results_iterator, total=len(tasks), desc="Processing Sites"))

    load_data['mean_features'] = site_results
    logging.info("Parallel processing complete. Site-level features extracted.")

    # --- Aggregate Data to Well Level ---
    logging.info("Aggregating data to well level...")
    metadata_cols = ["Metadata_Well","Metadata_Timepoint","Metadata_Plate"]
    df_subset = load_data[metadata_cols + ['mean_features']]

    agg_functions = {
        'mean_features': lambda arrays: np.mean(np.stack(arrays.values), axis=0)
    }
    for col in metadata_cols:
        if col != 'Metadata_Well':
            agg_functions[col] = 'first'

    well_level_data = df_subset.groupby('Metadata_Well').agg(agg_functions).reset_index()

    meta_data=pd.merge(
        left=well_level_data,
        right=meta_data,
        on=['Metadata_Well','Metadata_Plate'],
        how='inner' 
    )

    # --- Save Final Results ---
    s3_output_path = f"s3://{args.bucket_output}/{args.out_data_path}"
    logging.info(f"Saving final aggregated data to {s3_output_path}")
    well_level_data.to_parquet(s3_output_path, engine='pyarrow')
    
    logging.info("Script finished successfully.")


if __name__ == '__main__':
    # --- 4. Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Run the cell image analysis pipeline.")
    
    parser.add_argument('--box-size', type=int, default=200, help='Size of the square box to crop around each cell.')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of parallel worker processes.')
    parser.add_argument('--bucket-input', type=str, required=True, help='Name of the S3 bucket for input data.')
    parser.add_argument('--bucket-output', type=str, required=True, help='Name of the S3 bucket for output results.')
    parser.add_argument('--load-data-key', type=str, required=True, help='S3 key (path within bucket) to the load_data.csv file.')
    parser.add_argument('--meta-data-key', type=str, required=True, help='S3 key (path within bucket) to the load_data.csv file.')
    parser.add_argument('--out-data-path', type=str, required=True, help='S3 key (path within bucket) for the final output Parquet file.')

    args = parser.parse_args()
    main(args)