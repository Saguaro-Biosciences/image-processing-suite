import os
import argparse
import logging
import time
import tifffile
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from queue import Empty

# Use torch.multiprocessing for efficient tensor sharing between processes
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Event

# Import scientific libraries
from skimage.measure import regionprops
from cellpose import models
from transformers import AutoImageProcessor, AutoModel

# --- 1. Setup Logging and Constants ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- MODEL AND PIPELINE CONFIGURATION ---
# Swapped to EfficientNetV2-M for a balance of performance and VRAM usage.
# This is the single most important change for VRAM reduction.
MODEL_NAME = "timm/tf_efficientnetv2_m.in21k_ft_in1k"
CELLPOSE_MODEL = 'nuclei'
FEATURE_LENGTH = 1280 # Corresponds to EfficientNetV2-M/L/XL
BOX_SIZE = 200
# Batch size for inference on the GPU. This is a critical tuning parameter.
# A larger batch size improves GPU utilization but uses more VRAM.
# Start with a moderate value and increase based on available VRAM.
INFERENCE_BATCH_SIZE = 256

# --- Helper Functions ---
def scale_to_8bit(image_16bit):
    """
    Intelligently scales a 16-bit image to 8-bit by stretching
    the actual dynamic range of the image to the full 0-255 range.
    """
    min_val, max_val = np.min(image_16bit), np.max(image_16bit)
    if max_val == min_val:
        return np.zeros(image_16bit.shape, dtype=np.uint8)
    
    # Use np.float32 for intermediate calculations to avoid overflow and precision issues
    scaled_image = 255.0 * (image_16bit.astype(np.float32) - min_val) / (max_val - min_val)
    return scaled_image.astype(np.uint8)

# --- 2. Producer-Consumer Worker Functions ---

def producer_worker(task_queue, data_queue, worker_id):
    """
    Producer Process: Handles CPU-bound tasks.
    - Fetches a site task from the task_queue.
    - Loads images from disk (I/O-bound).
    - Runs Cellpose segmentation (CPU/GPU-bound).
    - Crops, masks, and pads all cells found in the site (CPU-bound).
    - Places the prepared data (list of cell crops) into the data_queue.
    """
    logging.info(f"Producer-{worker_id} started.")
    
    # Initialize Cellpose model within the worker process to ensure process safety.
    # The GPU will be primarily used by the consumer, so running Cellpose on CPU
    # can be a good strategy to avoid contention if the consumer is the bottleneck.
    # However, if producers are slow, using the GPU here might be beneficial.
    # We will default to CPU to dedicate the GPU to the consumer.
    cell_model = models.CellposeModel(gpu=False, model_type=CELLPOSE_MODEL)
    
    half_box = BOX_SIZE // 2

    while True:
        task = task_queue.get()
        if task is None:  # Sentinel value to signal termination
            logging.info(f"Producer-{worker_id} received sentinel. Shutting down.")
            break
        

        site_id, site_image_paths = task
        try:
            all_channels = [tifffile.imread(path) for path in site_image_paths]
            image_4ch = np.stack(all_channels, axis=-1)
            
            # Run Cellpose to get masks
            masks, _, _ = cell_model.eval(image_4ch, diameter=100) # Grayscale mode
            props = regionprops(masks)
            
            if not props:
                # Still put an empty item on the queue to signal completion for this site
                data_queue.put((site_id,))
                continue

            # Process each cell found in the site and collect crops
            all_cell_crops = []
            h, w, _ = image_4ch.shape

            for prop in props:
                y_center, x_center = map(int, prop.centroid)
                target_id = prop.label
                y1, y2 = max(0, y_center - half_box), min(h, y_center + half_box)
                x1, x2 = max(0, x_center - half_box), min(w, x_center + half_box)
                
                mask_crop = masks[y1:y2, x1:x2]
                binary_mask = (mask_crop == target_id)[:, :, np.newaxis]
                
                cell_crop_4ch = image_4ch[y1:y2, x1:x2, :]
                masked_cell_crop = cell_crop_4ch * binary_mask
                
                pad_h = BOX_SIZE - masked_cell_crop.shape[0]
                pad_w = BOX_SIZE - masked_cell_crop.shape[1]
                padded_crop = np.pad(masked_cell_crop, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
                all_cell_crops.append(padded_crop)

            # Put the collected data for the site into the data queue
            data_queue.put((site_id, all_cell_crops))

        except Exception as e:
            logging.error(f"Producer-{worker_id} failed on site {site_id}: {e}")
            data_queue.put((site_id,)) # Signal failure/completion

def consumer_worker(data_queue, results_dict, total_sites, stop_event):
    """
    Consumer Process: Handles GPU-bound tasks.
    - Initializes the feature extraction model ONCE on the GPU.
    - Continuously pulls prepared cell data from the data_queue.
    - Prepares batches of cell crops for the model.
    - Runs batched inference using Automatic Mixed Precision (AMP).
    - Calculates the mean feature profile for the site.
    - Stores the result in the shared results_dict.
    """
    logging.info("Consumer started.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        logging.warning("CUDA not available. Consumer running on CPU.")

    # Load Hugging Face model and processor ONCE
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    
    processed_sites = 0
    pbar = tqdm(total=total_sites, desc="Consumer Processing Sites")

    while not stop_event.is_set():
        try:
            # Get data from the queue with a timeout to allow checking the stop_event
            # --- FIX STARTS HERE ---
            item = data_queue.get(timeout=1)

            if len(item) == 1: # This handles the case of no cells found
                site_id = item[0]
                site_cell_crops = [] # Manually set crops to an empty list
            else:
                site_id, site_cell_crops = item
            # --- FIX ENDS HERE ---

            if not site_cell_crops:
                # If no cells were found, store a zero vector
                # This block now correctly handles both cases
                results_dict[site_id] = np.zeros((4, FEATURE_LENGTH), dtype=np.float32)
                processed_sites += 1
                pbar.update(1)
                continue
            
            if not site_cell_crops:
                # If no cells were found, store a zero vector
                results_dict[site_id] = np.zeros((4, FEATURE_LENGTH), dtype=np.float32)
                processed_sites += 1
                pbar.update(1)
                continue

            
            # Prepare all single-channel crops for one large batch
            batch_pil_images = []
            for cell_crop in site_cell_crops:
                for ch in range(4):
                    raw_16bit_channel = cell_crop[:, :, ch]
                    scaled_8bit_channel = scale_to_8bit(raw_16bit_channel)
                    pil_image = Image.fromarray(scaled_8bit_channel).convert("RGB")
                    batch_pil_images.append(pil_image)

            site_features =[]
            # Process in mini-batches to manage VRAM for sites with many cells
            for i in range(0, len(batch_pil_images), INFERENCE_BATCH_SIZE):
                mini_batch = batch_pil_images[i : i + INFERENCE_BATCH_SIZE]
                inputs = processor(images=mini_batch, return_tensors="pt").to(device)

                # Use torch.no_grad() and autocast for maximum performance and memory efficiency
                with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(**inputs)
                
                # Move features to CPU, convert to float32 for stable aggregation
                features = outputs.pooler_output.cpu().to(torch.float32).numpy()
                site_features.append(features)
            
            # Concatenate features from all mini-batches
            all_features_array = np.vstack(site_features)
            
            # Reshape to (num_cells, 4, feature_length)
            num_cells = len(site_cell_crops)
            reshaped_features = all_features_array.reshape(num_cells, 4, FEATURE_LENGTH)
            
            # Calculate the mean feature profile for the site
            mean_site_features = np.mean(reshaped_features, axis=0)
            results_dict[site_id] = mean_site_features
            
            # Clean up to release memory
            del site_cell_crops, batch_pil_images, all_features_array, reshaped_features
            
            processed_sites += 1
            pbar.update(1)

        except Empty:
            # Queue was empty, loop again to check stop_event
            continue
        except Exception as e:
            logging.error(f"Consumer failed on site {site_id}: {e}")
            results_dict[site_id] = np.zeros((4, FEATURE_LENGTH), dtype=np.float32)
            processed_sites += 1
            pbar.update(1)

    pbar.close()
    logging.info("Consumer finished processing.")


# --- 3. Main Execution Block ---
def main(args):
    """
    Main function to orchestrate the Producer-Consumer pipeline.
    """
    logging.info(f"Starting analysis with parameters: {args}")

    # --- Load Data ---
    s3_input_path_load = f"s3://{args.bucket_input}/{args.load_data_key}"
    s3_input_path_meta = f"s3://{args.bucket_input}/{args.meta_data_key}"
    try:
        logging.info(f"Reading load_data CSV from {s3_input_path_load}")
        load_data = pd.read_csv(s3_input_path_load)
        logging.info(f"Reading meta_data CSV from {s3_input_path_meta}")
        meta_data = pd.read_csv(s3_input_path_meta)
    except Exception as e:
        logging.error(f"Failed to read input CSVs from S3. Error: {e}")
        return

    # --- Prepare Tasks for Producers ---
    channel_columns = ['FileName_CL488Y', 'FileName_CL640', 'FileName_DNA', 'FileName_CL488R']
    
    # Create a list of tasks. Each task is a tuple of (site_id, list_of_paths).
    # Using a unique site_id (like index) is crucial for collecting results.
    tasks = [
        (index, [f"/home/ubuntu/data/{row[c]}" for c in channel_columns])
        for index, row in load_data.iterrows()
    ]
    num_tasks = len(tasks)
    logging.info(f"Prepared {num_tasks} sites for processing.")

    # --- Initialize Multiprocessing Environment ---
    # Use a Manager to create a shared dictionary for results
    with mp.Manager() as manager:
        task_queue = Queue()
        data_queue = Queue(maxsize=args.max_workers * 2) # Bounded queue to prevent memory overflow
        results_dict = manager.dict()
        stop_event = Event()

        # Populate the task queue for producers
        for task in tasks:
            task_queue.put(task)

        # Add sentinel values to the queue for each producer to signal completion
        for _ in range(args.max_workers):
            task_queue.put(None)

        # --- Start Producer and Consumer Processes ---
        producers = [
            Process(target=producer_worker, args=(task_queue, data_queue, i), name=f"Producer-{i}")
            for i in range(args.max_workers)
        ]
        consumer = Process(target=consumer_worker, args=(data_queue, results_dict, num_tasks, stop_event), name="Consumer")

        logging.info(f"Starting {args.max_workers} producers and 1 consumer...")
        consumer.start()
        for p in producers:
            p.start()
        
        # --- Monitor and Wait for Completion ---
        # Wait for producers to finish their work
        for p in producers:
            p.join()
        
        logging.info("All producers have finished. Waiting for consumer to process remaining items.")
        
        # Wait for the consumer to finish processing all items in the queue
        while len(results_dict) < num_tasks:
            logging.info(f"Consumer progress: {len(results_dict)}/{num_tasks} sites processed.")
            time.sleep(10)
        
        # Signal the consumer to stop and wait for it to terminate
        stop_event.set()
        consumer.join()

        logging.info("All processes have completed.")

        # --- Process and Save Results ---
        # Retrieve results from the shared dictionary, ensuring correct order
        site_results = [results_dict[i] for i in range(num_tasks)]
        
        load_data['mean_features'] = site_results
        logging.info("Site-level features extracted and merged.")

        # Aggregate Data to Well Level
        logging.info("Aggregating data to well level...")
        metadata_cols =["Metadata_Well","Metadata_Timepoint","Metadata_Plate"]
        df_subset = load_data[metadata_cols + ['mean_features']]

        agg_functions = {
            'mean_features': lambda arrays: np.mean(np.stack(arrays.values), axis=0)
        }
        well_level_data = df_subset.groupby('Metadata_Well').agg(agg_functions).reset_index()

        # Merge with metadata
        final_data = pd.merge( # Assign to final_data
            left=well_level_data,
            right=meta_data,
            on=['Metadata_Well','Metadata_Plate'],
            how='inner'
        )
        final_data['mean_features'] = final_data['mean_features'].apply(lambda x: x.tolist())

        # Save Final Results
        logging.info(f"Saving final results to {args.out_data_path}")
        final_data.to_parquet(args.out_data_path, engine='pyarrow')
        
        logging.info("Script finished successfully.")


if __name__ == '__main__':
    # --- 4. Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Run the optimized cell image analysis pipeline.")
    
    # max_workers now controls the number of CPU-bound producer processes
    parser.add_argument('--max-workers', type=int, default=os.cpu_count()//2 +1, help='Number of parallel producer processes.')
    parser.add_argument('--bucket-input', type=str, required=True, help='Name of the S3 bucket for input data.')
    parser.add_argument('--load-data-key', type=str, required=True, help='S3 key to the load_data.csv file.')
    parser.add_argument('--meta-data-key', type=str, required=True, help='S3 key to the meta_data.csv file.')
    parser.add_argument('--out-data-path', type=str, required=True, help='Local or S3 path for the final output Parquet file.')

    args = parser.parse_args()

    # Set multiprocessing start method to 'spawn' for CUDA compatibility.
    # This is critical for preventing deadlocks and CUDA errors.
    try:
        mp.set_start_method('spawn', force=True)
        logging.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        logging.warning("Multiprocessing start method already set.")

    main(args)