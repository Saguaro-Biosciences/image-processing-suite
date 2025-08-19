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
MODEL_NAME = "timm/tf_efficientnetv2_l.in21k"
CELLPOSE_MODEL = 'nuclei'
FEATURE_LENGTH = 1280
BOX_SIZE = 200
INFERENCE_BATCH_SIZE = 256

# --- Helper Functions ---
def scale_to_8bit(image_16bit):
    """
    Intelligently scales a 16-bit image to 8-bit.
    """
    min_val, max_val = np.min(image_16bit), np.max(image_16bit)
    if max_val == min_val:
        return np.zeros(image_16bit.shape, dtype=np.uint8)
    
    scaled_image = 255.0 * (image_16bit.astype(np.float32) - min_val) / (max_val - min_val)
    return scaled_image.astype(np.uint8)

# --- 2. Producer-Consumer Worker Functions ---

def producer_worker(task_queue, data_queue, worker_id):
    """
    Producer Process: Handles CPU-bound I/O tasks ONLY.
    - Fetches a site task from the task_queue.
    - Loads the 4-channel image from disk.
    - Places the raw image array into the data_queue for the consumer.
    """
    logging.info(f"Producer-{worker_id} started.")

    while True:
        task = task_queue.get()
        if task is None:
            logging.info(f"Producer-{worker_id} received sentinel. Shutting down.")
            break

        site_id, site_image_paths = task
        try:
            all_channels = [tifffile.imread(path) for path in site_image_paths]
            image_4ch = np.stack(all_channels, axis=-1)
            
            # Put the raw image data onto the queue for the GPU worker
            data_queue.put((site_id, image_4ch))

        except Exception as e:
            logging.error(f"Producer-{worker_id} failed on site {site_id}: {e}")
            # Put a placeholder to signal completion even on failure
            data_queue.put((site_id, None))

def consumer_worker(data_queue, results_dict, total_sites, stop_event):
    """
    Consumer Process: Handles ALL GPU-bound tasks.
    - Initializes BOTH Cellpose and the feature extractor on the GPU.
    - Pulls raw image data from the data_queue.
    - Runs Cellpose segmentation on the GPU.
    - Crops cells (fast CPU task).
    - Runs batched feature extraction on the GPU.
    - Stores the final result.
    """
    logging.info("Consumer started.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        logging.warning("CUDA not available. Consumer running on CPU.")

    # --- Load BOTH models onto the single GPU ONCE ---
    logging.info("Loading Cellpose model onto GPU...")
    cell_model = models.CellposeModel(gpu=(device.type == 'cuda'), model_type=CELLPOSE_MODEL)
    
    logging.info("Loading feature extraction model onto GPU...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    feature_model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    
    half_box = BOX_SIZE // 2
    pbar = tqdm(total=total_sites, desc="Consumer Processing Sites")

    while not stop_event.is_set():
        try:
            item = data_queue.get(timeout=1)
            site_id, image_4ch = item

            # Handle case where producer failed to load image
            if image_4ch is None:
                results_dict[site_id] = np.zeros((4, FEATURE_LENGTH), dtype=np.float32)
                pbar.update(1)
                continue
            
            # --- 1. Run Cellpose Segmentation (GPU) ---
            masks, _, _ = cell_model.eval(image_4ch, diameter=100)
            props = regionprops(masks)
            
            if not props:
                results_dict[site_id] = np.zeros((4, FEATURE_LENGTH), dtype=np.float32)
                pbar.update(1)
                continue

            # --- 2. Crop Cells (CPU) ---
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

            # --- 3. Run Batched Feature Extraction (GPU) ---
            batch_pil_images = []
            for cell_crop in all_cell_crops:
                for ch in range(4):
                    scaled_8bit = scale_to_8bit(cell_crop[:, :, ch])
                    pil_image = Image.fromarray(scaled_8bit).convert("RGB")
                    batch_pil_images.append(pil_image)

            site_features = []
            for i in range(0, len(batch_pil_images), INFERENCE_BATCH_SIZE):
                mini_batch = batch_pil_images[i : i + INFERENCE_BATCH_SIZE]
                inputs = processor(images=mini_batch, return_tensors="pt").to(device)

                with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = feature_model(**inputs)
                
                features = outputs.pooler_output.cpu().to(torch.float32).numpy()
                site_features.append(features)
            
            all_features_array = np.vstack(site_features)
            reshaped_features = all_features_array.reshape(len(all_cell_crops), 4, FEATURE_LENGTH)
            
            # Calculate and store the mean feature profile
            mean_site_features = np.mean(reshaped_features, axis=0)
            results_dict[site_id] = mean_site_features
            
            pbar.update(1)

        except Empty:
            continue
        except Exception as e:
            logging.error(f"Consumer failed on site {site_id}: {e}")
            results_dict[site_id] = np.zeros((4, FEATURE_LENGTH), dtype=np.float32)
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
    #s3_input_path_meta = f"s3://{args.bucket_input}/{args.meta_data_key}"
    try:
        logging.info(f"Reading load_data CSV from {s3_input_path_load}")
        load_data = pd.read_csv(s3_input_path_load)
        #logging.info(f"Reading meta_data CSV from {s3_input_path_meta}")
        #meta_data = pd.read_csv(s3_input_path_meta)
    except Exception as e:
        logging.error(f"Failed to read input CSVs from S3. Error: {e}")
        return

    # --- Prepare Tasks for Producers ---
    channel_columns = ['FileName_CL488Y', 'FileName_CL640', 'FileName_DNA', 'FileName_CL488R']
    
    tasks = [
        (index, [f"/home/ubuntu/data/{row[c]}" for c in channel_columns])
        for index, row in load_data.iterrows()
    ]
    num_tasks = len(tasks)
    logging.info(f"Prepared {num_tasks} sites for processing.")

    # --- Initialize Multiprocessing Environment ---
    with mp.Manager() as manager:
        task_queue = Queue()
        # The data_queue now holds full images, so its size should be smaller
        # to avoid excessive RAM usage. It acts as a buffer between CPU and GPU.
        data_queue = Queue(maxsize=args.max_workers)
        results_dict = manager.dict()
        stop_event = Event()

        for task in tasks:
            task_queue.put(task)

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
        for p in producers:
            p.join()
        
        logging.info("All producers have finished. Waiting for consumer to process remaining items.")
        
        while len(results_dict) < num_tasks:
            logging.info(f"Consumer progress: {len(results_dict)}/{num_tasks} sites processed.")
            time.sleep(10)
        
        stop_event.set()
        consumer.join()

        logging.info("All processes have completed.")

        # --- Process and Save Results ---
        site_results = [results_dict[i] for i in range(num_tasks)]
        
        load_data['mean_features'] = site_results
        logging.info("Site-level features extracted and merged.")

        # Aggregate Data to Well Level
        logging.info("Aggregating data to well level...")
        metadata_cols = ["Metadata_Well", "Metadata_Timepoint", "Metadata_Plate"]
        df_subset = load_data[metadata_cols + ['mean_features']]

        agg_functions = {
            'mean_features': lambda arrays: np.mean(np.stack(arrays.values), axis=0)
        }
        for col in metadata_cols:
            if col != 'Metadata_Well':
                agg_functions[col] = 'first'

        well_level_data = df_subset.groupby('Metadata_Well').agg(agg_functions).reset_index()

        #final_data=pd.merge(left=well_level_data,right=meta_data,on=['Metadata_Well','Metadata_Plate'],how='inner')
        well_level_data['mean_features'] = well_level_data['mean_features'].apply(lambda x: x.tolist())

        logging.info(f"Saving final results to {args.out_data_path}")
        well_level_data.to_parquet(args.out_data_path, engine='pyarrow')
        
        logging.info("Script finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the optimized cell image analysis pipeline.")
    
    parser.add_argument('--max-workers', type=int, default=os.cpu_count() * 2, help='Number of parallel CPU I/O processes.')
    parser.add_argument('--bucket-input', type=str, required=True, help='Name of the S3 bucket for input data.')
    parser.add_argument('--load-data-key', type=str, required=True, help='S3 key to the load_data.csv file.')
    parser.add_argument('--meta-data-key', type=str, required=False, help='S3 key to the meta_data.csv file.')
    parser.add_argument('--out-data-path', type=str, required=True, help='Local or S3 path for the final output Parquet file.')

    args = parser.parse_args()

    try:
        mp.set_start_method('spawn', force=True)
        logging.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        logging.warning("Multiprocessing start method already set.")

    main(args)