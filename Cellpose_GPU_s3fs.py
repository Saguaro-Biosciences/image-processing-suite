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
INFERENCE_BATCH_SIZE = 1000 

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

def producer_worker(task_queue, data_queue, worker_id,channels,csv_image_key): 
    """ 
    Producer Process: Handles CPU-bound I/O tasks ONLY. 
    - Fetches a site task from the task_queue. 
    - Loads the 4-channel image from disk. 
    - Places the raw image array into the data_queue for the consumer. 
    """ 
    logging.info(f"Producer-{worker_id} started.") 
    if csv_image_key:
        try:
            channel_correction = [np.load(f'{csv_image_key}/{c}_illum.npy') for c in channels]
            logging.info(f"Producer-{worker_id} loaded correction arrays.")
        except Exception as e:
            logging.error(f"Producer-{worker_id} FAILED to load correction arrays: {e}")
            # If loading fails, this worker can't do anything.
            return

    while True: 
        task = task_queue.get() 
        if task is None: 
            logging.info(f"Producer-{worker_id} received sentinel. Shutting down.") 
            break 

        site_id, site_image_paths = task 
        if csv_image_key:
            try: 
                all_channels = [tifffile.imread(path)/channel_correction[n] for n,path in enumerate(site_image_paths)] 
                image_4ch = np.stack(all_channels, axis=-1) 
                
                # Put the raw image data onto the queue for the GPU worker 
                data_queue.put((site_id, image_4ch)) 

            except Exception as e: 
                logging.error(f"Producer-{worker_id} failed on site {site_id}: {e}") 
                # Put a placeholder to signal completion even on failure 
                data_queue.put((site_id, None)) 
        else:
            try: 
                all_channels = [tifffile.imread(path) for path in site_image_paths] 
                image_4ch = np.stack(all_channels, axis=-1) 
                
                # Put the raw image data onto the queue for the GPU worker 
                data_queue.put((site_id, image_4ch)) 

            except Exception as e: 
                logging.error(f"Producer-{worker_id} failed on site {site_id}: {e}") 
                # Put a placeholder to signal completion even on failure 
                data_queue.put((site_id, None))

def consumer_worker(data_queue, results_dict, stop_event, worker_id, gpu_id=0): 
    import os
    import gc 
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    internal_device_id = 0
    
    from skimage.measure import regionprops 
    from cellpose import models 
    from transformers import AutoImageProcessor, AutoModel
    
    logging.info(f"Consumer-{worker_id} started. Physical GPU: {gpu_id} -> Mapped to cuda:{internal_device_id}") 
    
    device = torch.device(f"cuda:{internal_device_id}" if torch.cuda.is_available() else "cpu") 
    
    # --- Load Models --- 
    logging.info(f"Consumer-{worker_id}: Loading Cellpose model...") 
    cell_model = models.CellposeModel(gpu=(device.type == 'cuda'), model_type=CELLPOSE_MODEL, device=device) 
    
    logging.info(f"Consumer-{worker_id}: Loading feature extraction model...") 
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME) 
    feature_model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval() 
    
    half_box = BOX_SIZE // 2 
    
    # =========================================================================
    # CHANGE 1: Initialize Batch Size OUTSIDE the loop
    # The worker "learns" the safe limit and keeps it for future images.
    # =========================================================================
    current_batch_size = INFERENCE_BATCH_SIZE 

    while not stop_event.is_set(): 
        try: 
            item = data_queue.get(timeout=1) 
            site_id, image_4ch = item 
            n_channels = image_4ch.shape[-1] if image_4ch is not None else 0

            if image_4ch is None: 
                results_dict[site_id] = np.zeros((n_channels, FEATURE_LENGTH), dtype=np.float32) 
                continue 
            
            # --- 1. Run Cellpose Segmentation --- 
            try:
                masks, _, _ = cell_model.eval(image_4ch, diameter=100) 
            except torch.cuda.OutOfMemoryError:
                logging.warning(f"Consumer-{worker_id}: OOM during Cellpose (Site {site_id}). Clearing cache.")
                torch.cuda.empty_cache()
                gc.collect()
                masks, _, _ = cell_model.eval(image_4ch, diameter=100) 

            props = regionprops(masks) 
            
            if not props: 
                results_dict[site_id] = np.zeros((n_channels, FEATURE_LENGTH), dtype=np.float32) 
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

            # --- 3. Run Batched Feature Extraction --- 
            batch_pil_images = [] 
            for cell_crop in all_cell_crops: 
                for ch in range(n_channels): 
                    scaled_8bit = scale_to_8bit(cell_crop[:, :, ch]) 
                    pil_image = Image.fromarray(scaled_8bit).convert("RGB") 
                    batch_pil_images.append(pil_image) 

            site_features = [] 
            idx = 0
            total_images = len(batch_pil_images)

            while idx < total_images:
                end_idx = min(idx + current_batch_size, total_images)
                mini_batch = batch_pil_images[idx : end_idx]
                
                try:
                    inputs = processor(images=mini_batch, return_tensors="pt").to(device) 
                    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.float16): 
                        outputs = feature_model(**inputs) 
                    
                    features = outputs.pooler_output.cpu().to(torch.float32).numpy() 
                    site_features.append(features)
                    idx = end_idx 

                    # Optional: Slowly try to increase batch size again if it got too small?
                    # For now, it's safer to stay low to prevent repeated crashing.

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    old_bs = current_batch_size
                    # Reduce persistent batch size
                    current_batch_size = max(1, current_batch_size // 2)
                    
                    logging.warning(f"Consumer-{worker_id}: OOM at site {site_id}. PERMANENTLY dropping batch size: {old_bs} -> {current_batch_size}")
                    
                    if old_bs == 1:
                        logging.error(f"Consumer-{worker_id}: Site {site_id} failed with batch_size=1.")
                        site_features = []
                        break
            
            if len(site_features) > 0:
                all_features_array = np.vstack(site_features) 
                n_cells = len(all_cell_crops)
                reshaped_features = all_features_array.reshape(n_cells, n_channels, FEATURE_LENGTH) 
                mean_site_features = np.mean(reshaped_features, axis=0) 
                results_dict[site_id] = (mean_site_features, n_cells)
                logging.info(f"Consumer-{worker_id}: Finished SITE {site_id} ({n_cells} cells processed).")
            else:
                 results_dict[site_id] = (np.zeros((n_channels, FEATURE_LENGTH), dtype=np.float32), 0)

        except Empty: 
            continue 
        except Exception as e: 
            logging.error(f"Consumer-{worker_id} failed: {e}") 
            if 'site_id' in locals() and 'n_channels' in locals(): 
                results_dict[site_id] = (np.zeros((n_channels, FEATURE_LENGTH), dtype=np.float32), 0) 

    logging.info(f"Consumer-{worker_id} finished processing.")


# --- 3. Main Execution Block --- 
def main(args): 
    """ 
    Main function to orchestrate the Producer-Consumer pipeline. 
    """ 
    logging.info(f"Starting analysis with parameters: {args}") 

    # --- Load Data --- 
    s3_input_path_load = f"s3://{args.bucket_input}/{args.load_data_key}" 
    try: 
        logging.info(f"Reading load_data CSV from {s3_input_path_load}") 
        load_data = pd.read_csv(s3_input_path_load) 
    except Exception as e: 
        logging.error(f"Failed to read input CSVs from S3. Error: {e}") 
        return 

    # --- Prepare Tasks for Producers --- 
    channel_columns = [f'FileName_{c}' for c in args.channels]
    if getattr(args, "csv_image_key", None):
        image_df=pd.read_csv(f"{args.csv_image_key}/Image.csv")
        not_failing_images = (image_df.filter(like='ImageQC_').sum(axis=1) < 2)
        load_data=load_data[not_failing_images].copy()

    else:
        logging.info("No csv_image_key provided — skipping image QC filtering.")
    tasks = [ 
        (index, [f"{args.data_base_path}/{row[c]}" for c in channel_columns]) 
        for index, row in load_data.iterrows() 
    ] 
    num_tasks = len(tasks) 
    logging.info(f"Prepared {num_tasks} sites for processing.") 

    # --- Initialize Multiprocessing Environment --- 
    with mp.Manager() as manager: 
        task_queue = Queue() 
        # A buffer between CPU producers and GPU consumers. 
        # Sized relative to consumers to prevent excessive RAM usage. 
        data_queue = Queue(maxsize=args.num_consumers) 
        results_dict = manager.dict() 
        stop_event = Event() 

        # Populate the task queue for producers 
        for task in tasks: 
            task_queue.put(task) 

        # Add sentinel values to signal producers to stop 
        for _ in range(args.max_workers): 
            task_queue.put(None) 

        # --- Start Producer and Consumer Processes --- 
        producers = [ 
            Process(target=producer_worker, args=(task_queue, data_queue, i,args.channels,args.csv_image_key), name=f"Producer-{i}") 
            for i in range(args.max_workers) 
        ] 
        # **MODIFIED: Create a list of consumers for GPUs >1** 
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            logging.warning("No GPUs detected. Defaulting to GPU logic on CPU (index 0).")
            available_gpus = 1

        consumers = [ 
            Process(
                target=consumer_worker, 
                args=(data_queue, results_dict, stop_event, i, i % available_gpus), 
                name=f"Consumer-{i}"
            ) 
            for i in range(args.num_consumers)
        ] 

        logging.info(f"Starting {args.max_workers} producers and {args.num_consumers} consumers...") 
        # **MODIFIED: Start all consumers** 
        for c in consumers: 
            c.start() 
        for p in producers: 
            p.start() 
        
        # --- Monitor and Wait for Completion --- 
        for p in producers: 
            p.join() 
        
        logging.info("All producers have finished. Waiting for consumers to process remaining items.") 
        
        # Main progress monitoring loop 
        pbar = tqdm(total=num_tasks, desc="Overall Progress") 
        last_processed_count = 0 
        while len(results_dict) < num_tasks: 
            current_processed_count = len(results_dict) 
            pbar.update(current_processed_count - last_processed_count) 
            last_processed_count = current_processed_count 
            time.sleep(2) 
        
        # Final update to ensure the progress bar reaches 100% 
        pbar.update(num_tasks - last_processed_count) 
        pbar.close() 

        logging.info("All tasks processed. Signaling consumers to shut down.")
        stop_event.set() 

        # **MODIFIED: Signal all consumers to stop and wait for them** stop_event.set() 
        for c in consumers: 
            c.join() 

        logging.info("All processes have completed.") 

        # --- Process and Save Results --- 
        # Ensure results are sorted by site_id for correct merging 
        # --- Process and Save Results --- 
        original_indices = [task[0] for task in tasks]
        
        # Retrieve the tuple (feature_array, cell_count)
        raw_results = [results_dict[i] for i in original_indices]
        
        # Unpack into two separate lists
        site_features = [item[0] for item in raw_results]
        site_counts = [item[1] for item in raw_results]
        
        logging.info("Site-level processing complete. Preparing outputs.") 

        # --- OUTPUT 1: Cell Counts CSV ---
        # Add the counts to the original load_data dataframe which has the Metadata
        load_data['Cell_Count'] = site_counts
        
        # Define output path for counts (e.g., replace .parquet with _counts.csv)
        counts_out_path = args.out_data_path.replace('.parquet', '_counts.csv')
        
        logging.info(f"Saving site-level cell counts to {counts_out_path}")
        # Save complete metadata + counts
        load_data.to_csv(counts_out_path, index=False)

        # --- OUTPUT 2: Feature Aggregation (Parquet) ---
        results_df = pd.DataFrame({'mean_features': site_features}, index=original_indices)
        # We join again to ensure the 'mean_features' column is attached to the dataframe used for aggregation
        # Note: 'load_data' now has 'Cell_Count', which is fine.
        load_data = load_data.join(results_df)

        logging.info("Aggregating features to well level...") 
        metadata_cols = ["Metadata_Well", "Metadata_Timepoint", "Metadata_Plate"] 
        
        # We only aggregate features here, but you could also aggregate Cell_Count (sum) if you wanted
        df_subset = load_data[metadata_cols + ['mean_features']] 

        agg_functions = { 
            'mean_features': lambda arrays: np.mean(np.stack(arrays.values), axis=0) 
        } 
        for col in metadata_cols: 
            if col != 'Metadata_Well': 
                agg_functions[col] = 'first' 

        well_level_data = df_subset.groupby('Metadata_Well').agg(agg_functions).reset_index() 
        well_level_data['mean_features'] = well_level_data['mean_features'].apply(lambda x: x.tolist()) 

        logging.info(f"Saving feature results to {args.out_data_path}") 
        well_level_data.to_parquet(args.out_data_path, engine='pyarrow') 
        # Explicitly close queues to release semaphores and silence warnings
        task_queue.close()
        task_queue.join_thread()
        
        data_queue.close()
        data_queue.join_thread()
        logging.info("Script finished successfully.") 


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Run the optimized cell image analysis pipeline.")
    parser.add_argument('--data-base-path', type=str, required=True, help='Base path to the mounted data directory.') 
    parser.add_argument('--num-consumers', type=int, default=2, help='Number of parallel GPU consumer processes.') 
    parser.add_argument('--max-workers', type=int, default=24, help='Number of parallel CPU I/O producer processes.') 
    parser.add_argument('--bucket-input', type=str, required=True, help='Name of the S3 bucket for input data.') 
    parser.add_argument('--load-data-key', type=str, required=True, help='S3 key to the load_data.csv file.')
    parser.add_argument('--csv-image-key', type=str, required=False, help='S3 key to the Image.csv file.')
    parser.add_argument('--channels', nargs='+', type=str, required=True, help='Channel list and order (first 3 are used for segmentation).')
    parser.add_argument('--out-data-path', type=str, required=True, help='Local or S3 path for the final output Parquet file.') 

    args = parser.parse_args() 

    try: 
        mp.set_start_method('spawn', force=True) 
        logging.info("Set multiprocessing start method to 'spawn'.") 
    except RuntimeError: 
        logging.warning("Multiprocessing start method already set.") 

    main(args)
