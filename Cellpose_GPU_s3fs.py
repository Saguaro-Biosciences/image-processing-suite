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
    """ 
    logging.info(f"Producer-{worker_id} started.") 
    if csv_image_key:
        try:
            channel_correction = [np.load(f'{csv_image_key}/{c}_illum.npy') for c in channels]
            logging.info(f"Producer-{worker_id} loaded correction arrays.")
        except Exception as e:
            logging.error(f"Producer-{worker_id} FAILED to load correction arrays: {e}")
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
                data_queue.put((site_id, image_4ch)) 

            except Exception as e: 
                logging.error(f"Producer-{worker_id} failed on site {site_id}: {e}") 
                data_queue.put((site_id, None)) 
        else:
            try: 
                all_channels = [tifffile.imread(path) for path in site_image_paths] 
                image_4ch = np.stack(all_channels, axis=-1) 
                data_queue.put((site_id, image_4ch)) 

            except Exception as e: 
                logging.error(f"Producer-{worker_id} failed on site {site_id}: {e}") 
                data_queue.put((site_id, None))

def consumer_worker(data_queue, results_dict, stop_event, worker_id, expected_n_channels, gpu_id=0, single_cell_mode=False): 
    import os
    import gc 
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    internal_device_id = 0
    
    from skimage.measure import regionprops 
    from cellpose import models 
    from transformers import AutoImageProcessor, AutoModel
    
    logging.info(f"Consumer-{worker_id} started. Physical GPU: {gpu_id} -> Mapped to cuda:{internal_device_id} | Single Cell Mode: {single_cell_mode}") 
    
    device = torch.device(f"cuda:{internal_device_id}" if torch.cuda.is_available() else "cpu") 
    
    # --- Load Models --- 
    logging.info(f"Consumer-{worker_id}: Loading Cellpose model...") 
    cell_model = models.CellposeModel(gpu=(device.type == 'cuda'), model_type=CELLPOSE_MODEL, device=device) 
    
    logging.info(f"Consumer-{worker_id}: Loading feature extraction model...") 
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME) 
    feature_model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval() 
    
    half_box = BOX_SIZE // 2 
    current_batch_size = INFERENCE_BATCH_SIZE 

    # Helper for empty return
    def return_empty_result(s_id):
        if single_cell_mode:
            # Return empty array with shape (0, n_ch, feat)
            results_dict[s_id] = (np.zeros((0, expected_n_channels, FEATURE_LENGTH), dtype=np.float32), 0)
        else:
            # Return zero vector for site aggregation
            results_dict[s_id] = (np.zeros((expected_n_channels, FEATURE_LENGTH), dtype=np.float32), 0)

    while not stop_event.is_set(): 
        try: 
            item = data_queue.get(timeout=1) 
            site_id, image_4ch = item 
            
            # --- Shape Consistency ---
            if image_4ch is None: 
                logging.warning(f"Consumer-{worker_id}: [ERROR] Site {site_id} is None. Returning placeholder.")
                return_empty_result(site_id)
                continue 

            if image_4ch.shape[-1] != expected_n_channels:
                logging.error(f"Consumer-{worker_id}: [ERROR] Site {site_id} has wrong shape {image_4ch.shape}. Skipping.")
                return_empty_result(site_id)
                continue
            
            n_channels = expected_n_channels
            
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
                return_empty_result(site_id)
                continue 

            # --- 2. Crop Cells (CPU) --- 
            all_cell_crops = [] 
            cell_coords = []
            h, w, _ = image_4ch.shape 
            for prop in props: 
                y_center, x_center = map(int, prop.centroid) 
                
                # Purge cells touching absolute image limits
                if (y_center - half_box < 0) or (y_center + half_box > h) or \
                   (x_center - half_box < 0) or (x_center + half_box > w):
                    continue
                
                target_id = prop.label 
                y1, y2 = y_center - half_box, y_center + half_box 
                x1, x2 = x_center - half_box, x_center + half_box 
                
                mask_crop = masks[y1:y2, x1:x2] 
                binary_mask = (mask_crop == target_id)[:, :, np.newaxis] 
                cell_crop_4ch = image_4ch[y1:y2, x1:x2, :] 
                masked_cell_crop = cell_crop_4ch * binary_mask 
                
                all_cell_crops.append(masked_cell_crop)
                cell_coords.append((y_center, x_center)) 

            if not all_cell_crops:
                return_empty_result(site_id)
                continue
            
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

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    old_bs = current_batch_size
                    current_batch_size = max(1, current_batch_size // 2)
                    logging.warning(f"Consumer-{worker_id}: OOM at site {site_id}. Dropping batch size: {old_bs} -> {current_batch_size}")
                    if old_bs == 1:
                        site_features = []
                        break
            
            if len(site_features) > 0:
                all_features_array = np.vstack(site_features) 
                n_cells = len(all_cell_crops)
                
                # Reshape to (N_cells, N_channels, Feature_Length)
                reshaped_features = all_features_array.reshape(n_cells, n_channels, FEATURE_LENGTH) 
                
                if single_cell_mode:
                    #Return cell level embeddings
                    results_dict[site_id] = (reshaped_features, n_cells, cell_coords)
                else:
                    #Pass SUM instead of MEAN to allow true well-level weighting
                    sum_site_features = np.sum(reshaped_features, axis=0) 
                    results_dict[site_id] = (sum_site_features, n_cells, cell_coords)
                
                logging.info(f"Consumer-{worker_id}: Finished SITE {site_id} ({n_cells} cells processed).")
            else:
                return_empty_result(site_id)

        except Empty: 
            continue 
        except Exception as e: 
            logging.error(f"Consumer-{worker_id} failed: {e}") 
            if 'site_id' in locals(): 
                return_empty_result(site_id)

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
        not_failing_images = (image_df.filter(like='ImageQC_').sum(axis=1) < 1)
        load_data=load_data[not_failing_images].copy()
    else:
        logging.info("No csv_image_key provided — skipping image QC filtering.")

    tasks = [ 
        (index, [f"{args.data_base_path}/{row[c]}" for c in channel_columns]) 
        for index, row in load_data.iterrows() 
    ] 
    num_tasks = len(tasks) 
    logging.info(f"Prepared {num_tasks} sites for processing. Out of ") # Add the original dim[1] of the load data set 

    # --- Initialize Multiprocessing Environment --- 
    with mp.Manager() as manager: 
        task_queue = Queue() 
        data_queue = Queue(maxsize=args.num_consumers) 
        results_dict = manager.dict() 
        stop_event = Event() 

        for task in tasks: 
            task_queue.put(task) 
        for _ in range(args.max_workers): 
            task_queue.put(None) 

        # --- Start Producers --- 
        producers = [ 
            Process(target=producer_worker, args=(task_queue, data_queue, i,args.channels,args.csv_image_key), name=f"Producer-{i}") 
            for i in range(args.max_workers) 
        ] 

        # --- Start Consumers ---
        expected_n_channels = len(args.channels)
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            logging.warning("No GPUs detected. Defaulting to GPU logic on CPU (index 0).")
            available_gpus = 1

        consumers = [ 
            Process(
                target=consumer_worker, 
                args=(data_queue, results_dict, stop_event, i, expected_n_channels, i % available_gpus, args.single_cell), 
                name=f"Consumer-{i}"
            ) 
            for i in range(args.num_consumers)
        ] 

        logging.info(f"Starting {args.max_workers} producers and {args.num_consumers} consumers...") 
        for c in consumers: c.start() 
        for p in producers: p.start() 
        
        # --- Monitor --- 
        for p in producers: p.join() 
        logging.info("All producers have finished. Waiting for consumers...") 
        
        pbar = tqdm(total=num_tasks, desc="Overall Progress") 
        last_processed_count = 0 
        while len(results_dict) < num_tasks: 
            current_processed_count = len(results_dict) 
            pbar.update(current_processed_count - last_processed_count) 
            last_processed_count = current_processed_count 
            time.sleep(2) 
        pbar.update(num_tasks - last_processed_count) 
        pbar.close() 

        logging.info("All tasks processed. Signaling consumers to shut down.")
        stop_event.set() 
        for c in consumers: c.join() 
        logging.info("All processes have completed.") 

        # --- Process Results --- 
        original_indices = [task[0] for task in tasks]
        raw_results = [results_dict[i] for i in original_indices]
        
        site_features = [item[0] for item in raw_results]
        site_counts = [item[1] for item in raw_results]
        site_coords = [item[2] for item in raw_results]
        
        load_data['Cell_Count'] = site_counts
        counts_out_path = args.out_data_path.replace('.parquet', '_counts.csv')
        logging.info(f"Saving site-level cell counts to {counts_out_path}")
        load_data.to_csv(counts_out_path, index=False)

        # --- Opt OUTPUT Save Coordinates Output ---
        if args.save_coords:
            logging.info("Building Cell Coordinates Table...")
            coords_records = []
            for idx, coords_list in zip(original_indices, site_coords):
                well = load_data.loc[idx, 'Metadata_Well']
                site = load_data.loc[idx, 'Metadata_Site'] if 'Metadata_Site' in load_data.columns else str(idx)
                for cell_idx, (y, x) in enumerate(coords_list):
                    coords_records.append({
                        'Cell_ID': f"{well}_{site}_cell{cell_idx}",
                        'Y_Center': y,
                        'X_Center': x
                    })
            if coords_records:
                coords_df = pd.DataFrame(coords_records)
                coords_out_path = args.out_data_path.replace('.parquet', '_coords.parquet')
                coords_df.to_parquet(coords_out_path, engine='pyarrow')
                logging.info(f"Saved coordinates to {coords_out_path}")

        # --- OUTPUT 2: Features (Branch Logic) ---
        
        if args.single_cell:
            logging.info("SINGLE CELL MODE DETECTED: Skipping Aggregation.")
            logging.info("Expanding site-level metadata to cell-level...")

            # 1. Attach features and counts to the dataframe (site level)
            # Use a temporary column to hold the arrays
            load_data['features_temp'] = site_features
            
            # 2. Filter out sites with 0 cells to avoid errors during expansion
            valid_sites = load_data[load_data['Cell_Count'] > 0].copy()
            
            if valid_sites.empty:
                logging.warning("No cells detected in the entire dataset. Saving empty parquet.")
                valid_sites.to_parquet(args.out_data_path, engine='pyarrow')
                return

            # 3. Expand Metadata: Repeat site rows N times (where N = Cell_Count)
            # 'index.repeat' creates a new index with repeated labels
            expanded_df = valid_sites.loc[valid_sites.index.repeat(valid_sites['Cell_Count'])].copy()
            
            # 4. Flatten the features
            # 'features_temp' contains list of (N_cells, C, F) arrays.
            # We stack them into one massive (Total_Cells, C, F) array
            all_features_stacked = np.vstack(valid_sites['features_temp'].values)
            
            # 5. Assign to the expanded dataframe
            # Since parquet columns handle lists better than raw 3D numpy arrays, 
            # we convert the (C, F) array for each cell into a list (or flat list depending on preference).
            # Here we keep structure: list of lists
            logging.info(f"Formatting {len(expanded_df)} cells for Parquet export...")
            
            # We must assign the features row-by-row. 
            # Since all_features_stacked is aligned with expanded_df, we can listify it.
            # Note: This step can be memory intensive.
            expanded_df['single_cell_features'] = list(all_features_stacked)
            
            # Clean up temporary columns
            expanded_df['single_cell_features'] = expanded_df['single_cell_features'].apply(lambda x: x.tolist())
            expanded_df = expanded_df.drop(columns=['features_temp'])
            
            logging.info(f"Saving SINGLE CELL results to {args.out_data_path}")
            expanded_df.to_parquet(args.out_data_path, engine='pyarrow')

        else:
            # --- Standard Aggregation Mode ---
            logging.info("Standard Mode: Aggregating features to well level...")
            load_data['sum_features'] = site_features
            metadata_cols = ["Metadata_Well", "Metadata_Timepoint", "Metadata_Plate"] 
            
            def sum_arrays(series):
                return np.sum(np.stack(series.values), axis=0)

            agg_funcs = {'sum_features': sum_arrays, 'Cell_Count': 'sum'}
            for col in metadata_cols: 
                if col != 'Metadata_Well' and col in load_data.columns: 
                    agg_funcs[col] = 'first' 

            well_level_data = load_data.groupby('Metadata_Well').agg(agg_funcs).reset_index()
            
            # Calculate final weighted mean
            well_level_data['mean_features'] = well_level_data.apply(
                lambda row: (row['sum_features'] / row['Cell_Count']).tolist() if row['Cell_Count'] > 0 else np.zeros((expected_n_channels, FEATURE_LENGTH)).tolist(), 
                axis=1
            )
            well_level_data = well_level_data.drop(columns=['sum_features'])

            well_level_data.to_parquet(args.out_data_path, engine='pyarrow') 

            logging.info(f"Saving AGGREGATED results to {args.out_data_path}") 

        # Cleanup
        task_queue.close(); task_queue.join_thread()
        data_queue.close(); data_queue.join_thread()
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
    
    # --- ADDED FLAG ---
    parser.add_argument('--single_cell', action='store_true', help='If set, skips well-level aggregation and saves features for every single cell.')
    parser.add_argument('--save-coords', action='store_true', help='Saves a coordinate table for all valid cells.')
    
    args = parser.parse_args() 

    try: 
        mp.set_start_method('spawn', force=True) 
        logging.info("Set multiprocessing start method to 'spawn'.") 
    except RuntimeError: 
        logging.warning("Multiprocessing start method already set.") 

    main(args)