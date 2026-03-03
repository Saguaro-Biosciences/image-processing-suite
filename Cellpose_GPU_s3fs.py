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

def consumer_worker(data_queue, results_dict, stop_event, worker_id, expected_n_channels, gpu_id=0, single_cell_mode=False, xgb_model_path=None, filter_dead_cells=False): 
    import os
    import gc 
    import xgboost as xgb
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    internal_device_id = 0
    
    from skimage.measure import regionprops 
    from cellpose import models 
    from transformers import AutoImageProcessor, AutoModel
    
    logging.info(f"Consumer-{worker_id} started. GPU: {gpu_id} | Single Cell: {single_cell_mode} | XGB: {bool(xgb_model_path)}") 
    device = torch.device(f"cuda:{internal_device_id}" if torch.cuda.is_available() else "cpu") 
    
    # --- Load Deep Learning Models --- 
    cell_model = models.CellposeModel(gpu=(device.type == 'cuda'), model_type=CELLPOSE_MODEL, device=device) 
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME) 
    feature_model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval() 
    
    # --- Load XGBoost Model ---
    bst = None
    if xgb_model_path:
        logging.info(f"Consumer-{worker_id}: Loading XGBoost model...")
        bst = xgb.Booster()
        bst.load_model(xgb_model_path)

    half_box = BOX_SIZE // 2 
    current_batch_size = INFERENCE_BATCH_SIZE 

    def return_empty_result(s_id):
        if single_cell_mode:
            results_dict[s_id] = (np.zeros((0, expected_n_channels, FEATURE_LENGTH), dtype=np.float32), 0, [], np.array([], dtype=bool))
        else:
            results_dict[s_id] = (np.zeros((expected_n_channels, FEATURE_LENGTH), dtype=np.float32), 0, [], np.array([], dtype=bool))

    while not stop_event.is_set(): 
        try: 
            item = data_queue.get(timeout=1) 
            site_id, image_4ch = item 
            
            if image_4ch is None or image_4ch.shape[-1] != expected_n_channels: 
                return_empty_result(site_id)
                continue 
            
            n_channels = expected_n_channels
            
            # --- 1. Run Cellpose --- 
            try:
                masks, _, _ = cell_model.eval(image_4ch, diameter=100) 
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                masks, _, _ = cell_model.eval(image_4ch, diameter=100) 

            props = regionprops(masks) 
            if not props: 
                return_empty_result(site_id)
                continue 

            # --- 2. Crop Cells --- 
            all_cell_crops = [] 
            cell_coords = []
            h, w, _ = image_4ch.shape 
            for prop in props: 
                y_center, x_center = map(int, prop.centroid) 
                if (y_center - half_box < 0) or (y_center + half_box > h) or (x_center - half_box < 0) or (x_center + half_box > w):
                    continue
                
                target_id = prop.label 
                y1, y2, x1, x2 = y_center - half_box, y_center + half_box, x_center - half_box, x_center + half_box 
                mask_crop = masks[y1:y2, x1:x2] 
                binary_mask = (mask_crop == target_id)[:, :, np.newaxis] 
                all_cell_crops.append(image_4ch[y1:y2, x1:x2, :] * binary_mask)
                cell_coords.append((y_center, x_center)) 

            if not all_cell_crops:
                return_empty_result(site_id)
                continue
            
            # --- 3. Extract Features --- 
            batch_pil_images = [] 
            for cell_crop in all_cell_crops: 
                for ch in range(n_channels): 
                    scaled_8bit = scale_to_8bit(cell_crop[:, :, ch]) 
                    batch_pil_images.append(Image.fromarray(scaled_8bit).convert("RGB")) 

            site_features, idx = [], 0
            total_images = len(batch_pil_images)

            while idx < total_images:
                end_idx = min(idx + current_batch_size, total_images)
                mini_batch = batch_pil_images[idx : end_idx]
                try:
                    inputs = processor(images=mini_batch, return_tensors="pt").to(device) 
                    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.float16): 
                        outputs = feature_model(**inputs) 
                    site_features.append(outputs.pooler_output.cpu().to(torch.float32).numpy())
                    idx = end_idx 
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    current_batch_size = max(1, current_batch_size // 2)
                    if current_batch_size == 1:
                        site_features = []
                        break
            
            if len(site_features) > 0:
                n_cells = len(all_cell_crops)
                reshaped_features = np.vstack(site_features).reshape(n_cells, n_channels, FEATURE_LENGTH) 
                
                # --- 4. XGBoost Inference & Filtering Logic ---
                is_dead = np.zeros(n_cells, dtype=bool)
                if bst is not None:
                    # Flatten features to 2D for XGBoost: [N_cells, Channels * Features]
                    flat_features = reshaped_features.reshape(n_cells, -1) 
                    dtrain = xgb.DMatrix(flat_features)
                    preds = bst.predict(dtrain)
                    is_dead = (preds > 0.5) # Boolean array of dead cells
                
                if single_cell_mode:
                    # Return all features, count, coords, and the flags
                    results_dict[site_id] = (reshaped_features, n_cells, cell_coords, is_dead)
                else:
                    if bst is not None and filter_dead_cells:
                        alive_mask = ~is_dead
                        alive_count = np.sum(alive_mask)
                        if alive_count > 0:
                            sum_feats = np.sum(reshaped_features[alive_mask], axis=0)
                        else:
                            sum_feats = np.zeros((expected_n_channels, FEATURE_LENGTH), dtype=np.float32)
                        
                        # Return summed LIVE features, LIVE count
                        results_dict[site_id] = (sum_feats, alive_count, cell_coords, is_dead)
                    else:
                        # Standard aggregate mode without filtering
                        results_dict[site_id] = (np.sum(reshaped_features, axis=0), n_cells, cell_coords, is_dead)
                
                logging.info(f"Consumer-{worker_id}: Finished SITE {site_id} ({n_cells} total cells, {np.sum(is_dead)} dead).")
            else:
                return_empty_result(site_id)

        except Empty: continue 
        except Exception as e: 
            logging.error(f"Consumer-{worker_id} failed: {e}") 
            if 'site_id' in locals(): return_empty_result(site_id)

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

        consumers = [Process(
            target=consumer_worker, 
            args=(data_queue, results_dict, stop_event, i, expected_n_channels, i % available_gpus, args.single_cell, args.xgb_model_path, args.filter_dead_cells)
        ) for i in range(args.num_consumers)]

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

        task_queue.close(); task_queue.join_thread()
        data_queue.close(); data_queue.join_thread()
        logging.info("All processes have completed.") 

        # --- PROCESS RESULTS --- 
        original_indices = [task[0] for task in tasks]
        raw_results = [results_dict[i] for i in original_indices]
        
        site_features = [item[0] for item in raw_results]
        site_counts = [item[1] for item in raw_results]
        site_coords = [item[2] for item in raw_results]
        site_dead_flags = [item[3] for item in raw_results] 

        # --- Prepare Aggregated Features ---
        aggregated_features = []
        final_site_counts = []

        if args.single_cell:
            # Manually calculate sums from single-cell arrays for the well-level output
            for feats, flags in zip(site_features, site_dead_flags):
                if len(feats) == 0:
                    aggregated_features.append(np.zeros((expected_n_channels, FEATURE_LENGTH), dtype=np.float32))
                    final_site_counts.append(0)
                else:
                    if args.xgb_model_path and args.filter_dead_cells:
                        alive_mask = ~flags
                        alive_count = np.sum(alive_mask)
                        if alive_count > 0:
                            aggregated_features.append(np.sum(feats[alive_mask], axis=0))
                        else:
                            aggregated_features.append(np.zeros((expected_n_channels, FEATURE_LENGTH), dtype=np.float32))
                        final_site_counts.append(alive_count)
                    else:
                        aggregated_features.append(np.sum(feats, axis=0))
                        final_site_counts.append(len(feats))
        else:
            # Consumers already filtered and summed
            aggregated_features = site_features
            final_site_counts = site_counts

        load_data['Cell_Count'] = final_site_counts
        counts_out_path = args.out_data_path.replace('.parquet', '_counts.csv')
        load_data.to_csv(counts_out_path, index=False)

        # --- OUTPUT 1: Coordinates ---
        if args.save_coords:
            coords_records = []
            for idx, coords_list, dead_flags in zip(original_indices, site_coords, site_dead_flags):
                well = load_data.loc[idx, 'Metadata_Well']
                site = load_data.loc[idx, 'Metadata_Site'] if 'Metadata_Site' in load_data.columns else str(idx)
                for cell_idx, (y, x) in enumerate(coords_list):
                    is_dead = dead_flags[cell_idx] if len(dead_flags) > 0 else False
                    coords_records.append({'Cell_ID': f"{well}_{site}_cell{cell_idx}", 'Y_Center': y, 'X_Center': x, 'Is_Dead': is_dead})
            if coords_records:
                pd.DataFrame(coords_records).to_parquet(args.out_data_path.replace('.parquet', '_coords.parquet'), engine='pyarrow')

        # --- OUTPUT 2: ALWAYS Output Well-Level Aggregation ---
        logging.info("Aggregating features to well level...")
        load_data_agg = load_data.copy()
        load_data_agg['sum_features'] = aggregated_features
        metadata_cols = ["Metadata_Well", "Metadata_Timepoint", "Metadata_Plate"] 
        
        agg_funcs = {'sum_features': lambda s: np.sum(np.stack(s.values), axis=0), 'Cell_Count': 'sum'}
        for col in metadata_cols: 
            if col != 'Metadata_Well' and col in load_data_agg.columns: agg_funcs[col] = 'first' 

        well_level_data = load_data_agg.groupby('Metadata_Well').agg(agg_funcs).reset_index()
        well_level_data['mean_features'] = well_level_data.apply(
            lambda row: (row['sum_features'] / row['Cell_Count']).tolist() if row['Cell_Count'] > 0 else np.zeros((expected_n_channels, FEATURE_LENGTH)).tolist(), 
            axis=1
        )
        well_level_data = well_level_data.drop(columns=['sum_features'])
        
        agg_out_path = args.out_data_path.replace('.parquet', '_well_aggregated.parquet') if args.single_cell else args.out_data_path
        well_level_data.to_parquet(agg_out_path, engine='pyarrow') 
        logging.info(f"Saved well-aggregated results to {agg_out_path}")

        # --- OUTPUT 3: OOM-Safe Single-Cell Features ---
        if args.single_cell:
            logging.info("SINGLE CELL MODE DETECTED: Formatting single-cell output safely...")
            
            # Use raw lengths, not filtered cell counts, to ensure rows match extracted arrays
            valid_mask = [len(f) > 0 for f in site_features]
            valid_indices = [i for i, v in enumerate(valid_mask) if v]
            
            if not valid_indices:
                logging.warning("No cells detected in dataset. Saving empty parquet.")
                sc_out_path = args.out_data_path.replace('.parquet', '_single_cell.parquet')
                load_data.to_parquet(sc_out_path, engine='pyarrow')
                return

            valid_sites = load_data.iloc[valid_indices].copy()
            valid_features = [site_features[i] for i in valid_indices]
            valid_flags = [site_dead_flags[i] for i in valid_indices]

            # Expand the dataframe
            repeats = [len(f) for f in valid_features]
            expanded_df = valid_sites.loc[valid_sites.index.repeat(repeats)].copy()
            expanded_df['Cell_Index'] = expanded_df.groupby(level=0).cumcount()
            
            # Free up RAM aggressively before the big stack
            del load_data, load_data_agg, well_level_data, site_features, site_dead_flags, valid_sites
            
            logging.info("Stacking single-cell features...")
            stacked_features = np.vstack(valid_features)
            stacked_features_flat = stacked_features.reshape(stacked_features.shape[0], -1)
            
            expanded_df['single_cell_features'] = list(stacked_features_flat)
            
            if args.xgb_model_path:
                expanded_df['is_dead_cell'] = np.concatenate(valid_flags)
                
            if 'Cell_Count' in expanded_df.columns:
                expanded_df = expanded_df.drop(columns=['Cell_Count'])
                
            sc_out_path = args.out_data_path.replace('.parquet', '_single_cell.parquet')
            logging.info(f"Saving SINGLE CELL results to {sc_out_path}...")
            expanded_df.to_parquet(sc_out_path, engine='pyarrow')

        logging.info("Script finished successfully.")


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Run cell image analysis pipeline. Takes into account image level QC, XGboost model assesment for dead cells. Singel cell and well level embedding extraction using Efficientnet.")
    parser.add_argument('--data-base-path', type=str, required=True) 
    parser.add_argument('--num-consumers', type=int, default=2) 
    parser.add_argument('--max-workers', type=int, default=24) 
    parser.add_argument('--bucket-input', type=str, required=True) 
    parser.add_argument('--load-data-key', type=str, required=True)
    parser.add_argument('--csv-image-key', type=str, required=False)
    parser.add_argument('--channels', nargs='+', type=str, required=True)
    parser.add_argument('--out-data-path', type=str, required=True) 
    parser.add_argument('--single_cell', action='store_true')
    parser.add_argument('--save-coords', action='store_true')
    parser.add_argument('--xgb-model-path', type=str, default=None, help='Path to XGBoost json model to identify dead cells.')
    parser.add_argument('--filter-dead-cells', action='store_true', help='If provided in aggregate mode, dead cells will be excluded from the sum.')
    
    args = parser.parse_args() 
    try: mp.set_start_method('spawn', force=True) 
    except RuntimeError: pass 
    main(args)