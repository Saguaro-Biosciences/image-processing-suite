import os
import argparse
import logging
import time
import tifffile
import pandas as pd
import numpy as np
import torch
# from PIL import Image  # No longer needed
from tqdm import tqdm
from queue import Empty

# Use torch.multiprocessing for efficient tensor sharing between processes
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Event

# Import scientific libraries
from skimage.measure import regionprops
from cellpose import models
# from transformers import AutoImageProcessor, AutoModel  # No longer needed

# --- 1. Setup Logging and Constants ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- MODEL AND PIPELINE CONFIGURATION ---
# MODEL_NAME = "timm/tf_efficientnetv2_l.in21k"  # No longer needed
CELLPOSE_MODEL = 'nuclei'
# FEATURE_LENGTH = 1280  # No longer needed
# BOX_SIZE = 200  # No longer needed
# INFERENCE_BATCH_SIZE = 256  # No longer needed

# --- Helper Functions ---
# def scale_to_8bit(image_16bit):  # No longer needed
#     """
#     Intelligently scales a 16-bit image to 8-bit.
#     """
#     min_val, max_val = np.min(image_16bit), np.max(image_16bit)
#     if max_val == min_val:
#         return np.zeros(image_16bit.shape, dtype=np.uint8)
#
#     scaled_image = 255.0 * (image_16bit.astype(np.float32) - min_val) / (max_val - min_val)
#     return scaled_image.astype(np.uint8)

# --- 2. Producer-Consumer Worker Functions ---

def producer_worker(task_queue, data_queue, worker_id, channels):
    """
    Producer Process: Handles CPU-bound I/O tasks ONLY.
    - Fetches a site task from the task_queue.
    - Loads the 4-channel image from disk.
    - Places the raw image array into the data_queue for the consumer.
    """
    logging.info(f"Producer-{worker_id} started.")
    try:
        channel_correction = [np.load(f'/home/ubuntu/data/{c}_illum.npy') for c in channels]
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
        try:
            all_channels = [tifffile.imread(path) / channel_correction[n] for n, path in enumerate(site_image_paths)]
            image_4ch = np.stack(all_channels, axis=-1)

            # Put the raw image data onto the queue for the GPU worker
            data_queue.put((site_id, image_4ch))

        except Exception as e:
            logging.error(f"Producer-{worker_id} failed on site {site_id}: {e}")
            # Put a placeholder to signal completion even on failure
            data_queue.put((site_id, None))

def consumer_worker(data_queue, results_dict, stop_event, worker_id, gpu_id=0):
    """
    Consumer Process: Handles GPU-bound segmentation.
    - Initializes Cellpose on its assigned GPU.
    - Pulls raw image data from the data_queue.
    - Runs Cellpose segmentation on the GPU.
    - Stores the number of segmented cells.
    """
    logging.info(f"Consumer-{worker_id} started, assigned to GPU:{gpu_id}.")

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        logging.warning(f"Consumer-{worker_id}: CUDA not available. Running on CPU.")

    # --- Load Cellpose model onto the assigned GPU ONCE ---
    logging.info(f"Consumer-{worker_id}: Loading Cellpose model onto {device}...")
    cell_model = models.CellposeModel(gpu=(device.type == 'cuda'), model_type=CELLPOSE_MODEL)

    # --- Remove feature extraction model loading ---
    # logging.info(f"Consumer-{worker_id}: Loading feature extraction model onto {device}...")
    # processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    # feature_model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    # half_box = BOX_SIZE // 2

    while not stop_event.is_set():
        try:
            item = data_queue.get(timeout=1)
            site_id, image_4ch = item
            # n_channels=image_4ch.shape[-1] # No longer needed

            # Handle case where producer failed to load image
            if image_4ch is None:
                results_dict[site_id] = 0  # Store 0 cells
                continue

            # --- 1. Run Cellpose Segmentation (GPU) ---
            masks, _, _ = cell_model.eval(image_4ch, diameter=100)
            props = regionprops(masks)

            # --- 2. Get cell count ---
            num_cells = len(props)

            # --- 3. Store the result (number of cells) ---
            results_dict[site_id] = num_cells

            # --- All feature extraction, cropping, and batching code removed ---

        except Empty:
            continue
        except Exception as e:
            # It's helpful to log which site failed if possible
            site_id_str = f"site {site_id}" if 'site_id' in locals() else "an unknown site"
            logging.error(f"Consumer-{worker_id} failed on {site_id_str}: {e}")
            # Ensure progress continues even on error
            if 'site_id' in locals():
                results_dict[site_id] = 0  # Store 0 cells on failure

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
    image_df = pd.read_csv("/home/ubuntu/data/Image.csv")
    not_failing_images = (image_df.filter(like='ImageQC_').sum(axis=1) < 2)
    load_data = load_data[not_failing_images].copy()
    tasks = [
        (index, [f"/home/ubuntu/data/{row[c]}" for c in channel_columns])
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
            Process(target=producer_worker, args=(task_queue, data_queue, i, args.channels), name=f"Producer-{i}")
            for i in range(args.max_workers)
        ]
        # **MODIFIED: Create a list of consumers**
        consumers = [
            Process(target=consumer_worker, args=(data_queue, results_dict, stop_event, i, 0), name=f"Consumer-{i}")
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
        original_indices = [task[0] for task in tasks]
        # site_results will now be a list of integers (cell counts)
        site_results = [results_dict[i] for i in original_indices]

        # Create a DataFrame with the cell counts
        results_df = pd.DataFrame({'cell_count': site_results}, index=original_indices)
        load_data = load_data.join(results_df)
        logging.info("Site-level cell counts calculated and merged.")

        # Aggregate Data to Well Level
        logging.info("Aggregating data to well level...")
        metadata_cols = ["Metadata_Well", "Metadata_Timepoint", "Metadata_Plate"]
        # Use the new 'cell_count' column
        df_subset = load_data[metadata_cols + ['cell_count']]

        agg_functions = {
            # Calculate the mean of the 'cell_count' column
            'cell_count': 'mean'
        }
        for col in metadata_cols:
            if col != 'Metadata_Well':
                agg_functions[col] = 'first'

        well_level_data = df_subset.groupby('Metadata_Well').agg(agg_functions).reset_index()
        # The line to convert features to list is no longer needed
        # well_level_data['mean_features'] = well_level_data['mean_features'].apply(lambda x: x.tolist())

        logging.info(f"Saving final results to {args.out_data_path}")
        well_level_data.to_parquet(args.out_data_path, engine='pyarrow')

        logging.info("Script finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the optimized cell image analysis pipeline.")

    parser.add_argument('--num-consumers', type=int, default=2, help='Number of parallel GPU consumer processes.')
    parser.add_argument('--max-workers', type=int, default=os.cpu_count() * 2,
                        help='Number of parallel CPU I/O producer processes.')
    parser.add_argument('--bucket-input', type=str, required=True, help='Name of the S3 bucket for input data.')
    parser.add_argument('--load-data-key', type=str, required=True, help='S3 key to the load_data.csv file.')
    parser.add_argument('--channels', nargs='+', type=str, required=True,
                        help='Channel list and order (first 3 are used for segmentation).')
    parser.add_argument('--out-data-path', type=str, required=True,
                        help='Local or S3 path for the final output Parquet file.')

    args = parser.parse_args()

    try:
        mp.set_start_method('spawn', force=True)
        logging.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        logging.warning("Multiprocessing start method already set.")

    main(args)