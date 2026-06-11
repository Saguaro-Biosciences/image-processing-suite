import io
import os
import time
import logging
import argparse
import subprocess

import fsspec
import tifffile
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from queue import Empty

# Use torch.multiprocessing for clean CUDA handling across processes
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Event

# --- 1. Setup Logging and Constants ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- PIPELINE CONFIGURATION ---
# NOTE: the Cellpose model/diameter are kept IDENTICAL to the original embedding run.
# This is what guarantees the segmentation reproduces the same cell numbering, so the
# is_dead_cell flags read from the existing single-cell parquet line up with the cells
# we re-segment here. Do not change these unless the source parquet was produced with
# different settings.
CELLPOSE_MODEL = 'nuclei'
BOX_SIZE = 200  # Box size in pixels for the per-cell crop (same as the original pipeline)


# --- Helper Functions ---
def scale_to_8bit(image_16bit):
    """
    Per-(crop, channel) min-max stretch to 8-bit, IDENTICAL to the inference pipeline
    (Cellpose_GPU_s3fs.py). This is the treatment EfficientNet sees, so the exported
    TIFFs must apply it too: it both (a) makes the crops match the model's training
    distribution and (b) makes them display correctly in any viewer (raw float crops
    with values >> 1 otherwise clip to a solid white mask).

    Because the background is masked to 0, min is ~0 and this reduces to 255 * x / max.
    """
    min_val, max_val = np.min(image_16bit), np.max(image_16bit)
    if max_val == min_val:
        return np.zeros(image_16bit.shape, dtype=np.uint8)

    scaled_image = 255.0 * (image_16bit.astype(np.float32) - min_val) / (max_val - min_val)
    return scaled_image.astype(np.uint8)


def parse_name_prefix(data_base_path):
    """
    Build the filename prefix from the data base path.

    Example:
        'Phenotypic_screen_HY-L022-custom_U2OS/Subset3_10uM_Run03/20250802T111229_48h_P13/'
    yields:
        'P13_Run03_48h'   ->  plate _ run _ timepoint

    plate     = last '_'-token of the last path segment    (20250802T111229_48h_P13 -> P13)
    timepoint = second-to-last token of the last segment   (... -> 48h)
    run       = last '_'-token of the second-to-last segment (Subset3_10uM_Run03 -> Run03)
    """
    parts = [p for p in data_base_path.strip('/').split('/') if p]
    last = parts[-1].split('_')
    prev = parts[-2].split('_') if len(parts) >= 2 else ['']
    plate = last[-1]
    timepoint = last[-2] if len(last) >= 2 else ''
    run = prev[-1]
    return f"{plate}_{run}_{timepoint}"


# --- 2. Producer-Consumer Worker Functions ---

def producer_worker(task_queue, data_queue, worker_id, channels, csv_image_key):
    """
    Producer Process: Handles CPU-bound I/O ONLY. Loads each channel TIFF, applies the
    illumination correction arrays, stacks the channels, and hands the stacked image to
    the consumer queue. The illumination correction MUST match the original run so that
    Cellpose re-segments the same cells.
    """
    logging.info(f"Producer-{worker_id} started.")

    if csv_image_key:
        try:
            # Loads and applies illumination correction arrays.
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

        max_retries = 5
        retries = 0
        success = False

        while retries < max_retries and not success:
            try:
                if csv_image_key:
                    # Loads in memory the channels for the specific site-task, illumination-corrected.
                    all_channels = [tifffile.imread(path) / channel_correction[n]
                                    for n, path in enumerate(site_image_paths)]
                else:
                    # Loads without correction.
                    all_channels = [tifffile.imread(path) for path in site_image_paths]

                image_4ch = np.stack(all_channels, axis=-1)
                data_queue.put((site_id, image_4ch))
                success = True  # Flag success to exit the retry loop

            except PermissionError:
                # Specifically catches [Errno 13] Permission denied
                logging.warning(f"Producer-{worker_id} PermissionError on site {site_id}. "
                                f"Restarting autofs... (Attempt {retries + 1}/{max_retries})")
                try:
                    subprocess.run(["sudo", "systemctl", "restart", "autofs"], check=True)
                    time.sleep(10)  # Give the OS a moment to remount
                except subprocess.CalledProcessError as sub_e:
                    logging.error(f"Producer-{worker_id} failed to restart autofs: {sub_e}")
                retries += 1

            except Exception as e:
                # Catches any other errors (corrupt file, etc.)
                logging.error(f"Producer-{worker_id} failed on site {site_id} with error: {e}")
                data_queue.put((site_id, None))
                break  # Exit the retry loop for non-permission errors

        # If the loop finished but we never succeeded (exhausted retries)
        if not success and retries >= max_retries:
            logging.error(f"Producer-{worker_id} permanently failed on site {site_id} "
                          f"after {max_retries} autofs restarts.")
            data_queue.put((site_id, None))


def consumer_worker(data_queue, results_dict, stop_event, worker_id, expected_n_channels,
                    gpu_id, channels, name_prefix, out_tiff_dir, meta_map, dead_map):
    """
    GPU Process: Cellpose segmentation ONLY (no embedding / XGBoost inference).

    For each segmented cell that is ALIVE (per the dead_map read from the existing
    single-cell parquet), writes one masked crop TIFF PER CHANNEL to out_tiff_dir on S3:

        {name_prefix}_{Well}_{Site}_cell{k}_{channel}.tiff

    where k is the cell index within the site, matching the original pipeline's Cell_Index
    (same regionprops label order + same edge-skip). A per-site cell-count consistency check
    against the parquet guards against silent misalignment.
    """
    # It is important to set CUDA_VISIBLE_DEVICES and import GPU libs INSIDE the worker, as
    # this is deployed on partitioned GPUs.
    import os
    import gc

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    internal_device_id = 0

    from skimage.measure import regionprops
    from cellpose import models

    logging.info(f"Consumer-{worker_id} started. GPU: {gpu_id}")
    device = torch.device(f"cuda:{internal_device_id}" if torch.cuda.is_available() else "cpu")

    # --- Load Cellpose (identical configuration to the original run) ---
    cell_model = models.CellposeModel(gpu=(device.type == 'cuda'), model_type=CELLPOSE_MODEL, device=device)

    half_box = BOX_SIZE // 2

    def record(s_id, status, n_segmented=0, n_alive=0, n_dead=0, mismatch=False):
        results_dict[s_id] = {
            'status': status,
            'n_segmented': n_segmented,
            'n_alive_exported': n_alive,
            'n_dead_skipped': n_dead,
            'mismatch': mismatch,
        }

    while not stop_event.is_set():
        try:
            # Get a site/FOV stacked image from the data queue.
            item = data_queue.get(timeout=1)
            site_id, image_4ch = item
            site_key = int(site_id)

            # Check the stacked image has the expected number of channels.
            if image_4ch is None or image_4ch.shape[-1] != expected_n_channels:
                record(site_id, 'empty')
                continue

            n_channels = expected_n_channels
            well, site = meta_map.get(site_key, ('NA', site_key))

            # --- 1. Run Cellpose (segmentation on the first 3 channels via the stack) ---
            try:
                masks, _, _ = cell_model.eval(image_4ch, diameter=100)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                masks, _, _ = cell_model.eval(image_4ch, diameter=100)

            props = regionprops(masks)
            if not props:
                record(site_id, 'empty')
                continue

            h, w, _ = image_4ch.shape

            # --- 2. Build the ordered list of non-edge cells ---
            # The k-th kept cell == the original pipeline's Cell_Index == k, because we
            # iterate regionprops in label order and apply the SAME edge-skip rule. Only
            # kept (non-edge) cells advance k, exactly as the original code did.
            kept = []  # (k, y1, y2, x1, x2, target_id)
            k = 0
            for prop in props:
                y_center, x_center = map(int, prop.centroid)
                # Remove cells at the edges of the image (cannot crop a full box).
                if (y_center - half_box < 0) or (y_center + half_box > h) or \
                   (x_center - half_box < 0) or (x_center + half_box > w):
                    continue
                y1, y2 = y_center - half_box, y_center + half_box
                x1, x2 = x_center - half_box, x_center + half_box
                kept.append((k, y1, y2, x1, x2, prop.label))
                k += 1

            n_segmented = k
            if n_segmented == 0:
                record(site_id, 'empty')
                continue

            # --- 3. Determine alive cells from the existing single-cell parquet ---
            is_dead_arr = dead_map.get(site_key) if dead_map is not None else None

            # SAFEGUARD: if the re-segmentation does not reproduce the original cell count
            # for this site, the Cell_Index alignment is broken -> skip to avoid mislabeling.
            if is_dead_arr is not None and len(is_dead_arr) != n_segmented:
                logging.warning(
                    f"Consumer-{worker_id}: SITE {site_id} cell-count MISMATCH "
                    f"(re-segmented={n_segmented}, parquet={len(is_dead_arr)}). Skipping export."
                )
                record(site_id, 'mismatch', n_segmented=n_segmented, mismatch=True)
                continue

            # --- 4. Write masked per-channel crops for ALIVE cells ---
            n_alive, n_dead = 0, 0
            for (cell_k, y1, y2, x1, x2, target_id) in kept:
                if is_dead_arr is not None and bool(is_dead_arr[cell_k]):
                    n_dead += 1
                    continue

                binary_mask = (masks[y1:y2, x1:x2] == target_id)  # (BOX_SIZE, BOX_SIZE) bool
                for ch_idx in range(n_channels):
                    # Masked crop for this channel: cell pixels keep their (corrected) intensity,
                    # everything outside the cell mask is zeroed. Then apply the SAME 8-bit
                    # scaling the inference pipeline feeds EfficientNet, so the exported TIFF is
                    # exactly what the model sees (and displays correctly instead of clipping to
                    # a white mask).
                    masked = image_4ch[y1:y2, x1:x2, ch_idx] * binary_mask
                    crop = scale_to_8bit(masked)
                    fname = f"{name_prefix}_{well}_{site}_cell{cell_k}_{channels[ch_idx]}.tiff"
                    uri = f"{out_tiff_dir}/{fname}"

                    # Write TIFF into an in-memory buffer (seekable, as tifffile needs) and
                    # stream the complete bytes to S3 via fsspec.
                    buf = io.BytesIO()
                    tifffile.imwrite(buf, crop)
                    with fsspec.open(uri, 'wb') as f:
                        f.write(buf.getvalue())
                n_alive += 1

            record(site_id, 'success', n_segmented=n_segmented, n_alive=n_alive, n_dead=n_dead)
            logging.info(f"Consumer-{worker_id}: SITE {site_id} exported {n_alive} alive cells "
                         f"({n_dead} dead skipped, {n_segmented} segmented).")

        # Handle exceptions per-site to skip a site instead of killing the pipeline.
        except Empty:
            continue
        except Exception as e:
            logging.error(f"Consumer-{worker_id} failed: {e}")
            if 'site_id' in locals():
                record(site_id, 'error')


# --- 3. Main Execution Block ---
def main(args):
    """
    Orchestrates the Producer-Consumer pipeline: segment each site, then export masked
    per-channel crops of the alive cells as TIFFs to S3.
    """
    # Wake up the image folder mount.
    try:
        os.listdir(args.data_base_path)
    except Exception:
        pass
    logging.info(f"Starting TIFF export with parameters: {args}")

    # Filename prefix from the data base path (e.g. P13_Run03_48h), overridable via CLI.
    name_prefix = args.name_prefix or parse_name_prefix(args.data_base_path)
    logging.info(f"Using filename prefix: {name_prefix}")

    # --- Load load_data CSV ---
    s3_input_path_load = f"s3://{args.bucket_input}/{args.load_data_key}"
    try:
        logging.info(f"Reading load_data CSV from {s3_input_path_load}")
        load_data = pd.read_csv(s3_input_path_load)
    except Exception as e:
        logging.error(f"Failed to read load_data CSV from S3. Error: {e}")
        return

    channel_columns = [f'FileName_{c}' for c in args.channels]

    # --- QC filtering (MUST match the original run so __index_level_0__ aligns) ---
    if getattr(args, "csv_image_key", None):
        image_df = pd.read_csv(f"{args.csv_image_key}/Image.csv")
        not_failing_images = (image_df.filter(like='ImageQC_').sum(axis=1) < 1)
        load_data = load_data[not_failing_images].copy()
    else:
        logging.warning("No csv_image_key provided -> skipping QC filtering AND illumination "
                        "correction. Segmentation may NOT reproduce the original cell numbering, "
                        "which can break alive/dead alignment.")

    # --- Read dead-cell flags from the existing single-cell parquet ---
    # Only the lightweight columns are read (NOT single_cell_features). dead_map maps each
    # site index (__index_level_0__, == load_data row index) to a boolean array of is_dead_cell
    # ordered by Cell_Index.
    dead_map = None
    if args.single_cell_parquet:
        logging.info(f"Reading dead-cell flags from {args.single_cell_parquet}")
        try:
            sc = pd.read_parquet(args.single_cell_parquet,
                                 columns=['__index_level_0__', 'Cell_Index', 'is_dead_cell'])
            dead_map = {}
            for site_idx, grp in sc.groupby('__index_level_0__'):
                ordered = grp.sort_values('Cell_Index')
                dead_map[int(site_idx)] = ordered['is_dead_cell'].to_numpy().astype(bool)
            logging.info(f"Loaded dead-cell flags for {len(dead_map)} sites.")
        except Exception as e:
            logging.error(f"Failed to read single-cell parquet ({e}). Exporting ALL cells "
                          f"(no dead-cell filtering).")
            dead_map = None
    else:
        logging.warning("No --single-cell-parquet provided -> exporting ALL segmented cells "
                        "(cannot filter dead cells).")

    # --- Build {site index: (well, site)} for filenames ---
    has_well = 'Metadata_Well' in load_data.columns
    has_site = 'Metadata_Site' in load_data.columns
    meta_map = {}
    for idx, row in load_data.iterrows():
        well = row['Metadata_Well'] if has_well else 'NA'
        site = row['Metadata_Site'] if has_site else idx
        meta_map[int(idx)] = (well, site)

    # --- Prepare tasks for producers ---
    tasks = [
        (index, [f"{args.data_base_path}/{row[c]}" for c in channel_columns])
        for index, row in load_data.iterrows()
    ]
    num_tasks = len(tasks)
    logging.info(f"Prepared {num_tasks} sites for processing.")

    # --- S3 output directory for the TIFFs ---
    out_base = args.out_data_path.rsplit('/', 1)[0]
    out_tiff_dir = f"{out_base}/cell_tiffs/{name_prefix}"
    logging.info(f"TIFFs will be written under: {out_tiff_dir}")

    # --- Initialize multiprocessing environment ---
    with mp.Manager() as manager:
        task_queue = Queue()
        data_queue = Queue(maxsize=args.num_consumers)
        results_dict = manager.dict()
        stop_event = Event()

        for task in tasks:
            task_queue.put(task)
        for _ in range(args.max_workers):
            task_queue.put(None)

        # --- Start producers ---
        producers = [
            Process(target=producer_worker,
                    args=(task_queue, data_queue, i, args.channels, args.csv_image_key),
                    name=f"Producer-{i}")
            for i in range(args.max_workers)
        ]

        # --- Start consumers ---
        expected_n_channels = len(args.channels)
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            logging.warning("No GPUs detected. Defaulting to device index 0.")
            available_gpus = 1

        consumers = [Process(
            target=consumer_worker,
            args=(data_queue, results_dict, stop_event, i, expected_n_channels,
                  i % available_gpus, args.channels, name_prefix, out_tiff_dir,
                  meta_map, dead_map),
            name=f"Consumer-{i}"
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

        logging.info("All sites processed. Signaling consumers to shut down.")
        stop_event.set()
        for c in consumers: c.join()

        task_queue.close(); task_queue.join_thread()
        data_queue.close(); data_queue.join_thread()
        logging.info("All processes have completed.")

        # --- Summary ---
        rows = []
        n_mismatch = 0
        for idx in [t[0] for t in tasks]:
            res = results_dict.get(idx, {'status': 'missing', 'n_segmented': 0,
                                         'n_alive_exported': 0, 'n_dead_skipped': 0,
                                         'mismatch': False})
            well, site = meta_map.get(int(idx), ('NA', idx))
            if res.get('mismatch'):
                n_mismatch += 1
            rows.append({
                'Index': idx,
                'Metadata_Well': well,
                'Metadata_Site': site,
                'status': res['status'],
                'n_segmented': res.get('n_segmented', 0),
                'n_alive_exported': res.get('n_alive_exported', 0),
                'n_dead_skipped': res.get('n_dead_skipped', 0),
                'mismatch': res.get('mismatch', False),
            })
        summary = pd.DataFrame(rows)
        summary_path = args.out_data_path.replace('.parquet', '_export_summary.csv')
        summary.to_csv(summary_path, index=False)

        total_exported = int(summary['n_alive_exported'].sum())
        logging.info(f"Export complete. {total_exported} alive cells exported across "
                     f"{len(summary)} sites. Summary -> {summary_path}")
        if dead_map is not None and n_mismatch > 0:
            logging.warning(f"{n_mismatch} sites had a cell-count mismatch vs the parquet and were "
                            f"NOT exported. If this count is large, the segmentation is not "
                            f"reproducing the original numbering (check Cellpose/torch version and "
                            f"that the SAME illumination correction is applied).")
        logging.info("Script finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Segment cell-profiling images with Cellpose and export masked per-channel "
                    "TIFF crops of the ALIVE cells (alive/dead read from an existing single-cell "
                    "parquet). No embedding/XGBoost inference.")
    parser.add_argument('--bucket-input', type=str, required=True,
                        help='Base input bucket where the load_data CSV lives.')
    parser.add_argument('--data-base-path', type=str, required=True,
                        help='Base path where the raw images are stored, e.g. '
                             '/home/storage/Phenotypic_screen_.../Subset3_10uM_Run03/2025..._48h_P13. '
                             'The filename prefix (P13_Run03_48h) is parsed from this.')
    parser.add_argument('--num-consumers', type=int, default=2,
                        help='Number of parallel Cellpose GPU workers.')
    parser.add_argument('--max-workers', type=int, default=24,
                        help='Number of CPU producers that prepare the data.')
    parser.add_argument('--load-data-key', type=str, required=True,
                        help='S3 key (within --bucket-input) to the load_data CSV.')
    parser.add_argument('--csv-image-key', type=str, required=False,
                        help='Path to the folder with Image.csv (for QC filtering) and the '
                             '<channel>_illum.npy correction arrays. MUST match the original run.')
    parser.add_argument('--channels', nargs='+', type=str, required=True,
                        help='Channel prefixes as they appear in load_data (FileName_<channel>). '
                             'Order is paramount: it must match the original run, and the channel '
                             'name is used in the output filename.')
    parser.add_argument('--out-data-path', type=str, required=True,
                        help='S3 path (…/something.parquet). TIFFs go to '
                             '<dir-of-out-data-path>/cell_tiffs/<prefix>/ and a summary CSV next to it.')
    parser.add_argument('--single-cell-parquet', type=str, default=None,
                        help='Existing per-run single-cell parquet (e.g. embeddings_P13_48_single_cell.parquet) '
                             'with __index_level_0__, Cell_Index, is_dead_cell. Used to export only '
                             'alive cells. If omitted, ALL segmented cells are exported.')
    parser.add_argument('--name-prefix', type=str, default=None,
                        help='Override the auto-parsed <plate>_<run>_<timepoint> filename prefix.')

    args = parser.parse_args()
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main(args)
