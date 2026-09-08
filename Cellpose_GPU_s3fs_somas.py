import tempfile
import shutil
import os
import fsspec
import subprocess
import pyarrow as pa
import pyarrow.parquet as pq
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
 
# --- 1. Setup Logging and Consants ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
 
# --- MODEL AND PIPELINE CONFIGURATION ---
MODEL_NAME = "timm/tf_efficientnetv2_l.in21k"
CELLPOSE_MODEL = 'nuclei'
FEATURE_LENGTH = 1280
BOX_SIZE = 200  # Box size in number of pixes for the size of the cell crop
INFERENCE_BATCH_SIZE = 1000  # Do not change, it is the initial number of batch cells to assess, it will n/2 until the GPU is able to handle the load
 
# DNA channel auto-detection, for the DNA-contrast segmentation filter.
# Matched against --channels entries (case-insensitive substring match).
DNA_CHANNEL_NAME_PATTERNS = ["dna", "hoechst", "dapi", "nuclei", "nucleus"]
 
 
# --- Helper Functions ---
def scale_to_8bit(image_16bit):
    """
    Scales a 16-bit image to 8-bit, as per model requirements.
    """
    min_val, max_val = np.min(image_16bit), np.max(image_16bit)
    if max_val == min_val:
        return np.zeros(image_16bit.shape, dtype=np.uint8)
 
    scaled_image = 255.0 * (image_16bit.astype(np.float32) - min_val) / (max_val - min_val)
    return scaled_image.astype(np.uint8)
 
 
def resolve_dna_channel_index(channels):
    """
    Finds which index in args.channels corresponds to the DNA/nuclear stain,
    by matching common naming patterns. Raises clearly rather than silently
    guessing wrong, since a wrong index would silently corrupt every
    dna_contrast() computation downstream.
    """
    for i, ch in enumerate(channels):
        if any(pat in ch.lower() for pat in DNA_CHANNEL_NAME_PATTERNS):
            logging.info(f"Resolved DNA channel: index {i} ('{ch}')")
            return i
    raise ValueError(
        f"Could not auto-detect DNA channel from --channels {channels}. "
        f"Expected one of {DNA_CHANNEL_NAME_PATTERNS} in a channel name. "
        f"Pass --dna-channel-index explicitly instead."
    )
 
 
# --- DNA-contrast soma segmentation (validated on P620, Cellpose v4/Cellpose-SAM) ---
def dna_contrast(masks, dna, band_px=(2, 12), min_band=20):
    """
    Computes, for every segmented object, a dimensionless log2 contrast score
    between the object's own DNA intensity and its LOCAL background DNA
    intensity (a thin ring of background pixels near that specific object,
    not a single global background value). Real nuclei should show strong
    positive contrast; hallucinated objects (segmented from background
    texture/noise, with no real DNA under them) should sit near zero.
 
    band_px: (inner, outer) distance in pixels from any object's edge that
    defines its local background ring.
    min_band: minimum number of local background pixels required to trust
    the local estimate; objects with fewer (e.g. tightly packed in a
    rosette, surrounded by other objects) fall back to the image-wide
    background median instead.
    """
    import scipy.ndimage as ndi
 
    labs = np.unique(masks)
    labs = labs[labs != 0]
    fg = masks > 0
    d, (iy, ix) = ndi.distance_transform_edt(~fg, return_indices=True)
    band = (~fg) & (d >= band_px[0]) & (d <= band_px[1])
    bg_lab = np.where(band, masks[iy, ix], 0)  # each band pixel -> nearest object's label
 
    obj = np.asarray(ndi.mean(dna, masks, labs), float)
    bg = np.asarray(ndi.mean(dna, bg_lab, labs), float)
 
    cnt = np.asarray(ndi.sum(band, bg_lab, labs), float)
    glob = float(np.median(dna[~fg]))
    bg = np.where(cnt >= min_band, bg, glob)
 
    return labs, np.log2((obj + 1.0) / (bg + 1.0)), cnt
 
 
def per_image_threshold(x, min_objects=30, bic_margin=10, clamp=(0.2, 3.0), default=1.0):
    """
    Derives a per-image contrast threshold separating real objects from
    hallucinated ones, using a 2-component Gaussian Mixture Model fit to
    THIS image's contrast score distribution -- with guards against forcing
    a split where none is statistically supported (too few objects, or a
    genuinely unimodal distribution), and against trusting an Otsu threshold
    that lands somewhere implausible.
 
    Returns (threshold, explanation_string).
    """
    from skimage.filters import threshold_otsu
    from sklearn.mixture import GaussianMixture
 
    x = x[np.isfinite(x)]
    if len(x) < min_objects:
        return default, f"only {len(x)} objects -> fell back to default {default}"
    X = x.reshape(-1, 1)
    b1 = GaussianMixture(1, random_state=0).fit(X).bic(X)
    b2 = GaussianMixture(2, random_state=0, n_init=5).fit(X).bic(X)
    if b2 > b1 - bic_margin:  # 2 components not clearly better
        return default, f"unimodal (BIC 1c={b1:.0f} vs 2c={b2:.0f}) -> no split"
    t = float(threshold_otsu(x))
    if not (clamp[0] <= t <= clamp[1]):
        return default, f"Otsu gave {t:.2f}, outside clamp {clamp} -> fell back"
    return t, f"per-image threshold {t:.2f} (BIC 1c={b1:.0f} 2c={b2:.0f})"
 
 
def segment_somas_dna_contrast(cell_model, image_4ch, dna_index, diameter=100,
                                 min_size_filter=None):
    """
    Drop-in replacement for cell_model.eval(image_4ch, diameter=100) that
    additionally filters hallucinated objects via per-object DNA contrast.
 
    "Segment everything, then filter" strategy: runs Cellpose with its own
    rejection criteria disabled (flow_threshold=0, min_size=-1) so nothing
    is pre-censored, then removes objects that fail EITHER of two independent
    criteria:
      1. per-object DNA contrast (dna_contrast + per_image_threshold) --
         catches hallucinated objects with no real DNA signal underneath.
      2. minimum object area (min_size_filter, in px) -- catches objects
         that DO have real DNA signal but are morphologically abnormal
         (e.g. shrunken/pyknotic nuclei from dying/dead cells), which the
         contrast filter alone would not flag since dead-cell nuclei can
         still show real, positive DNA contrast. Relevant for positive
         control wells where dead/dying cells are expected.
 
    image_4ch: channels-last array (H, W, C), matching this pipeline's stacking.
    dna_index: index into the channel (last) axis for the DNA/nuclear stain.
    min_size_filter: if set, objects with area below this value (px) are
    dropped regardless of their DNA contrast score. If None, only the
    contrast criterion is applied.
 
    Returns (masks, info) -- masks matches the shape/dtype cell_model.eval
    would normally return, so downstream regionprops(masks) is unaffected.
    """
    import fastremap
    from skimage.measure import regionprops
 
    masks, flows, styles = cell_model.eval(
        image_4ch, diameter=diameter, flow_threshold=0, min_size=-1
    )
    if masks.max() == 0:
        return masks, {"n_in": 0, "n_kept": 0, "n_dropped_contrast": 0,
                        "n_dropped_size": 0, "n_dropped_both": 0,
                        "thr": None, "why": "no objects detected"}
 
    dna = image_4ch[..., dna_index].astype(np.float32)
    labs, x, band_cnt = dna_contrast(masks, dna)
    thr, why = per_image_threshold(x)
 
    fails_contrast = ~(x >= thr)
 
    # Minimum size criterion, independent of contrast -- catches morphologically
    # abnormal (e.g. shrunken/pyknotic dead-cell) objects that still pass the
    # contrast check since they can retain real, positive DNA signal.
    fails_size = np.zeros_like(fails_contrast)
    if min_size_filter is not None:
        area_by_label = {p.label: p.area for p in regionprops(masks.astype(int))}
        areas = np.array([area_by_label.get(l, 0) for l in labs])
        fails_size = areas < min_size_filter
 
    drop_mask = fails_contrast | fails_size
    drop = labs[drop_mask]
 
    if drop.size:
        masks = fastremap.mask(masks, drop.astype(masks.dtype))
    fastremap.renumber(masks, in_place=True)
 
    info = {
        "n_in": len(labs), "n_kept": int(masks.max()),
        "n_dropped_contrast": int(np.sum(fails_contrast & ~fails_size)),
        "n_dropped_size": int(np.sum(fails_size & ~fails_contrast)),
        "n_dropped_both": int(np.sum(fails_contrast & fails_size)),
        "thr": float(thr), "why": why,
    }
    return masks, info
 
 
# --- Axon-network descriptors on the soma-subtracted image -------------------
# Somas are already found above, so erase them and describe whatever signal is
# left in the field: that residual IS the neurite/axon network. Two independent
# descriptors, either or both enabled from the CLI:
#
#   1. NETWORK_METRICS     16-level intensity distribution + 4 occupancy
#                          thresholds, per tile, summarised as mean and SD over
#                          tiles. Cheap, CPU, interpretable.
#   2. NETWORK_EMBEDDINGS  EfficientNet embedding of soma-filled crops, mean
#                          pooled over tiles. Reuses the feature_model already
#                          loaded in the consumer, so it costs no extra VRAM.
#
# The two use DIFFERENT tile geometry on purpose, and it is not interchangeable:
#   * metrics want many small tiles (125 px x 256) -- the descriptor is a pixel
#     distribution, and small tiles resolve local heterogeneity.
#   * embeddings want few large crops (1000 px x 8). The image processor squashes
#     any crop to 384x384, so crop size sets the downsample factor. 384 px crops
#     keep native detail but see only 3.7% of the field and lose network-scale
#     structure; 2000 px (whole image) is a 5.2x downsample that erases 2-5 px
#     neurites. 1000 px is the measured compromise.
#
# Benchmarked on the SAG-CL-033/034 NLB plates, 2026-09-08. Caveat carried
# forward: that benchmark had one well per compound, so it establishes relative
# cost and behaviour, NOT a validated compound-detection claim.

NETWORK_LOG2R_EDGES = np.arange(-0.5, 3.0 + 1e-9, 0.25)      # 15 edges -> 16 bins
NETWORK_N_BINS = len(NETWORK_LOG2R_EDGES) + 1
NETWORK_OCC_MULT = ((2 ** 0.5, "1_4"), (2.0, "2"), (4.0, "4"), (8.0, "8"))
NETWORK_FEATURE_NAMES = ([f"hist{k:02d}" for k in range(NETWORK_N_BINS)]
                         + [f"occ_{lab}xB" for _, lab in NETWORK_OCC_MULT])


def network_estimate_background(px, n_bins=512, hi_pct=60.0):
    """
    Background level B for one channel: the mode of the low-intensity part of the
    non-soma pixel histogram (the diffuse background peak), not the mean, so the
    bright network tail cannot drag it up. Every descriptor below is expressed
    relative to B, which is what keeps values comparable across wells and plates
    with different gain or exposure.
    """
    px = px[np.isfinite(px)]
    if px.size < 100:
        return float(np.median(px)) if px.size else 0.0
    lo, hi = np.percentile(px, [0.1, hi_pct])
    if hi <= lo:
        return float(np.median(px))
    h, e = np.histogram(px, bins=n_bins, range=(lo, hi))
    i = int(np.argmax(h))
    return float(0.5 * (e[i] + e[i + 1]))


def network_soma_mask(masks, dilate_px=10):
    """Cell-body mask, dilated to swallow the PSF halo around each soma."""
    import scipy.ndimage as ndi
    m = masks > 0
    if dilate_px > 0:
        m = ndi.binary_dilation(m, ndi.generate_binary_structure(2, 2), iterations=int(dilate_px))
    return m


def _network_tile_origins(shape, tile, n_tiles, seed):
    """Uniformly random tile origins. Seeded per site so a rerun reproduces exactly."""
    h, w = shape
    tile = min(tile, h, w)
    rng = np.random.default_rng(seed)
    return tile, list(zip(rng.integers(0, h - tile + 1, n_tiles),
                          rng.integers(0, w - tile + 1, n_tiles)))


def network_distribution_metrics(plane, soma, tile=125, n_tiles=256, seed=0,
                                 min_valid_frac=0.25):
    """
    16-level distribution + 4 occupancy thresholds for one channel.

    The 16 bins are fixed steps of log2(pixel / B) from 0.71x to 8x background,
    with a catch-all above -- fixed rather than learned per plate, so vectors from
    different plates are directly comparable.

    Soma pixels are EXCLUDED from the statistics rather than counted as
    background: filling them would make each tile's signal fractions depend on how
    many cells happen to sit in it, which is the confound the network descriptor
    exists to avoid.

    Returns (mean_vector, sd_vector, info) over tiles, each of length
    len(NETWORK_FEATURE_NAMES). Tiles that are mostly soma are skipped.
    """
    valid = ~soma
    if valid.sum() < 1000:
        n = len(NETWORK_FEATURE_NAMES)
        return np.full(n, np.nan), np.full(n, np.nan), {"n_tiles_used": 0, "B": np.nan}

    B = network_estimate_background(plane[valid])
    ratio = plane / max(B, 1e-6)
    # bin index map computed once for the whole channel; each tile is then a slice
    idx = np.digitize(np.log2(np.maximum(ratio, 1e-9)), NETWORK_LOG2R_EDGES)
    occ_maps = [(ratio > m) for m, _ in NETWORK_OCC_MULT]

    tile, origins = _network_tile_origins(plane.shape, tile, n_tiles, seed)
    rows = []
    for y, x in origins:
        sy, sx = slice(int(y), int(y) + tile), slice(int(x), int(x) + tile)
        keep = valid[sy, sx]
        n_valid = int(keep.sum())
        if n_valid < min_valid_frac * tile * tile:
            continue
        h = np.bincount(idx[sy, sx][keep], minlength=NETWORK_N_BINS).astype(np.float64)
        h /= n_valid
        occ = [float(o[sy, sx][keep].mean()) for o in occ_maps]
        rows.append(np.concatenate([h, occ]))

    if not rows:
        n = len(NETWORK_FEATURE_NAMES)
        return np.full(n, np.nan), np.full(n, np.nan), {"n_tiles_used": 0, "B": B}
    R = np.vstack(rows)
    sd = R.std(0, ddof=1) if len(R) > 1 else np.zeros(R.shape[1])
    return R.mean(0), sd, {"n_tiles_used": len(R), "B": B}


def network_embed_tiles(plane, soma, processor, feature_model, device,
                        tile=1000, n_tiles=8, seed=0, batch_size=8):
    """
    EfficientNet embedding of the soma-subtracted channel.

    Here the somas ARE filled with the background level rather than excluded --
    a CNN has to be handed a continuous image, and holes would themselves become
    a feature. Each crop is scaled to 8-bit independently, matching the per-crop
    convention this pipeline already uses for single-cell crops.

    Returns (mean_embedding, per_tile_embeddings). Mean pooling was compared
    against median, L2-normalised mean, and mean-concat-SD on the benchmark
    plates: all four scored identically, so the cheapest is used. The per-tile
    array is returned so that choice can be revisited without re-running the GPU
    pass.
    """
    from PIL import Image

    valid = ~soma
    B = network_estimate_background(plane[valid]) if valid.sum() else float(np.median(plane))
    filled = np.where(soma, B, plane)

    tile, origins = _network_tile_origins(plane.shape, tile, n_tiles, seed)
    crops = []
    for y, x in origins:
        c = filled[int(y):int(y) + tile, int(x):int(x) + tile]
        crops.append(Image.fromarray(scale_to_8bit(c)).convert("RGB"))

    embs = []
    for i in range(0, len(crops), batch_size):
        inputs = processor(images=crops[i:i + batch_size], return_tensors="pt").to(device)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            out = feature_model(**inputs)
        embs.append(out.pooler_output.cpu().to(torch.float32).numpy())
    E = np.vstack(embs)
    return E.mean(0), E


def compute_network_features(image_4ch, masks, cfg, processor=None,
                             feature_model=None, device=None, seed=0):
    """
    Runs whichever network descriptors are enabled over the requested channels.

    cfg keys: channel_indices, channel_names, dilate, metrics (bool),
              embeddings (bool), metric_tile, metric_n_tiles, embed_tile,
              embed_n_tiles, save_tiles (bool)
    Returns a dict ready to be dropped into an npz.
    """
    soma = network_soma_mask(masks, cfg["dilate"])
    out = {"soma_area_frac": float(soma.mean()), "n_somas": int(masks.max())}

    if cfg["metrics"]:
        means, sds, backgrounds, used = [], [], [], []
        for ci in cfg["channel_indices"]:
            m, s, info = network_distribution_metrics(
                image_4ch[..., ci].astype(np.float32), soma,
                tile=cfg["metric_tile"], n_tiles=cfg["metric_n_tiles"], seed=seed)
            means.append(m); sds.append(s)
            backgrounds.append(info["B"]); used.append(info["n_tiles_used"])
        out["metrics_mean"] = np.vstack(means).astype(np.float32)   # (n_ch, 20)
        out["metrics_sd"] = np.vstack(sds).astype(np.float32)
        out["metrics_background"] = np.array(backgrounds, np.float32)
        out["metrics_n_tiles"] = np.array(used, np.int32)

    if cfg["embeddings"]:
        pooled, per_tile = [], []
        for ci in cfg["channel_indices"]:
            mu, E = network_embed_tiles(
                image_4ch[..., ci].astype(np.float32), soma,
                processor, feature_model, device,
                tile=cfg["embed_tile"], n_tiles=cfg["embed_n_tiles"], seed=seed)
            pooled.append(mu); per_tile.append(E)
        out["embeddings"] = np.vstack(pooled).astype(np.float32)     # (n_ch, 1280)
        if cfg.get("save_tiles"):
            out["embeddings_per_tile"] = np.stack(per_tile).astype(np.float32)
    return out


# --- 2. Producer-Consumer Worker Functions ---
 
def producer_worker(task_queue, data_queue, worker_id, channels, csv_image_key):
    """
    Producer Process: Handles CPU-bound I/O tasks ONLY. Namely, loading and applying illumination correction arrays.
    Preparing the data for the consumer queue.
    """
    logging.info(f"Producer-{worker_id} started.")
 
    if csv_image_key:
        try:
            # Loads and applies Illumination correction arrays.
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
                    # Loads in memory the channels for the specific site-task to be fed into the consumer
                    all_channels = [tifffile.imread(path)/channel_correction[n] for n,path in enumerate(site_image_paths)]
                else:
                    # Loads without correction
                    all_channels = [tifffile.imread(path) for path in site_image_paths]
 
                image_4ch = np.stack(all_channels, axis=-1)
                data_queue.put((site_id, image_4ch))
                success = True  # Flag success to exit the retry loop
 
            except PermissionError as e:
                # Specifically catches [Errno 13] Permission denied
                logging.warning(f"Producer-{worker_id} PermissionError on site {site_id}. Restarting autofs... (Attempt {retries + 1}/{max_retries})")
 
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
            logging.error(f"Producer-{worker_id} permanently failed on site {site_id} after {max_retries} autofs restarts.")
            data_queue.put((site_id, None))
 
 
def consumer_worker(data_queue, results_dict, stop_event, worker_id, expected_n_channels,
                     temp_dir, dna_index, gpu_id=0, xgb_model_path=None, min_size_filter=None,
                     network_cfg=None):
    """
    Producer Process: Handles GPU taks ONLY. Namely segmentation with cellpose (DNA-contrast
    filtered), per channel embedding extraction from the model, dead cell assessment and post
    processing of the results.
    """
    # It is important to load and set certain libraries and ENV variable inside as this will be deployed in partitioned GPU. Each model is about 10 GB of vRAM.
    import os
    import gc
 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    internal_device_id = 0
 
    from skimage.measure import regionprops
    from cellpose import models
    from transformers import AutoImageProcessor, AutoModel
 
    logging.info(f"Consumer-{worker_id} started. GPU: {gpu_id} | XGB: {bool(xgb_model_path)}")
    device = torch.device(f"cuda:{internal_device_id}" if torch.cuda.is_available() else "cpu")
 
    # --- Load Deep Learning Model ---
    cell_model = models.CellposeModel(gpu=(device.type == 'cuda'), model_type=CELLPOSE_MODEL, device=device)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    feature_model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
 
    # --- Load XGBoost Model of dead cell classifier ---
    bst = None
    if xgb_model_path:
        import xgboost as xgb
        logging.info(f"Consumer-{worker_id}: Loading XGBoost model...")
        bst = xgb.Booster()
        bst.load_model(xgb_model_path)
 
    half_box = BOX_SIZE // 2
    current_batch_size = INFERENCE_BATCH_SIZE
 
    def return_empty_result(s_id, network_path=None):
        # Have 0s in case of errors, meaning no cells or corrupeted images. This should never hapen if proper QC was followed.
        results_dict[s_id] = {'status': 'empty', 'n_cells': 0, 'network_path': network_path}
 
    while not stop_event.is_set():
        try:
            # Get Site/FOV set of n channels image from the data queue.
            item = data_queue.get(timeout=1)
            site_id, image_4ch = item
 
            # Check that the stacked images is of N expected channels
            if image_4ch is None or image_4ch.shape[-1] != expected_n_channels:
                return_empty_result(site_id)
                continue
 
            n_channels = expected_n_channels
 
            # --- 1. Run Cellpose with DNA-contrast filtering ---
            # Segments everything first (Cellpose's own rejection criteria disabled),
            # then removes hallucinated objects using per-object DNA signal contrast
            # against local background -- independent of cellprob/flow_threshold tuning.
            try:
                masks, seg_info = segment_somas_dna_contrast(
                    cell_model, image_4ch, dna_index, diameter=100, min_size_filter=min_size_filter
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                masks, seg_info = segment_somas_dna_contrast(
                    cell_model, image_4ch, dna_index, diameter=100, min_size_filter=min_size_filter
                )
 
            logging.info(f"Consumer-{worker_id}: SITE {site_id} segmentation -- "
                         f"{seg_info['n_in']} raw, {seg_info['n_kept']} kept, "
                         f"{seg_info.get('n_dropped_contrast', 0)} dropped(contrast), "
                         f"{seg_info.get('n_dropped_size', 0)} dropped(size), "
                         f"{seg_info.get('n_dropped_both', 0)} dropped(both) "
                         f"({seg_info.get('why', 'n/a')})")
 
            # --- 1b. Axon-network descriptors on the soma-subtracted image ---
            # Computed here, before any early return, because the network is
            # measurable even in a site where no soma survived the filter.
            network_path = None
            if network_cfg is not None:
                try:
                    net = compute_network_features(
                        image_4ch, masks, network_cfg, processor=processor,
                        feature_model=feature_model, device=device,
                        seed=abs(hash(str(site_id))) % (2 ** 31))
                    network_path = os.path.join(temp_dir, f"network_{site_id}.npz")
                    np.savez_compressed(network_path, **net)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); gc.collect()
                    logging.warning(f"Consumer-{worker_id}: SITE {site_id} network descriptors OOM -- skipped.")
                except Exception as e:
                    logging.error(f"Consumer-{worker_id}: SITE {site_id} network descriptors failed: {e}")

            props = regionprops(masks)
            if not props:
                return_empty_result(site_id, network_path)
                continue
 
            # --- 2. Crop Cells ---
            all_cell_crops = []
            cell_coords = []
            h, w, _ = image_4ch.shape
            # for all cell masks create an equal sized box.
            for prop in props:
                y_center, x_center = map(int, prop.centroid)
                # Remove cell in the edges of images
                if (y_center - half_box < 0) or (y_center + half_box > h) or (x_center - half_box < 0) or (x_center + half_box > w):
                    continue
 
                target_id = prop.label
                y1, y2, x1, x2 = y_center - half_box, y_center + half_box, x_center - half_box, x_center + half_box
                mask_crop = masks[y1:y2, x1:x2]
                binary_mask = (mask_crop == target_id)[:, :, np.newaxis]
                all_cell_crops.append(image_4ch[y1:y2, x1:x2, :] * binary_mask)
                cell_coords.append((y_center, x_center))  # keep coordinates for better retrival at later timepoints
 
            if not all_cell_crops:
                return_empty_result(site_id, network_path)
                continue
 
            # --- 3. Extract Features ---
            batch_pil_images = []
            for cell_crop in all_cell_crops:
                # Iterate of each channel, stack it cell/channel over itself to reach a 3D image. Scale to the model bit trianed data.
                for ch in range(n_channels):
                    scaled_8bit = scale_to_8bit(cell_crop[:, :, ch])
                    batch_pil_images.append(Image.fromarray(scaled_8bit).convert("RGB"))
 
            site_features, idx = [], 0
            total_images = len(batch_pil_images)
            # Images=Cells
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
                    # Channels have to be in the same order as the model was trained. Mostly in line with the order of the channels for segmention.
                    flat_features = reshaped_features.reshape(n_cells, -1)
                    dtrain = xgb.DMatrix(flat_features)
                    preds = bst.predict(dtrain)
                    is_dead = (preds > 0.5)  # Boolean array of dead cells
 
                # Save to disk instead of pushing to Manager dict
                temp_path = os.path.join(temp_dir, f"site_{site_id}.npz")
                np.savez_compressed(temp_path, features=reshaped_features, coords=np.array(cell_coords), is_dead=is_dead)
 
                # Return all features, count, coords, flags, and segmentation QC info
                results_dict[site_id] = {
                    'status': 'success', 'filepath': temp_path, 'n_cells': n_cells,
                    'network_path': network_path,
                    'seg_n_raw': seg_info['n_in'],
                    'seg_n_dropped_contrast': seg_info.get('n_dropped_contrast', 0),
                    'seg_n_dropped_size': seg_info.get('n_dropped_size', 0),
                    'seg_n_dropped_both': seg_info.get('n_dropped_both', 0),
                    'seg_threshold': seg_info['thr'], 'seg_why': seg_info.get('why', ''),
                }
                logging.info(f"Consumer-{worker_id}: Finished SITE {site_id} ({n_cells} total cells, {np.sum(is_dead)} dead).")
            else:
                return_empty_result(site_id, network_path)
 
        # Handldde error related exceptions to skip a site instead of killing the pipeline.
        except Empty: continue
        except Exception as e:
            logging.error(f"Consumer-{worker_id} failed: {e}")
            if 'site_id' in locals():
                return_empty_result(site_id, locals().get('network_path'))
 
 
# --- 3. Main Execution Block ---
def main(args):
    """
    Main function to orchestrate the Producer-Consumer pipeline. Get inputs and get format outputs.
    """
    # Wakeup for the image folder
    try:
        os.listdir(args.data_base_path)
    except Exception:
        pass
    logging.info(f"Starting analysis with parameters: {args}")
 
    # --- Resolve DNA channel index for the DNA-contrast segmentation filter ---
    dna_index = args.dna_channel_index if args.dna_channel_index is not None else resolve_dna_channel_index(args.channels)
    logging.info(f"Using DNA channel index: {dna_index} ('{args.channels[dna_index]}')")
 
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
    logging.info(f"Prepared {num_tasks} sites for processing. Out of ")
    temp_dir = tempfile.mkdtemp(prefix="cellpose_results_")
    logging.info(f"Created temporary caching directory: {temp_dir}")  # Add the original dim[1] of the load data set
 
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
 
        # --- Axon-network descriptor config (None disables both modules) ---
        network_cfg = None
        if args.network_metrics or args.network_embeddings:
            if args.network_channels:
                missing = [c for c in args.network_channels if c not in args.channels]
                if missing:
                    raise ValueError(f"--network-channels {missing} not in --channels {args.channels}")
                net_names = list(args.network_channels)
            else:
                # DNA drives segmentation; it is not a network readout
                net_names = [c for i, c in enumerate(args.channels) if i != dna_index]
            network_cfg = {
                "channel_indices": [args.channels.index(c) for c in net_names],
                "channel_names": net_names,
                "dilate": args.network_dilate,
                "metrics": args.network_metrics,
                "embeddings": args.network_embeddings,
                "metric_tile": args.network_metric_tile,
                "metric_n_tiles": args.network_metric_tiles,
                "embed_tile": args.network_embed_tile,
                "embed_n_tiles": args.network_embed_tiles,
                "save_tiles": args.network_save_tiles,
            }
            logging.info(f"Network descriptors ON for channels {net_names} "
                         f"(metrics={args.network_metrics} @ {args.network_metric_tile}px x{args.network_metric_tiles}, "
                         f"embeddings={args.network_embeddings} @ {args.network_embed_tile}px x{args.network_embed_tiles})")

        # --- Start Consumers ---
        expected_n_channels = len(args.channels)
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            logging.warning("No GPUs detected. Defaulting to GPU logic on CPU (index 0).")
            available_gpus = 1
 
        consumers = [Process(
            target=consumer_worker,
            args=(data_queue, results_dict, stop_event, i, expected_n_channels,
                  temp_dir, dna_index, i % available_gpus, args.xgb_model_path, args.min_size_filter,
                  network_cfg)
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
 
        # --- PROCESS RESULTS (SINGLE-SITE STREAMING) ---
        logging.info("Starting single-pass result processing...")
        original_indices = [task[0] for task in tasks]
 
        # 1. Setup Data Structures for Aggregations & Coordinates
        aggregated_features = []
        final_site_counts = []
        final_dead_counts = []
        coords_records = []
        seg_qc_records = []  # per-site DNA-contrast segmentation QC info
 
        # 2. Setup Single-Cell Parquet Writer
        sc_writer = None
        local_sc_path = None
        chunk_dfs = []
        sites_in_chunk = 0
        CHUNK_SIZE = 500  # Write to disk every 500 sites
 
        if args.single_cell:
            sc_out_path = args.out_data_path.replace('.parquet', '_single_cell.parquet')
            temp_fd, local_sc_path = tempfile.mkstemp(suffix=".parquet")
            os.close(temp_fd)
            logging.info("Single-cell writer initialized. Streaming directly to disk...")
 
        # 3. Main Streaming Loop
        for idx in tqdm(original_indices, desc="Processing Output"):
            res = results_dict[idx]
            well = load_data.loc[idx, 'Metadata_Well']
            site = load_data.loc[idx, 'Metadata_Site'] if 'Metadata_Site' in load_data.columns else str(idx)
 
            if res['status'] == 'empty':
                aggregated_features.append(np.zeros((expected_n_channels, FEATURE_LENGTH), dtype=np.float32))
                final_site_counts.append(0)
                if args.xgb_model_path:
                    final_dead_counts.append(0)
                continue
 
            # --- A. Load data from disk ---
            data = np.load(res['filepath'])
            feats = data['features']
            coords = data['coords']
            flags = data['is_dead']
            n_cells = len(feats)
 
            # --- A.1 Segmentation QC record ---
            seg_qc_records.append({
                'Metadata_Well': well, 'Metadata_Site': site,
                'Seg_N_Raw': res.get('seg_n_raw'),
                'Seg_N_Dropped_Contrast': res.get('seg_n_dropped_contrast'),
                'Seg_N_Dropped_Size': res.get('seg_n_dropped_size'),
                'Seg_N_Dropped_Both': res.get('seg_n_dropped_both'),
                'Seg_Threshold': res.get('seg_threshold'), 'Seg_Why': res.get('seg_why'),
            })
 
            # --- B. Coordinates ---
            if args.save_coords:
                for cell_idx, (y, x) in enumerate(coords):
                    is_dead = flags[cell_idx] if len(flags) > 0 else False
                    coords_records.append({
                        'Cell_ID': f"{well}_{site}_cell{cell_idx}",
                        'Y_Center': y,
                        'X_Center': x,
                        'Is_Dead': is_dead
                    })
 
            # --- C. Well-Level Aggregation ---
            if args.xgb_model_path and args.filter_dead_cells:
                alive_mask = ~flags
                alive_count = np.sum(alive_mask)
                if alive_count > 0:
                    aggregated_features.append(np.sum(feats[alive_mask], axis=0))
                else:
                    aggregated_features.append(np.zeros((expected_n_channels, FEATURE_LENGTH), dtype=np.float32))
                final_site_counts.append(alive_count)
                final_dead_counts.append(flags.sum())
            else:
                aggregated_features.append(np.sum(feats, axis=0))
                final_site_counts.append(n_cells)
                if args.xgb_model_path:
                    final_dead_counts.append(flags.sum())
 
            # --- D. Single-Cell Chunking ---
            if args.single_cell and n_cells > 0:
                site_meta = load_data.loc[[idx]].copy()
                site_df = site_meta.loc[site_meta.index.repeat(n_cells)].copy()
                site_df['Cell_Index'] = np.arange(n_cells)
 
                # Flatten features: (n_cells, channels, feature_len) -> (n_cells, channels * feature_len)
                site_df['single_cell_features'] = list(feats.reshape(n_cells, -1))
 
                if args.xgb_model_path:
                    site_df['is_dead_cell'] = flags
                if 'Cell_Count' in site_df.columns:
                    site_df = site_df.drop(columns=['Cell_Count'])
 
                chunk_dfs.append(site_df)
                sites_in_chunk += 1
 
                # If chunk is full, write it to the Parquet file and clear RAM
                if sites_in_chunk >= CHUNK_SIZE:
                    combined_chunk = pd.concat(chunk_dfs, ignore_index=True)
                    table = pa.Table.from_pandas(combined_chunk)
 
                    if sc_writer is None:
                        sc_writer = pq.ParquetWriter(local_sc_path, table.schema)
                    sc_writer.write_table(table)
 
                    chunk_dfs = []
                    sites_in_chunk = 0
                    del combined_chunk, table
                    import gc
                    gc.collect()
 
            # --- E. FREE UP RAM  ---
            # Guarantees we never accumulate massive arrays in memory
            del feats, coords, flags, data
 
        # Flush any remaining single-cell chunks after the loop finishes
        if args.single_cell and chunk_dfs:
            combined_chunk = pd.concat(chunk_dfs, ignore_index=True)
            table = pa.Table.from_pandas(combined_chunk)
            if sc_writer is None:
                sc_writer = pq.ParquetWriter(local_sc_path, table.schema)
            sc_writer.write_table(table)
 
        if args.single_cell and sc_writer:
            sc_writer.close()
 
        # Clean up temp directory of NPZ files
        shutil.rmtree(temp_dir, ignore_errors=True)
 
        # --- 4. FINALIZE & SAVE OUTPUTS ---
 
        # Save Counts
        load_data['Cell_Count'] = final_site_counts
        if args.xgb_model_path:
            load_data['Dead_Cells'] = final_dead_counts
        load_data.to_csv(args.out_data_path.replace('.parquet', '_counts.csv'), index=False)
 
        # Save Segmentation QC (DNA-contrast filter diagnostics per site)
        if seg_qc_records:
            pd.DataFrame(seg_qc_records).to_csv(
                args.out_data_path.replace('.parquet', '_segmentation_qc.csv'), index=False
            )
            logging.info("Saved per-site segmentation QC (DNA-contrast filter diagnostics).")
 
        # Save Axon-network descriptors (site level and well level)
        if network_cfg is not None:
            net_names = network_cfg["channel_names"]
            metric_rows, embed_rows = [], []
            for idx in original_indices:
                npz_path = results_dict[idx].get('network_path')
                if not npz_path or not os.path.exists(npz_path):
                    continue
                d = np.load(npz_path)
                meta = {'Metadata_Well': load_data.loc[idx, 'Metadata_Well'],
                        'Metadata_Site': load_data.loc[idx, 'Metadata_Site']
                        if 'Metadata_Site' in load_data.columns else str(idx),
                        'Network_Soma_Area_Frac': float(d['soma_area_frac']),
                        'Network_N_Somas': int(d['n_somas'])}
                if 'metrics_mean' in d:
                    row = dict(meta)
                    for ci, ch in enumerate(net_names):
                        for fi, fname in enumerate(NETWORK_FEATURE_NAMES):
                            row[f'Net_{ch}_{fname}_mean'] = float(d['metrics_mean'][ci, fi])
                            row[f'Net_{ch}_{fname}_sd'] = float(d['metrics_sd'][ci, fi])
                        row[f'Net_{ch}_background'] = float(d['metrics_background'][ci])
                        row[f'Net_{ch}_n_tiles'] = int(d['metrics_n_tiles'][ci])
                    metric_rows.append(row)
                if 'embeddings' in d:
                    embed_rows.append(dict(meta, network_embedding=d['embeddings'].tolist()))

            if metric_rows:
                mdf = pd.DataFrame(metric_rows)
                mdf.to_parquet(args.out_data_path.replace('.parquet', '_network_metrics.parquet'),
                               engine='pyarrow')
                num = [c for c in mdf.columns if c.startswith('Net_') or c.startswith('Network_')]
                mdf.groupby('Metadata_Well')[num].mean().reset_index().to_parquet(
                    args.out_data_path.replace('.parquet', '_network_metrics_well.parquet'),
                    engine='pyarrow')
                logging.info(f"Saved network metrics for {len(mdf)} sites "
                             f"({len(num)} columns) and their well-level means.")
            if embed_rows:
                edf = pd.DataFrame(embed_rows)
                edf.to_parquet(args.out_data_path.replace('.parquet', '_network_embeddings.parquet'),
                               engine='pyarrow')
                # well level: mean over sites, per channel, elementwise
                well_emb = (edf.groupby('Metadata_Well')['network_embedding']
                              .apply(lambda s: np.mean(np.stack([np.asarray(v) for v in s]), axis=0).tolist())
                              .reset_index())
                well_emb.to_parquet(
                    args.out_data_path.replace('.parquet', '_network_embeddings_well.parquet'),
                    engine='pyarrow')
                logging.info(f"Saved network embeddings for {len(edf)} sites and "
                             f"{len(well_emb)} well-level means.")

        # Save Coordinates
        if args.save_coords and coords_records:
            pd.DataFrame(coords_records).to_parquet(args.out_data_path.replace('.parquet', '_coords.parquet'), engine='pyarrow')
 
        # Save Well Aggregation
        logging.info("Formatting well-level aggregations...")
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
 
        if args.filter_dead_cells:
            agg_out_path = args.out_data_path.replace('.parquet', '_filtered_well_aggregated.parquet')
        else:
            agg_out_path = args.out_data_path.replace('.parquet', '_well_aggregated.parquet')
        well_level_data.to_parquet(agg_out_path, engine='pyarrow')
        logging.info(f"Saved well-aggregated results to {agg_out_path}")
 
        # Transfer Single-Cell File to S3
        if args.single_cell and local_sc_path and os.path.exists(local_sc_path):
            logging.info(f"Transferring SINGLE CELL results to S3: {sc_out_path}...")
            with open(local_sc_path, 'rb') as f_in:
                with fsspec.open(sc_out_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(local_sc_path)
            logging.info("SINGLE CELL transfer complete.")
 
        logging.info("Script finished successfully.")
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run cell image analysis pipeline. Takes into account image level QC, XGboost model assesment for dead cells. Singel cell and well level embedding extraction using Efficientnet.")
    parser.add_argument('--bucket-input', type=str, required=True,help='Base input bucket where the intermediary results lie')
    parser.add_argument('--data-base-path', type=str, required=True,help='Base path to were the images are stored. ie. /home/storage/Images')
    parser.add_argument('--num-consumers', type=int, default=2,help='Number of models to be loaded in GPU-vRAM for the embedding extraction. ~ 10 GiB of vRAM per consumer.')
    parser.add_argument('--max-workers', type=int, default=24,help='Number of workers to prepare the data. 5 per consumer is more than enough. Higher ration risks OOM issues.')
    parser.add_argument('--load-data-key', type=str, required=True,help='S3 path to the load data file')
    parser.add_argument('--csv-image-key', type=str, required=False,help='S3 path to the Image data file with QC annotations')
    parser.add_argument('--channels', nargs='+', type=str, required=True,help='Channel prefixes as they apper in the load data files. Order is paramount as the first 3 are used for segmentation')
    parser.add_argument('--dna-channel-index', type=int, default=None,
                         help='Explicit index (into --channels) of the DNA/nuclear channel, used for '
                              'the DNA-contrast segmentation filter. If omitted, auto-detected by name '
                              'matching (dna/hoechst/dapi/nuclei/nucleus).')
    parser.add_argument('--min-size-filter', type=int, default=None,
                         help='Minimum object area (px), applied ALONGSIDE the DNA contrast '
                              'filter. Objects below this area are dropped even if they pass '
                              'the contrast check -- useful for removing shrunken/pyknotic '
                              'dead-cell nuclei (e.g. in positive control wells), which can '
                              'still show real DNA contrast but are morphologically abnormal. '
                              'If omitted, only the contrast criterion is applied.')
    # --- Axon-network descriptors (soma-subtracted image) ---
    parser.add_argument('--network-metrics', action='store_true',
                         help='Compute the 16-level intensity distribution + 4 occupancy '
                              'thresholds per tile on the soma-subtracted image, summarised '
                              'as mean and SD over tiles. CPU only, ~1-3 s per image.')
    parser.add_argument('--network-embeddings', action='store_true',
                         help='Compute EfficientNet embeddings of soma-filled crops, mean '
                              'pooled over tiles. Reuses the model already loaded for '
                              'single-cell features, so no extra VRAM. ~0.3 s per image.')
    parser.add_argument('--network-channels', nargs='+', type=str, default=None,
                         help='Channel names (subset of --channels) to describe the network '
                              'on. Defaults to every channel EXCEPT the DNA one, which is a '
                              'segmentation input rather than a network readout.')
    parser.add_argument('--network-metric-tile', type=int, default=125,
                         help='Tile size (px) for --network-metrics. Default 125.')
    parser.add_argument('--network-metric-tiles', type=int, default=256,
                         help='Number of random tiles for --network-metrics. Default 256 '
                              '(~0.9x image area at 125 px, where the sampling error stops '
                              'dominating).')
    parser.add_argument('--network-embed-tile', type=int, default=1000,
                         help='Crop size (px) for --network-embeddings. Default 1000. NOTE: '
                              'the image processor squashes any crop to 384x384, so this sets '
                              'the downsample factor. Smaller crops keep detail but see less '
                              'of the field; larger crops erase fine neurites.')
    parser.add_argument('--network-embed-tiles', type=int, default=8,
                         help='Number of random crops for --network-embeddings. Default 8, '
                              'where the spread over random draws closes.')
    parser.add_argument('--network-dilate', type=int, default=10,
                         help='Pixels to dilate the soma mask before erasing it, to swallow '
                              'the PSF halo around each cell body. Default 10.')
    parser.add_argument('--network-save-tiles', action='store_true',
                         help='Also keep the per-crop embeddings (not just the pooled mean), '
                              'so the pooling choice can be revisited without re-running GPU.')
    parser.add_argument('--out-data-path', type=str, required=True,help='S3 path to the folder where the outputs are desired.')
    parser.add_argument('--single-cell', action='store_true',help='Activates single cell output')
    parser.add_argument('--save-coords', action='store_true',help='Allows for the storage of the cell coordinates.')
    parser.add_argument('--xgb-model-path', type=str, default=None, help='Path to XGBoost json model to classify cells.')
    parser.add_argument('--filter-dead-cells', action='store_true', help='When provided dead cells will be excluded from the aggregation.')
 
    args = parser.parse_args()
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main(args)