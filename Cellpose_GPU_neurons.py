"""
Cellpose_GPU_neurons.py
Runs on P620. Segments somas (multiple configs, swept across cellprob_threshold
values) + network (tile-based) per FOV.
"""
import os
import re
import argparse
import numpy as np
import tifffile
import torch
import pandas as pd
from cellpose import models
from skimage.color import label2rgb
from scipy.ndimage import median_filter, binary_dilation
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from collections import defaultdict
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects

# =========================================================
# Network segmentation helpers
# =========================================================
def flatten_background(img, kernel_size=51):
    background = median_filter(img, size=kernel_size)
    flattened = img.astype(np.float32) - background
    flattened[flattened < 0] = 0
    return flattened

def mask_out_somas(neurite_channel_img, soma_mask, dilation_px=3):
    soma_binary = soma_mask > 0
    if dilation_px > 0:
        soma_binary = binary_dilation(soma_binary, iterations=dilation_px)
    masked_img = neurite_channel_img.copy()
    masked_img[soma_binary] = 0
    return masked_img, soma_binary

def tile_image(img, tile_size):
    h, w = img.shape
    tiles, coords = [], []
    for r in range(0, h - tile_size + 1, tile_size):
        for c in range(0, w - tile_size + 1, tile_size):
            tiles.append(img[r:r+tile_size, c:c+tile_size])
            coords.append((r, c))
    return np.array(tiles), coords

def flag_soma_tiles(tiles, zero_fraction_threshold=0.5):
    zero_fraction = np.mean(tiles == 0, axis=(1, 2))
    keep_mask = zero_fraction < zero_fraction_threshold
    return keep_mask, zero_fraction

def subtract_tile_background(tile, method="median_nonzero"):
    if method == "mean_all":
        bg = tile.mean()
    else:
        nonzero_vals = tile[tile > 0]
        bg = np.median(nonzero_vals) if nonzero_vals.size > 0 else 0
    subtracted = tile.astype(np.float32) - bg
    subtracted[subtracted < 0] = 0
    return subtracted

def edge_density_feature(tile, median_size=3, edge_threshold=None):
    denoised = median_filter(tile, size=median_size)
    edges = sobel(denoised)
    if edge_threshold is None:
        edge_threshold = np.percentile(edges[edges > 0], 75) if np.any(edges > 0) else 0
    edge_binary = edges > edge_threshold
    return edge_binary.mean()

def extract_tile_features(tile):
    nonzero_vals = tile[tile > 0]
    if nonzero_vals.size < 4:
        return None
    features = {
        "mean_intensity": nonzero_vals.mean(),
        "total_intensity": nonzero_vals.sum(),
        "occupancy_fraction": (tile > 0).mean(),
        "intensity_std": nonzero_vals.std(),
    }
    tile_uint8 = np.clip(tile / (tile.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    glcm = graycomatrix(tile_uint8, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=256, symmetric=True, normed=True)
    features["contrast"] = graycoprops(glcm, "contrast").mean()
    features["homogeneity"] = graycoprops(glcm, "homogeneity").mean()
    features["energy"] = graycoprops(glcm, "energy").mean()
    features["correlation"] = graycoprops(glcm, "correlation").mean()
    features["edge_density"] = edge_density_feature(tile)
    return features

def process_fov_network(neurite_channel_img, soma_mask, tile_size,
                          zero_fraction_threshold, illumination_kernel, dilation_px):
    flattened = flatten_background(neurite_channel_img, kernel_size=illumination_kernel)
    masked_img, _ = mask_out_somas(flattened, soma_mask, dilation_px=dilation_px)
    tiles, coords = tile_image(masked_img, tile_size)
    keep_mask, _ = flag_soma_tiles(tiles, zero_fraction_threshold)

    results = []
    for tile, coord, keep in zip(tiles, coords, keep_mask):
        if not keep:
            continue
        bg_sub_tile = subtract_tile_background(tile)
        features = extract_tile_features(bg_sub_tile)
        if features is None:
            continue
        features["row"], features["col"] = coord
        results.append(features)
    return pd.DataFrame(results)


# =========================================================
# Soma segmentation — cellprob_threshold sweep
# =========================================================
def format_threshold_tag(value):
    """
    Turns a threshold value into a filesystem-safe string for filenames,
    e.g. -1.5 -> 'm1p5', 0 -> '0', 2.0 -> '2'.
    """
    s = f"{value:g}"          # trims trailing zeros, e.g. 2.0 -> '2'
    s = s.replace("-", "m")   # 'm' prefix for negative, since '-' in filenames can be finicky in some tools
    s = s.replace(".", "p")   # 'p' for decimal point
    return s


def run_cellpose_sweep(model, img_stack, channels, config_name, out_dir,
                         cellprob_thresholds, diameter=None, channel_axis=0,
                         flow_threshold=0.4, min_size_filter=None):
    """
    Runs Cellpose once per value in cellprob_thresholds, saving masks with
    the threshold value encoded in the filename. Returns a list of dicts
    (one per threshold) with counts/diameter, for folding into the summary.
    """
    os.makedirs(out_dir, exist_ok=True)
    axis_arg = channel_axis if img_stack.ndim == 3 else None

    results = []
    for cp in cellprob_thresholds:
        masks, flows, styles = model.eval(
            img_stack, diameter=diameter, do_3D=False,
            channel_axis=axis_arg, channels=channels,
            flow_threshold=flow_threshold, cellprob_threshold=cp
        )
        n_raw = len(np.unique(masks)) - 1

        if min_size_filter is not None:
            masks = remove_small_objects(masks, min_size=min_size_filter)
        n_final = len(np.unique(masks)) - 1

        median_diameter = None
        if n_final > 0:
            props = regionprops(masks.astype(int))
            diameters = [2 * np.sqrt(p.area / np.pi) for p in props]
            median_diameter = float(np.median(diameters))

        tag = format_threshold_tag(cp)
        file_prefix = f"{config_name}_cp{tag}"

        tifffile.imwrite(os.path.join(out_dir, f"{file_prefix}_labels.tif"), masks.astype(np.uint16))
        mask_rgb = label2rgb(masks, bg_label=0, bg_color=(0, 0, 0))
        tifffile.imwrite(os.path.join(out_dir, f"{file_prefix}_colorized.tif"),
                          (mask_rgb * 255).astype(np.uint8), photometric='rgb')

        print(f"    {file_prefix}: {n_raw} raw objects, {n_final} after filter, "
              f"median diameter={median_diameter}")

        results.append({
            "config_name": config_name,
            "cellprob_threshold": cp,
            "n_objects_raw": n_raw,
            "n_objects_filtered": n_final,
            "median_diameter_px": median_diameter,
            "output_prefix": file_prefix,
        })

    return results


# =========================================================
# Main pipeline
# =========================================================
def main(input_dir, output_dir, tile_size, zero_fraction_threshold,
         illumination_kernel, dilation_px, neurite_channel, well_ids=None,
         run_network_segmentation=True, flow_threshold=0.4,
         cellprob_thresholds_dna=(0.0,), cellprob_thresholds_2ch=(0.0,),
         min_size_filter=None, diameter=None):

    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r"^W(?P<well>\d+)F(?P<field>\d+)T(?P<time>\d+)Z(?P<z>\d+)C(?P<channel>\d+)\.tif$")

    fov_groups = defaultdict(dict)
    for fname in os.listdir(input_dir):
        if not fname.endswith(".tif") or fname.startswith("._"):
            continue
        m = pattern.match(fname)
        if not m:
            print(f"Skipping unmatched file: {fname}")
            continue
        key = (m.group("well"), m.group("field"), m.group("time"))
        fov_groups[key][int(m.group("channel"))] = os.path.join(input_dir, fname)

    print(f"Found {len(fov_groups)} FOVs total (before well filtering)")

    # --- Filter to only requested well IDs, if provided ---
    if well_ids is not None:
        well_ids_set = set(well_ids)
        fov_groups = {
            key: val for key, val in fov_groups.items()
            if key[0] in well_ids_set  # key[0] is the well
        }
        print(f"Filtered to {len(fov_groups)} FOVs matching well IDs: {sorted(well_ids_set)}")

        found_wells = set(key[0] for key in fov_groups.keys())
        missing_wells = well_ids_set - found_wells
        if missing_wells:
            print(f"  WARNING: no FOVs found for requested wells: {sorted(missing_wells)}")

    worker_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {worker_device}")
    model = models.CellposeModel(gpu=(worker_device.type == 'cuda'), pretrained_model='cyto3')

    summary_rows = []  # long format: one row per FOV x config x cellprob_threshold

    for (well, field, time), channels_dict in fov_groups.items():
        fov_id = f"W{well}F{field}T{time}"
        print(f"\nProcessing FOV: {fov_id}")

        if len(channels_dict) != 7:
            print(f"  WARNING: expected 7 channels, found {len(channels_dict)} — skipping")
            continue

        ordered_channels = [channels_dict[k] for k in sorted(channels_dict.keys())]
        imgs_loaded = [tifffile.imread(p) for p in ordered_channels]
        img_multi_channel = np.stack(imgs_loaded, axis=0).astype(np.float32)
        Nuclei, ER, CL488Y, CL488R, CL561, BF1, BF2 = [img_multi_channel[i] for i in range(7)]

        fov_out_dir = os.path.join(output_dir, fov_id)
        os.makedirs(fov_out_dir, exist_ok=True)

        # --- Soma segmentation: DNA-only, swept across cellprob_thresholds_dna ---
        dna_results = run_cellpose_sweep(
            model, Nuclei, channels=[0, 0], config_name="single_channel_DNA",
            out_dir=fov_out_dir, cellprob_thresholds=cellprob_thresholds_dna,
            diameter=diameter, channel_axis=None, flow_threshold=flow_threshold,
            min_size_filter=None
        )
        for r in dna_results:
            summary_rows.append({"fov_id": fov_id, **r})

        # --- Soma segmentation: 2-channel, swept across cellprob_thresholds_2ch ---
        candidates = {"CL561": CL561}
        for name, ch in candidates.items():
            img_2ch = np.stack([ch, Nuclei], axis=0)
            ch2_results = run_cellpose_sweep(
                model, img_2ch, channels=[1, 2], config_name=f"two_channel_{name}_DNA",
                out_dir=fov_out_dir, cellprob_thresholds=cellprob_thresholds_2ch,
                diameter=diameter, channel_axis=0, flow_threshold=flow_threshold,
                min_size_filter=min_size_filter
            )
            for r in ch2_results:
                summary_rows.append({"fov_id": fov_id, **r})

        # --- Network segmentation (uses first DNA threshold's mask as the soma mask) ---
        # Network step needs ONE soma mask, not a sweep — uses the first cellprob_thresholds_dna value.
        if run_network_segmentation:
            masks_dna_for_network, _, _ = model.eval(
                Nuclei, diameter=diameter, do_3D=False, channel_axis=None,
                channels=[0, 0], flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_thresholds_dna[0]
            )
            neurite_img = img_multi_channel[neurite_channel - 1]
            network_df = process_fov_network(
                neurite_img, masks_dna_for_network, tile_size=tile_size,
                zero_fraction_threshold=zero_fraction_threshold,
                illumination_kernel=illumination_kernel, dilation_px=dilation_px
            )
            network_df.to_csv(os.path.join(fov_out_dir, "network_tile_features.csv"), index=False)
            print(f"    network: {len(network_df)} tiles kept "
                  f"(soma mask from cellprob_threshold={cellprob_thresholds_dna[0]})")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "segmentation_summary.csv"), index=False)
    print("\nSummary:")
    print(summary_df)
    print("\nDONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tile_size", type=int, default=15)
    parser.add_argument("--zero_fraction_threshold", type=float, default=0.5)
    parser.add_argument("--illumination_kernel", type=int, default=51)
    parser.add_argument("--dilation_px", type=int, default=3)
    parser.add_argument("--neurite_channel", type=int, default=2)
    parser.add_argument("--well_ids", type=str, default=None,
                         help="Comma-separated list of well IDs to process, e.g. '0026,0281,0243'. "
                              "If omitted, all wells are processed.")
    parser.add_argument("--run_network_segmentation", action="store_true",
                         help="If set, also run network/neurite tile-based segmentation. "
                              "Omit this flag to run soma segmentation only.")
    parser.add_argument("--flow_threshold", type=float, default=0.4,
                         help="Cellpose flow_threshold, applied to both soma configs.")
    parser.add_argument("--cellprob_thresholds_dna", type=str, default="0.0",
                         help="Comma-separated cellprob_threshold values to sweep for the "
                              "DNA-only config, e.g. '-2,-1,0,1'. Each value produces its own "
                              "output mask, named with the threshold in the filename.")
    parser.add_argument("--cellprob_thresholds_2ch", type=str, default="0.0",
                         help="Comma-separated cellprob_threshold values to sweep for the "
                              "2-channel config, e.g. '0,1,2,3'. Each value produces its own "
                              "output mask, named with the threshold in the filename.")
    parser.add_argument("--min_size_filter", type=int, default=None,
                         help="Minimum object area (px) to keep in the 2-channel config, "
                              "filters spurious tiny detections.")
    parser.add_argument("--diameter", type=float, default=None,
                         help="Fixed diameter (px) for Cellpose. If omitted, Cellpose "
                              "auto-estimates diameter independently for each run.")
    args = parser.parse_args()

    well_ids_list = args.well_ids.split(",") if args.well_ids else None
    cellprob_thresholds_dna = tuple(float(x) for x in args.cellprob_thresholds_dna.split(","))
    cellprob_thresholds_2ch = tuple(float(x) for x in args.cellprob_thresholds_2ch.split(","))

    main(args.input_dir, args.output_dir, args.tile_size, args.zero_fraction_threshold,
         args.illumination_kernel, args.dilation_px, args.neurite_channel,
         well_ids=well_ids_list, run_network_segmentation=args.run_network_segmentation,
         flow_threshold=args.flow_threshold,
         cellprob_thresholds_dna=cellprob_thresholds_dna,
         cellprob_thresholds_2ch=cellprob_thresholds_2ch,
         min_size_filter=args.min_size_filter,
         diameter=args.diameter)