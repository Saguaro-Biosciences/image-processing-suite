"""
Cellpose_GPU_neurons.py
Runs on P620. Segments somas (multiple configs, swept across cellprob_threshold
AND flow_threshold values, with configurable normalization) + network
(tile-based) per FOV.

NOTE: Written for Cellpose v4 (Cellpose-SAM). The `channels` argument to
model.eval() no longer exists in v4 -- Cellpose-SAM is trained to be
channel-order-invariant and uses the first 3 channels of whatever array you
give it. To combine channels (e.g. cytoplasm marker + nucleus), stack them
into a 2D/3D array yourself before calling eval(), rather than passing a
channels=[...] argument. See run_cellpose_sweep() below.
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
def build_three_channel_stack(nuclei, channel_a, ch_b, ch_c):
    combined_bc = np.maximum(ch_b, ch_c)
    return np.stack([channel_a, nuclei, combined_bc], axis=0)


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
# Soma segmentation — cellprob_threshold x flow_threshold sweep (Cellpose v4 API)
# =========================================================
def format_threshold_tag(value):
    """
    Turns a threshold value into a filesystem-safe string for filenames,
    e.g. -1.5 -> 'm1p5', 0 -> '0', 2.0 -> '2'.
    """
    s = f"{value:g}"          # trims trailing zeros, e.g. 2.0 -> '2'
    s = s.replace("-", "m")   # 'm' prefix for negative
    s = s.replace(".", "p")   # 'p' for decimal point
    return s


def run_cellpose_sweep(model, img_stack, config_name, out_dir,
                         cellprob_thresholds, flow_thresholds=(0.4,),
                         diameter=None, channel_axis=None,
                         min_size_filter=None, normalize_percentile=None,
                         save_cellprob_map=False):
    """
    Runs Cellpose once per combination of cellprob_thresholds x flow_thresholds
    (a full grid), saving masks with both threshold values encoded in the
    filename. Returns a list of dicts (one per combination) with counts/
    diameter, for folding into the summary.

    normalize_percentile: optional tuple (low, high), e.g. (2, 98). Passed to
    Cellpose's normalize={"percentile": [low, high]} to control contrast
    stretching before segmentation. If None, Cellpose's default (1, 99) is
    used. Tighter/wider bounds can reduce sensitivity to outlier bright
    pixels that otherwise amplify background noise into spurious detections.

    save_cellprob_map: if True, saves the raw pre-threshold cell-probability
    map (flows[2]) as a float32 tif for the FIRST threshold combination only
    (one map per config per FOV, not per threshold combo, to avoid excessive
    output volume). Useful for diagnosing whether errors are a normalization/
    signal problem vs. a threshold problem, without needing a local notebook.

    v4 (Cellpose-SAM) note: no `channels` argument. img_stack should already
    be the exact array you want segmented -- for single-channel input, pass
    a 2D array with channel_axis=None; for a manually-combined multi-channel
    array (e.g. cyto + nucleus stacked), pass a 3D array with channel_axis
    set to the stacking axis (matches how img_stack was built upstream).
    """
    os.makedirs(out_dir, exist_ok=True)

    normalize_arg = True
    if normalize_percentile is not None:
        lo, hi = normalize_percentile
        normalize_arg = {"percentile": [lo, hi]}

    results = []
    first_combo = True
    for ft in flow_thresholds:
        for cp in cellprob_thresholds:
            masks, flows, styles = model.eval(
                img_stack, diameter=diameter, do_3D=False,
                channel_axis=channel_axis,
                flow_threshold=ft, cellprob_threshold=cp,
                normalize=normalize_arg
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

            cp_tag = format_threshold_tag(cp)
            ft_tag = format_threshold_tag(ft)
            file_prefix = f"{config_name}_cp{cp_tag}_ft{ft_tag}"

            tifffile.imwrite(os.path.join(out_dir, f"{file_prefix}_labels.tif"), masks.astype(np.uint16))
            mask_rgb = label2rgb(masks, bg_label=0, bg_color=(0, 0, 0))
            tifffile.imwrite(os.path.join(out_dir, f"{file_prefix}_colorized.tif"),
                              (mask_rgb * 255).astype(np.uint8), photometric='rgb')

            if save_cellprob_map and first_combo:
                cellprob_map = np.asarray(flows[2]).astype(np.float32)
                tifffile.imwrite(os.path.join(out_dir, f"{config_name}_cellprob_map.tif"), cellprob_map)
                first_combo = False

            print(f"    {file_prefix}: {n_raw} raw objects, {n_final} after filter, "
                  f"median diameter={median_diameter}")

            results.append({
                "config_name": config_name,
                "cellprob_threshold": cp,
                "flow_threshold": ft,
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
         run_network_segmentation=True,
         flow_thresholds=(0.4,),
         cellprob_thresholds_dna=(0.0,), cellprob_thresholds_2ch=(0.0,),
         min_size_filter=None, diameter=None,
         normalize_percentile=None, save_cellprob_map=False,
         max_fovs=None):

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

    # --- Optional hard cap on number of FOVs processed (for quick tests) ---
    if max_fovs is not None:
        fov_groups = dict(list(fov_groups.items())[:max_fovs])
        print(f"Capped to first {len(fov_groups)} FOVs (max_fovs={max_fovs})")

    worker_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {worker_device}")
    # v4: no 'model_type' distinction like 'cyto3' needed for Cellpose-SAM's
    # default weights, but pretrained_model is still accepted if you have a
    # specific checkpoint. Using default (built-in SAM weights) here.
    model = models.CellposeModel(gpu=(worker_device.type == 'cuda'))

    summary_rows = []  # long format: one row per FOV x config x cellprob_threshold x flow_threshold

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

        # --- Soma segmentation: DNA-only, swept across cellprob_thresholds_dna x flow_thresholds ---
        # v4: single-channel 2D array, no channels arg, channel_axis=None
        dna_results = run_cellpose_sweep(
            model, Nuclei, config_name="single_channel_DNA",
            out_dir=fov_out_dir, cellprob_thresholds=cellprob_thresholds_dna,
            flow_thresholds=flow_thresholds,
            diameter=diameter, channel_axis=None,
            min_size_filter=None,
            normalize_percentile=normalize_percentile,
            save_cellprob_map=save_cellprob_map
        )
        for r in dna_results:
            summary_rows.append({"fov_id": fov_id, **r})

        # --- Soma segmentation: 2-channel, swept across cellprob_thresholds_2ch x flow_thresholds ---
        # v4: manually stack cyto + nucleus into one array (order no longer
        # matters per Cellpose-SAM docs), channel_axis=0 matches the stack axis
        candidates = {"ER": ER,
                      "CL488Y": CL488Y}
        for name, ch in candidates.items():
            img_2ch = np.stack([ch, Nuclei], axis=0)
            ch2_results = run_cellpose_sweep(
                model, img_2ch, config_name=f"two_channel_{name}_DNA",
                out_dir=fov_out_dir, cellprob_thresholds=cellprob_thresholds_2ch,
                flow_thresholds=flow_thresholds,
                diameter=diameter, channel_axis=0,
                min_size_filter=min_size_filter,
                normalize_percentile=normalize_percentile,
                save_cellprob_map=save_cellprob_map
            )
            for r in ch2_results:
                summary_rows.append({"fov_id": fov_id, **r})

        # --- Soma segmentation: 3-channel (Nuclei + ER + CL488Y), swept ---
        img_3ch = np.stack([Nuclei, ER, CL488Y], axis=0)
        ch3_results = run_cellpose_sweep(
            model, img_3ch, config_name="three_channel_CL488Y_ER_DNA",
            out_dir=fov_out_dir, cellprob_thresholds=cellprob_thresholds_2ch,
            flow_thresholds=flow_thresholds,
            diameter=diameter, channel_axis=0,
            min_size_filter=min_size_filter,
            normalize_percentile=normalize_percentile,
            save_cellprob_map=save_cellprob_map
        )
        for r in ch3_results:
            summary_rows.append({"fov_id": fov_id, **r})

        # --- Network segmentation (uses first DNA threshold combo's mask as the soma mask) ---
        # Network step needs ONE soma mask, not a sweep — uses the first
        # cellprob_thresholds_dna / flow_thresholds values.
        if run_network_segmentation:
            normalize_arg = True
            if normalize_percentile is not None:
                lo, hi = normalize_percentile
                normalize_arg = {"percentile": [lo, hi]}

            masks_dna_for_network, _, _ = model.eval(
                Nuclei, diameter=diameter, do_3D=False, channel_axis=None,
                flow_threshold=flow_thresholds[0],
                cellprob_threshold=cellprob_thresholds_dna[0],
                normalize=normalize_arg
            )
            neurite_img = img_multi_channel[neurite_channel - 1]
            network_df = process_fov_network(
                neurite_img, masks_dna_for_network, tile_size=tile_size,
                zero_fraction_threshold=zero_fraction_threshold,
                illumination_kernel=illumination_kernel, dilation_px=dilation_px
            )
            network_df.to_csv(os.path.join(fov_out_dir, "network_tile_features.csv"), index=False)
            print(f"    network: {len(network_df)} tiles kept "
                  f"(soma mask from cellprob_threshold={cellprob_thresholds_dna[0]}, "
                  f"flow_threshold={flow_thresholds[0]})")

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
    parser.add_argument("--max_fovs", type=int, default=None,
                         help="Process at most this many FOVs, for quick testing. "
                              "Applied after well_ids filtering.")
    parser.add_argument("--run_network_segmentation", action="store_true",
                         help="If set, also run network/neurite tile-based segmentation. "
                              "Omit this flag to run soma segmentation only.")
    parser.add_argument("--flow_thresholds", type=str, default="0.4",
                         help="Comma-separated flow_threshold values to sweep, e.g. '0.2,0.4,0.6'. "
                              "Combined with cellprob_threshold values as a full grid -- each "
                              "combination produces its own output mask, named with both "
                              "threshold values in the filename.")
    parser.add_argument("--cellprob_thresholds_dna", type=str, default="0.0",
                         help="Comma-separated cellprob_threshold values to sweep for the "
                              "DNA-only config, e.g. '-2,-1,0,1'.")
    parser.add_argument("--cellprob_thresholds_2ch", type=str, default="0.0",
                         help="Comma-separated cellprob_threshold values to sweep for the "
                              "2-channel and 3-channel configs, e.g. '0,1,2,3'.")
    parser.add_argument("--min_size_filter", type=int, default=None,
                         help="Minimum object area (px) to keep in the 2-channel and 3-channel "
                              "configs, filters spurious tiny detections.")
    parser.add_argument("--diameter", type=float, default=None,
                         help="Fixed diameter (px) for Cellpose. If omitted, Cellpose-SAM's "
                              "built-in size invariance is used (diameter is optional in v4).")
    parser.add_argument("--normalize_percentile_low", type=float, default=None,
                         help="Lower percentile bound for Cellpose's contrast normalization, "
                              "e.g. 2. Must be paired with --normalize_percentile_high. If "
                              "omitted, Cellpose's default (1, 99) is used.")
    parser.add_argument("--normalize_percentile_high", type=float, default=None,
                         help="Upper percentile bound for Cellpose's contrast normalization, "
                              "e.g. 98. Must be paired with --normalize_percentile_low.")
    parser.add_argument("--save_cellprob_map", action="store_true",
                         help="If set, saves the raw pre-threshold cell-probability map "
                              "(one per config per FOV, using the first threshold combination) "
                              "as a float32 tif, for diagnosing hallucination/miss patterns.")
    args = parser.parse_args()

    well_ids_list = args.well_ids.split(",") if args.well_ids else None
    flow_thresholds = tuple(float(x) for x in args.flow_thresholds.split(","))
    cellprob_thresholds_dna = tuple(float(x) for x in args.cellprob_thresholds_dna.split(","))
    cellprob_thresholds_2ch = tuple(float(x) for x in args.cellprob_thresholds_2ch.split(","))

    normalize_percentile = None
    if args.normalize_percentile_low is not None and args.normalize_percentile_high is not None:
        normalize_percentile = (args.normalize_percentile_low, args.normalize_percentile_high)
    elif args.normalize_percentile_low is not None or args.normalize_percentile_high is not None:
        raise ValueError("Both --normalize_percentile_low and --normalize_percentile_high "
                          "must be provided together, or neither.")

    main(args.input_dir, args.output_dir, args.tile_size, args.zero_fraction_threshold,
         args.illumination_kernel, args.dilation_px, args.neurite_channel,
         well_ids=well_ids_list, run_network_segmentation=args.run_network_segmentation,
         flow_thresholds=flow_thresholds,
         cellprob_thresholds_dna=cellprob_thresholds_dna,
         cellprob_thresholds_2ch=cellprob_thresholds_2ch,
         min_size_filter=args.min_size_filter,
         diameter=args.diameter,
         normalize_percentile=normalize_percentile,
         save_cellprob_map=args.save_cellprob_map,
         max_fovs=args.max_fovs)