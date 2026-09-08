"""
Cellpose_GPU_neurons.py
Runs on P620. Segments somas (multiple configs, swept across cellprob_threshold
AND flow_threshold values, with configurable normalization) + network
(tile-based) per FOV. Also supports an alternative "segment everything, then
filter by per-object DNA contrast (optionally combined with a minimum size
filter)" strategy as its own config, for cases where no single cellprob/flow
threshold achieves both high recall and low false positives.

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
import matplotlib
matplotlib.use("Agg")  # headless-safe: only ever used for fig.savefig(), never plt.show()
import matplotlib.pyplot as plt
from cellpose import models
from skimage.color import label2rgb
from scipy import ndimage as ndi
from scipy.ndimage import median_filter, binary_dilation
from skimage.filters import threshold_otsu, sobel
from skimage.feature import graycomatrix, graycoprops
from sklearn.mixture import GaussianMixture
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
# Soma segmentation — "segment everything, filter by DNA contrast
# (optionally + minimum size)" strategy
# =========================================================
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

    Returns: (labels array, log2 contrast score array, local-band pixel
    count array) -- all aligned by index.
    """
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


def per_image_threshold(x, min_objects=30, bic_margin=10,
                          clamp=(0.2, 3.0), default=1.0):
    """
    Derives a per-image contrast threshold separating real objects from
    hallucinated ones, using a 2-component Gaussian Mixture Model fit to
    THIS image's contrast score distribution -- with guards against forcing
    a split where none is statistically supported (too few objects, or a
    genuinely unimodal distribution), and against trusting an Otsu threshold
    that lands somewhere implausible.

    Returns (threshold, explanation_string).
    """
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


def run_cellpose_dna_contrast_filter(model, img_stack, dna_channel, config_name, out_dir,
                                       diameter=None, channel_axis=None,
                                       band_px=(2, 12), min_band=20,
                                       bic_margin=10, clamp=(0.2, 3.0), default_threshold=1.0,
                                       min_size_filter=None):
    """
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
         still show real, positive DNA contrast.

    This is an approach independent of Cellpose's own confidence outputs,
    useful when no cellprob/flow_threshold setting achieves both high
    recall and low false positives together.

    Saves: raw (unfiltered) labels, filtered labels, filtered colorized mask,
    and a histogram PNG of the contrast score distribution with the chosen
    threshold marked, for visual QC.
    """
    os.makedirs(out_dir, exist_ok=True)

    masks_raw, flows, styles = model.eval(
        img_stack, diameter=diameter, do_3D=False, channel_axis=channel_axis,
        flow_threshold=0, min_size=-1  # Cellpose's own built-in criteria, disabled
    )
    n_raw = len(np.unique(masks_raw)) - 1
    tifffile.imwrite(os.path.join(out_dir, f"{config_name}_dnacontrast_raw_labels.tif"),
                      masks_raw.astype(np.uint16))

    # Guard: if Cellpose found nothing at all, dna_contrast()/per_image_threshold()
    # would be operating on an empty label set -- short-circuit cleanly instead.
    if masks_raw.max() == 0:
        print(f"    {config_name} (DNA-contrast filter): 0 objects found, nothing to filter")
        tifffile.imwrite(os.path.join(out_dir, f"{config_name}_dnacontrast_filtered_labels.tif"),
                          masks_raw.astype(np.uint16))
        return {
            "config_name": f"{config_name}_dnacontrast",
            "n_objects_raw": 0,
            "n_objects_filtered": 0,
            "n_dropped_contrast": 0,
            "n_dropped_size": 0,
            "n_dropped_both": 0,
            "contrast_threshold_used": None,
            "threshold_explanation": "no objects detected in raw segmentation",
            "n_objects_global_bg_fallback": 0,
        }

    labs, x, band_cnt = dna_contrast(masks_raw, dna_channel.astype(np.float32),
                                       band_px=band_px, min_band=min_band)
    thr, why = per_image_threshold(x, bic_margin=bic_margin, clamp=clamp, default=default_threshold)

    fails_contrast = ~(x >= thr)

    # Minimum size criterion, independent of contrast -- catches morphologically
    # abnormal (e.g. shrunken/pyknotic dead-cell) objects that still pass the
    # contrast check since they can retain real, positive DNA signal.
    fails_size = np.zeros_like(fails_contrast)
    if min_size_filter is not None:
        area_by_label = {p.label: p.area for p in regionprops(masks_raw.astype(int))}
        areas = np.array([area_by_label.get(l, 0) for l in labs])
        fails_size = areas < min_size_filter

    drop_mask = fails_contrast | fails_size
    drop_labs = labs[drop_mask]
    keep_labs = set(labs[~drop_mask].tolist())

    masks_filtered = np.where(np.isin(masks_raw, list(keep_labs)), masks_raw, 0).astype(masks_raw.dtype)
    n_filtered = len(keep_labs)
    n_fallback_bg = int(np.sum(band_cnt < min_band))

    n_dropped_contrast = int(np.sum(fails_contrast & ~fails_size))
    n_dropped_size = int(np.sum(fails_size & ~fails_contrast))
    n_dropped_both = int(np.sum(fails_contrast & fails_size))

    tifffile.imwrite(os.path.join(out_dir, f"{config_name}_dnacontrast_filtered_labels.tif"),
                      masks_filtered.astype(np.uint16))
    mask_rgb = label2rgb(masks_filtered, bg_label=0, bg_color=(0, 0, 0))
    tifffile.imwrite(os.path.join(out_dir, f"{config_name}_dnacontrast_filtered_colorized.tif"),
                      (mask_rgb * 255).astype(np.uint8), photometric='rgb')

    # QC histogram -- color-code dropped-by-size objects separately from
    # dropped-by-contrast, so the two failure modes are visually distinguishable
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if min_size_filter is not None:
        ax.hist(x[~drop_mask], bins=40, color="#2a78d6", edgecolor="#fcfcfb",
                 linewidth=0.5, label="kept", alpha=0.9)
        ax.hist(x[fails_contrast & ~fails_size], bins=40, color="#c0392b", edgecolor="#fcfcfb",
                 linewidth=0.5, label="dropped (contrast)", alpha=0.7)
        ax.hist(x[fails_size & ~fails_contrast], bins=40, color="#e67e22", edgecolor="#fcfcfb",
                 linewidth=0.5, label="dropped (size)", alpha=0.7)
        ax.legend(fontsize=7)
    else:
        ax.hist(x, bins=40, color="#2a78d6", edgecolor="#fcfcfb", linewidth=0.5)
    ax.axvline(thr, color="#898781", lw=1.5)
    ax.annotate(f"threshold {thr:.2f}", xy=(thr, 0.97), xycoords=("data", "axes fraction"),
                xytext=(5, 0), textcoords="offset points", va="top",
                color="#52514e", fontsize=8)
    ax.set_xlabel("log2( object DNA / local background )")
    ax.set_ylabel("objects")
    ax.set_title(f"{config_name} — DNA contrast — n={len(x)}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{config_name}_dnacontrast_histogram.png"), dpi=150)
    plt.close(fig)

    print(f"    {config_name} (DNA-contrast filter): {n_raw} raw objects | {why} | "
          f"{n_fallback_bg} objects used global background fallback | "
          f"dropped: {n_dropped_contrast} contrast-only, {n_dropped_size} size-only, "
          f"{n_dropped_both} both | keeping {n_filtered} of {n_raw}")

    return {
        "config_name": f"{config_name}_dnacontrast",
        "n_objects_raw": n_raw,
        "n_objects_filtered": n_filtered,
        "n_dropped_contrast": n_dropped_contrast,
        "n_dropped_size": n_dropped_size,
        "n_dropped_both": n_dropped_both,
        "contrast_threshold_used": thr,
        "threshold_explanation": why,
        "n_objects_global_bg_fallback": n_fallback_bg,
    }


# =========================================================
# Main pipeline
# =========================================================
def main(input_dir, output_dir, tile_size, zero_fraction_threshold,
         illumination_kernel, dilation_px, neurite_channel, well_ids=None,
         run_network_segmentation=True,
         flow_thresholds=(0.4,),
         cellprob_thresholds_dna=(0.0,), cellprob_thresholds_2ch=(0.0,),
         cellprob_thresholds_3ch=(0.0,),
         min_size_filter=None, diameter=None,
         normalize_percentile=None, save_cellprob_map=False,
         max_fovs=None,
         skip_two_channel=False,
         run_dna_contrast_filter=False,
         dna_contrast_band_px=(2, 12), dna_contrast_min_band=20,
         dna_contrast_bic_margin=10, dna_contrast_clamp=(0.2, 3.0),
         dna_contrast_default=1.0, dna_contrast_min_size_filter=None):

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
        if not skip_two_channel:
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

                # --- Optional: same 2-channel input, alternative "segment everything,
                # filter by DNA contrast (+ optional min size)" strategy ---
                if run_dna_contrast_filter:
                    dc_result = run_cellpose_dna_contrast_filter(
                        model, img_2ch, Nuclei, config_name=f"two_channel_{name}_DNA",
                        out_dir=fov_out_dir, diameter=diameter, channel_axis=0,
                        band_px=dna_contrast_band_px, min_band=dna_contrast_min_band,
                        bic_margin=dna_contrast_bic_margin, clamp=dna_contrast_clamp,
                        default_threshold=dna_contrast_default,
                        min_size_filter=dna_contrast_min_size_filter
                    )
                    summary_rows.append({"fov_id": fov_id, **dc_result})
        else:
            print("    Skipping 2-channel config (--skip_two_channel)")

        # --- Soma segmentation: 3-channel (Nuclei + ER + CL488Y), swept ---
        img_3ch = np.stack([Nuclei, ER, CL488Y], axis=0)
        ch3_results = run_cellpose_sweep(
            model, img_3ch, config_name="three_channel_CL488Y_ER_DNA",
            out_dir=fov_out_dir, cellprob_thresholds=cellprob_thresholds_3ch,
            flow_thresholds=flow_thresholds,
            diameter=diameter, channel_axis=0,
            min_size_filter=min_size_filter,
            normalize_percentile=normalize_percentile,
            save_cellprob_map=save_cellprob_map
        )
        for r in ch3_results:
            summary_rows.append({"fov_id": fov_id, **r})

        if run_dna_contrast_filter:
            dc_result_3ch = run_cellpose_dna_contrast_filter(
                model, img_3ch, Nuclei, config_name="three_channel_CL488Y_ER_DNA",
                out_dir=fov_out_dir, diameter=diameter, channel_axis=0,
                band_px=dna_contrast_band_px, min_band=dna_contrast_min_band,
                bic_margin=dna_contrast_bic_margin, clamp=dna_contrast_clamp,
                default_threshold=dna_contrast_default,
                min_size_filter=dna_contrast_min_size_filter
            )
            summary_rows.append({"fov_id": fov_id, **dc_result_3ch})

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
    parser.add_argument("--skip_two_channel", action="store_true",
                         help="If set, skips the 2-channel (ER+DNA, CL488Y+DNA) configs "
                              "entirely, including their DNA-contrast-filtered variants if "
                              "--run_dna_contrast_filter is also set. Useful for faster test "
                              "runs focused only on DNA-only and/or 3-channel configs.")
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
                              "2-channel configs (ER+DNA, CL488Y+DNA), e.g. '0,1,2,3'.")
    parser.add_argument("--cellprob_thresholds_3ch", type=str, default="0.0",
                         help="Comma-separated cellprob_threshold values to sweep for the "
                              "3-channel config (Nuclei+ER+CL488Y), e.g. '0,1,2,3'. Tuned "
                              "separately from the 2-channel configs since the 3-channel "
                              "signal composition differs.")
    parser.add_argument("--min_size_filter", type=int, default=None,
                         help="Minimum object area (px) to keep in the 2-channel and 3-channel "
                              "threshold-swept configs, filters spurious tiny detections.")
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
    parser.add_argument("--run_dna_contrast_filter", action="store_true",
                         help="If set, additionally runs the 2-channel and 3-channel configs "
                              "through a 'segment everything, then filter by per-object DNA "
                              "contrast' strategy, as an extra config alongside the threshold "
                              "sweeps. Prioritizes recall (Cellpose's own rejection criteria "
                              "disabled) and relies on DNA signal contrast (optionally + a "
                              "minimum size filter) to remove hallucinated and/or "
                              "morphologically abnormal objects afterward. Useful when no "
                              "single cellprob/flow_threshold setting achieves both high "
                              "recall and low false positives.")
    parser.add_argument("--dna_contrast_band_inner_px", type=int, default=2,
                         help="Inner distance (px) from an object's edge defining its local "
                              "background ring, for DNA contrast filtering.")
    parser.add_argument("--dna_contrast_band_outer_px", type=int, default=12,
                         help="Outer distance (px) from an object's edge defining its local "
                              "background ring, for DNA contrast filtering.")
    parser.add_argument("--dna_contrast_min_band", type=int, default=20,
                         help="Minimum local background pixel count required before falling "
                              "back to the image-wide background median, for DNA contrast "
                              "filtering.")
    parser.add_argument("--dna_contrast_bic_margin", type=float, default=10,
                         help="BIC margin required before accepting a 2-component (bimodal) "
                              "split over a 1-component fit, for DNA contrast filtering.")
    parser.add_argument("--dna_contrast_clamp_low", type=float, default=0.2,
                         help="Lower bound for an acceptable Otsu-derived contrast threshold; "
                              "outside this range, falls back to the default threshold.")
    parser.add_argument("--dna_contrast_clamp_high", type=float, default=3.0,
                         help="Upper bound for an acceptable Otsu-derived contrast threshold; "
                              "outside this range, falls back to the default threshold.")
    parser.add_argument("--dna_contrast_default", type=float, default=1.0,
                         help="Default log2 contrast threshold used when a per-image bimodal "
                              "split isn't statistically supported (too few objects, unimodal "
                              "distribution, or an implausible Otsu result).")
    parser.add_argument("--dna_contrast_min_size_filter", type=int, default=None,
                         help="Minimum object area (px) to keep, applied ALONGSIDE the DNA "
                              "contrast filter (only relevant when --run_dna_contrast_filter "
                              "is set). Objects below this area are dropped even if they pass "
                              "the contrast check -- useful for removing shrunken/pyknotic "
                              "dead-cell nuclei, which can still show real DNA contrast but "
                              "are morphologically abnormal. If omitted, only the contrast "
                              "criterion is applied.")
    args = parser.parse_args()

    well_ids_list = args.well_ids.split(",") if args.well_ids else None
    flow_thresholds = tuple(float(x) for x in args.flow_thresholds.split(","))
    cellprob_thresholds_dna = tuple(float(x) for x in args.cellprob_thresholds_dna.split(","))
    cellprob_thresholds_2ch = tuple(float(x) for x in args.cellprob_thresholds_2ch.split(","))
    cellprob_thresholds_3ch = tuple(float(x) for x in args.cellprob_thresholds_3ch.split(","))

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
         cellprob_thresholds_3ch=cellprob_thresholds_3ch,
         min_size_filter=args.min_size_filter,
         diameter=args.diameter,
         normalize_percentile=normalize_percentile,
         save_cellprob_map=args.save_cellprob_map,
         max_fovs=args.max_fovs,
         skip_two_channel=args.skip_two_channel,
         run_dna_contrast_filter=args.run_dna_contrast_filter,
         dna_contrast_band_px=(args.dna_contrast_band_inner_px, args.dna_contrast_band_outer_px),
         dna_contrast_min_band=args.dna_contrast_min_band,
         dna_contrast_bic_margin=args.dna_contrast_bic_margin,
         dna_contrast_clamp=(args.dna_contrast_clamp_low, args.dna_contrast_clamp_high),
         dna_contrast_default=args.dna_contrast_default,
         dna_contrast_min_size_filter=args.dna_contrast_min_size_filter)
