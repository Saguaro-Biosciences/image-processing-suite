#!/usr/bin/env python
"""
STEP 1 of 2 -- DECIDE THE THRESHOLD.

Run this on one representative site. It segments permissively, measures every
object, fits a soma probability, and writes out a CSV plus three figures. You
look at the figures, pick a p_soma cut, and hand that number to
soma_filter_apply.py (STEP 2).

    python soma_filter_explore.py \
        --tiff  r01c03f01p01-ch1.tiff r01c03f01p01-ch2.tiff r01c03f01p01-ch3.tiff \
        --channel-names DNA Neurite Cyto \
        --dna DNA \
        --illum-dir /path/with/DNA_illum.npy \
        --diameter 100 \
        --out-dir ./soma_qc

Pass the channel TIFFs in the SAME ORDER as --channel-names. Use --illum-dir if
you have the illumination-correction arrays: the background-subtraction features
assume a flat field, and an uncorrected image makes the threshold depend on where
in the field an object happens to sit.

WHAT TO LOOK AT, IN ORDER
-------------------------
1. fig1 panel A, the p_soma histogram. Two clear humps means a threshold is
   justified and the valley between them is where to put it. One broad hump
   means it is a continuum -- a hard cut is arbitrary and you should carry
   p_soma forward as a weight instead. Decide this FIRST.
2. fig1 panel C, area vs DNA. This is the panel that shows whether area could
   ever have worked. If the rejects sit inside the same area range as the
   somas, min_size was never going to do it.
3. fig3, the contact sheet. The middle row is the objects nearest your cut --
   those are the only ones whose classification is actually in question. If the
   middle row looks like somas, lower the cut; if it looks like debris, raise it.
4. The removal-rate table printed at the end. Sanity-check that the fraction
   removed is plausible for this image before committing.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

import soma_filter as sf


# --------------------------------------------------------------------------
def load_image(args):
    """Return (H, W, C) float32, illumination-corrected if --illum-dir given."""
    if args.stack:
        arr = tifffile.imread(args.stack).astype(np.float32)
        if arr.ndim != 3:
            sys.exit(f"--stack must be 3D, got shape {arr.shape}")
        # put the short axis last
        if arr.shape[0] <= 5 and arr.shape[0] < min(arr.shape[1:]):
            arr = np.moveaxis(arr, 0, -1)
    else:
        planes = [tifffile.imread(p).astype(np.float32) for p in args.tiff]
        shapes = {p.shape for p in planes}
        if len(shapes) != 1:
            sys.exit(f"channel TIFFs have different shapes: {shapes}")
        arr = np.stack(planes, axis=-1)

    if arr.shape[-1] != len(args.channel_names):
        sys.exit(f"loaded {arr.shape[-1]} channels but got "
                 f"{len(args.channel_names)} --channel-names")

    if args.illum_dir:
        for i, name in enumerate(args.channel_names):
            path = os.path.join(args.illum_dir, f"{name}_illum.npy")
            if not os.path.exists(path):
                sys.exit(f"illumination array not found: {path}")
            corr = np.load(path).astype(np.float32)
            if corr.shape != arr.shape[:2]:
                sys.exit(f"{path} shape {corr.shape} != image {arr.shape[:2]}")
            arr[..., i] = arr[..., i] / np.maximum(corr, 1e-6)
        print(f"applied illumination correction from {args.illum_dir}")
    else:
        print("WARNING: no --illum-dir. Background-subtracted features will drift "
              "with field position and your threshold will not transfer.")
    return arr


def composite_rgb(image, channel_names, rgb):
    """Percentile-stretched RGB view for the overlay and contact sheet."""
    out = np.zeros(image.shape[:2] + (3,), dtype=np.float32)
    for slot, name in enumerate(rgb):
        if name is None or name not in channel_names:
            continue
        ch = image[..., channel_names.index(name)]
        lo, hi = np.percentile(ch, [1, 99.5])
        out[..., slot] = np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def crop(img, cy, cx, half):
    """Padded square crop centred on (cy, cx); works at the image edges."""
    h, w = img.shape[:2]
    y0, y1 = int(cy) - half, int(cy) + half
    x0, x1 = int(cx) - half, int(cx) + half
    pad = [(max(0, -y0), max(0, y1 - h)), (max(0, -x0), max(0, x1 - w))]
    if img.ndim == 3:
        pad.append((0, 0))
    sub = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    return np.pad(sub, pad, mode="constant")


# --------------------------------------------------------------------------
def fig_diagnostics(df, dna, p_cut, out_path):
    sf.apply_style()
    cmap = sf.soma_cmap()
    P = sf.PALETTE
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4))
    fig.subplots_adjust(hspace=0.34, wspace=0.28, right=0.90,
                        top=0.90, bottom=0.09, left=0.06)

    p = df["p_soma"].to_numpy()
    ok = np.isfinite(p)
    is_soma = ok & (p >= p_cut)
    scat = dict(s=16, linewidths=0.4, edgecolors=P["baseline"],
                cmap=cmap, vmin=0, vmax=1)

    def rule(ax, x, label):
        ax.axvline(x, color=P["muted"], lw=1.4, zorder=1)
        ax.annotate(label, xy=(x, 0.97), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points",
                    color=P["ink2"], fontsize=8, va="top")

    # --- A: is a threshold even justified? --------------------------------
    ax = axes[0, 0]
    ax.hist(p[ok], bins=40, range=(0, 1), color=P["soma"], edgecolor=P["surface"],
            linewidth=0.5)
    ax.axvspan(0.1, 0.9, color=P["grid"], zorder=0)
    rule(ax, p_cut, f"cut {p_cut:g}")
    ax.set_title("A  P(soma) — two humps ⇒ a cut is justified")
    ax.set_xlabel("p_soma"); ax.set_ylabel("objects")
    ax.grid(axis="y")

    # --- B: the cloud ------------------------------------------------------
    ax = axes[0, 1]
    xk, yk = f"clr_{dna}", f"{dna}_central"
    if xk in df and yk in df:
        sc = ax.scatter(df.loc[ok, xk], df.loc[ok, yk], c=p[ok], **scat)
        ax.set_xlabel(f"clr_{dna}  (relative {dna} signal)")
        ax.set_ylabel(f"{dna}_central  (log2 core / rim)")
    ax.set_title("B  the cloud: how much DNA, how centred")
    ax.grid(True)

    # --- C: could area ever have worked? -----------------------------------
    ax = axes[0, 2]
    yk = f"{dna}_bgsub"
    if yk in df:
        ax.scatter(df.loc[ok, "area"], df.loc[ok, yk].clip(lower=0.5), c=p[ok], **scat)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("area (px)"); ax.set_ylabel(f"{dna}_bgsub")
    ax.set_title("C  area vs DNA — overlap here ⇒ min_size cannot work")
    ax.grid(True)

    # --- D: the two cellpose-internal metrics ------------------------------
    ax = axes[1, 0]
    if "flow_error" in df:
        ax.scatter(df.loc[ok, "flow_error"], df.loc[ok, "cellprob_fill"], c=p[ok], **scat)
        ax.axvline(0.4, color=P["muted"], lw=1.4)
        ax.annotate("cellpose default\nflow_threshold 0.4", xy=(0.4, 0.03),
                    xycoords=("data", "axes fraction"), xytext=(6, 2),
                    textcoords="offset points", color=P["ink2"], fontsize=7.5,
                    bbox=dict(facecolor=P["surface"], edgecolor="none", pad=1.5))
        ax.set_xlabel("flow_error"); ax.set_ylabel("cellprob_fill (core − rim)")
    ax.set_title("D  cellpose internals — native cut available on x")
    ax.grid(True)

    # --- E, F: marginals, split by class ------------------------------------
    for ax, key, title in [
        (axes[1, 1], f"{dna}_bgsub", f"E  {dna} above local background"),
        (axes[1, 2], "area", "F  area — the axis that does NOT separate"),
    ]:
        if key not in df:
            continue
        v = df[key].to_numpy(dtype=float)
        good = np.isfinite(v) & ok
        lo, hi = np.nanpercentile(v[good], [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        ax.hist(v[good & is_soma], bins=bins, color=P["soma"], alpha=0.85,
                label=f"soma (p ≥ {p_cut:g})", edgecolor=P["surface"], linewidth=0.5)
        ax.hist(v[good & ~is_soma], bins=bins, color=P["reject"], alpha=0.85,
                label="reject", edgecolor=P["surface"], linewidth=0.5)
        ax.set_title(title); ax.set_xlabel(key); ax.set_ylabel("objects")
        ax.legend(loc="upper right"); ax.grid(axis="y")

    cax = fig.add_axes([0.915, 0.12, 0.012, 0.74])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax)
    cb.set_label("p_soma", color=P["ink2"], fontsize=8)
    cb.ax.tick_params(labelsize=7, color=P["muted"], labelcolor=P["muted"])
    cb.outline.set_visible(False)

    fig.suptitle("Soma-filter diagnostics — panels A and C decide whether a cut is the right tool",
                 color=P["ink"], fontsize=11.5, fontweight="semibold", x=0.06, ha="left")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_overlay(rgb, masks, df, p_cut, out_path):
    from skimage.segmentation import find_boundaries
    sf.apply_style()
    P = sf.PALETTE
    cmap = sf.soma_cmap(); cmap.set_bad((0, 0, 0, 0))

    lut = np.full(int(masks.max()) + 1, np.nan)
    lut[df["label"].to_numpy()] = df["p_soma"].to_numpy()
    pmap = lut[masks]
    b = ndi.binary_dilation(find_boundaries(masks, mode="outer"), iterations=1)
    layer = np.where(b, pmap, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6))
    axes[0].imshow(rgb); axes[0].set_title("all objects, coloured by p_soma")
    axes[0].imshow(np.ma.masked_invalid(layer), cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    keep_lut = np.zeros(int(masks.max()) + 1, dtype=bool)
    kept = df.loc[df["p_soma"] >= p_cut, "label"].to_numpy()
    keep_lut[kept] = True
    layer2 = np.where(b & keep_lut[masks], pmap, np.nan)
    axes[1].imshow(rgb); axes[1].set_title(f"survivors at p_soma ≥ {p_cut:g}  (n={len(kept)})")
    axes[1].imshow(np.ma.masked_invalid(layer2), cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Before / after — check the survivors are the objects you wanted",
                 color=P["ink"], fontsize=11.5, fontweight="semibold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_contact_sheet(rgb, df, p_cut, out_path, ncol=8, half=60, band=0.25):
    """
    Three rows: clear somas, the genuinely ambiguous band, clear rejects.

    Each row draws ONLY from objects on the correct side of the cut, and pads
    with blanks when there are fewer than ncol of them. An early version
    back-filled short rows from the other class, which silently mislabelled
    high-p somas as "rejects" -- exactly the kind of thing that would send you
    chasing the wrong threshold.
    """
    sf.apply_style()
    P = sf.PALETTE
    d = df[np.isfinite(df["p_soma"])].copy()
    if d.empty:
        return
    d["dist"] = (d["p_soma"] - p_cut).abs()
    somas = d[d["p_soma"] >= p_cut]
    rejects = d[d["p_soma"] < p_cut]
    amb = d[d["dist"] <= band]

    blocks = [
        ("clear somas (highest p)", somas.nlargest(ncol, "p_soma"), len(somas)),
        (f"AMBIGUOUS  |p − cut| ≤ {band:g}", amb.nsmallest(ncol, "dist"), len(amb)),
        ("clear rejects (lowest p)", rejects.nsmallest(ncol, "p_soma"), len(rejects)),
    ]

    fig, axes = plt.subplots(len(blocks), ncol,
                             figsize=(1.55 * ncol + 1.8, 1.9 * len(blocks)))
    axes = np.atleast_2d(axes)
    for r, (_, block, _) in enumerate(blocks):
        for c in range(ncol):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c < len(block):
                row = block.iloc[c]
                ax.imshow(crop(rgb, row["centroid_y"], row["centroid_x"], half))
                ax.set_title(f"p={row['p_soma']:.2f}", fontsize=7.5,
                             color=P["ink2"], pad=2)

    fig.subplots_adjust(left=0.16, right=0.99, top=0.85, bottom=0.02,
                        hspace=0.30, wspace=0.06)
    for r, (title, block, total) in enumerate(blocks):
        bb = axes[r, 0].get_position()
        note = f"(n={total}" + (f", showing {len(block)})" if len(block) < total else ")")
        fig.text(0.15, 0.5 * (bb.y0 + bb.y1), f"{title}\n{note}",
                 ha="right", va="center", fontsize=8, color=P["ink"])

    fig.suptitle("Boundary review — only the middle row is actually in question",
                 color=P["ink"], fontsize=11, fontweight="semibold", x=0.02, ha="left")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="STEP 1: measure objects and decide a p_soma threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tiff", nargs="+", help="one TIFF per channel, in --channel-names order")
    src.add_argument("--stack", help="single multi-channel TIFF instead")
    ap.add_argument("--channel-names", nargs="+", required=True,
                    help="channel names, same order as --tiff")
    ap.add_argument("--dna", default=None, help="which channel is the DNA/nuclear stain "
                                               "(default: first channel)")
    ap.add_argument("--illum-dir", default=None,
                    help="folder with <channel>_illum.npy correction arrays")
    ap.add_argument("--diameter", type=float, default=100.0,
                    help="cellpose diameter; keep whatever your pipeline uses")
    ap.add_argument("--p-cut", type=float, default=0.5,
                    help="provisional cut, only for the reference lines and contact sheet")
    ap.add_argument("--rgb", nargs=3, default=None, metavar=("R", "G", "B"),
                    help="channel names to put in R,G,B for the pictures "
                         "(default: channels 3,2,1 so DNA lands in blue)")
    ap.add_argument("--gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="gpu", action="store_false")
    ap.add_argument("--out-dir", default="./soma_qc")
    args = ap.parse_args()

    names = list(args.channel_names)
    dna = args.dna or names[0]
    if dna not in names:
        sys.exit(f"--dna {dna!r} is not one of {names}")
    os.makedirs(args.out_dir, exist_ok=True)

    image = load_image(args)
    print(f"image {image.shape}, channels {names}, DNA={dna}")

    from cellpose import models
    model = models.CellposeModel(gpu=args.gpu)
    print(f"cellpose device: {model.device}")

    print("segmenting permissively (flow_threshold=0, min_size=-1) ...")
    masks, flows, _ = sf.segment_permissive(model, image, diameter=args.diameter)
    n = int(masks.max())
    print(f"  {n} raw objects")
    if n == 0:
        sys.exit("no objects found -- check the channel order and --diameter")

    print("measuring ...")
    df = sf.score_objects(masks, flows, image, names, device=model.device)

    print("fitting the 2-component mixture ...")
    soma_model, info = sf.fit_soma_probability(df, dna=dna)

    # ---- outputs ---------------------------------------------------------
    csv = os.path.join(args.out_dir, "objects.csv")
    df.to_csv(csv, index=False)
    soma_model.save(os.path.join(args.out_dir, "soma_model.joblib"))
    np.save(os.path.join(args.out_dir, "masks_permissive.npy"), masks)

    rgb_order = args.rgb or ([names[2], names[1], names[0]] if len(names) >= 3
                             else [names[0]] * 3)
    rgb = composite_rgb(image, names, rgb_order)

    f1 = os.path.join(args.out_dir, "fig1_diagnostics.png")
    f2 = os.path.join(args.out_dir, "fig2_overlay.png")
    f3 = os.path.join(args.out_dir, "fig3_boundary_review.png")
    fig_diagnostics(df, dna, args.p_cut, f1)
    fig_overlay(rgb, masks, df, args.p_cut, f2)
    fig_contact_sheet(rgb, df, args.p_cut, f3)

    # ---- what to read ----------------------------------------------------
    print("\n=== component assignment (CHECK THIS FIRST) ===")
    print("If the row marked 'soma' does not have the higher DNA signal, the")
    print("probabilities are inverted -- use 1 - p_soma, or set the features by hand.")
    print(soma_model.report().to_string(float_format=lambda v: f"{v: .3f}"))

    print("\n=== fit info ===")
    print(json.dumps(info, indent=2, default=str))
    amb = info["ambiguous_fraction_0.1_0.9"]
    if amb is not None and amb > 0.35:
        print(f"\n!! {amb:.0%} of objects fall between p=0.1 and p=0.9. That is a "
              "continuum,\n   not two populations. A hard cut here is arbitrary -- "
              "prefer carrying\n   p_soma forward as a weight in the well-level "
              "aggregation.")

    print("\n=== removal rate vs cut (pick from here, then look at fig3) ===")
    p = df["p_soma"].to_numpy()
    rows = []
    for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        keep = np.isfinite(p) & (p >= c)
        rows.append({"p_cut": c, "kept": int(keep.sum()),
                     "dropped": int(len(p) - keep.sum()),
                     "removed_%": round(100 * (1 - keep.sum() / len(p)), 1)})
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\nwrote:\n  {csv}\n  {f1}\n  {f2}\n  {f3}\n"
          f"  {os.path.join(args.out_dir, 'soma_model.joblib')}")
    print("\nNext: pick a cut and run soma_filter_apply.py --p-cut <value>")


if __name__ == "__main__":
    main()
