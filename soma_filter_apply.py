#!/usr/bin/env python
"""
STEP 2 of 2 -- TEST THE CHOSEN THRESHOLD LOCALLY.

Once soma_filter_explore.py has told you a p_soma cut, run this to see exactly
what that cut does to one or more sites before touching the real pipeline.

    python soma_filter_apply.py \
        --tiff  r01c03f01p01-ch1.tiff r01c03f01p01-ch2.tiff r01c03f01p01-ch3.tiff \
        --channel-names DNA Neurite Cyto \
        --dna DNA \
        --illum-dir /path/with/DNA_illum.npy \
        --diameter 100 \
        --soma-model ./soma_qc/soma_model.joblib \
        --p-cut 0.5 \
        --out-dir ./soma_test

Reuse the --soma-model from step 1. Refitting per site (omit the flag) makes the
cut mean something slightly different on every image, which defeats the purpose.

WHERE THE THRESHOLD GOES IN THE REAL PIPELINE
---------------------------------------------
Two edits inside consumer_worker() in Cellpose_GPU_s3fs_export_tiffs.py.

  1. next to the model construction (~line 222):

        import soma_filter as sf
        cell_model = models.CellposeModel(gpu=(device.type == 'cuda'), device=device)
        SOMA_MODEL = sf.SomaModel.load(SOMA_MODEL_PATH)   # from step 1
        P_CUT      = 0.5                                  # your chosen number

  2. replace the eval call (~line 252):

        masks, _, _ = cell_model.eval(image_4ch, diameter=100)          # before
        masks, _, _ = sf.eval_somas(cell_model, image_4ch, channels,    # after
                                    p_cut=P_CUT, soma_model=SOMA_MODEL,
                                    diameter=100)

     and log what it removed, so a silent over-filter cannot hide:

        st = sf.eval_somas.last['stats']
        logging.info(f"SITE {site_id}: soma filter kept {st['n_kept']}/{st['n_in']} "
                     f"({st['removal_rate']:.0%} removed)")

Nothing else in the consumer changes -- eval_somas returns the same
(masks, flows, styles) tuple, and the label image it returns is renumbered
exactly the way cellpose renumbers its own.

TWO THINGS TO CHECK BEFORE YOU TRUST IT ACROSS A PLATE
------------------------------------------------------
* Removal rate must not track treatment. If treated wells lose more objects
  than DMSO wells, the filter is manufacturing your phenotype. The per-site
  removal rate logged above is what you group by condition to check this.
* The cut will NOT transfer across batches on raw intensities. Refit, or fit on
  per-plate-normalised features, when you move to a new plate.

AND ONE THING IT BREAKS
-----------------------
Filtering renumbers labels, so Cell_Index shifts. This cannot go into a run
whose job is to reproduce an existing single-cell parquet -- the alive/dead
flags would no longer line up. New analyses only.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import soma_filter as sf
from soma_filter_explore import composite_rgb, load_image


def fig_before_after(rgb, masks_before, masks_after, p_cut, out_path):
    from skimage.segmentation import find_boundaries
    from scipy import ndimage as ndi
    sf.apply_style()
    P = sf.PALETTE

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6))
    for ax, m, title, color in [
        (axes[0], masks_before, f"before — {int(masks_before.max())} objects", P["reject"]),
        (axes[1], masks_after, f"after p_soma ≥ {p_cut:g} — {int(masks_after.max())} objects", P["soma"]),
    ]:
        ax.imshow(rgb)
        b = ndi.binary_dilation(find_boundaries(m, mode="outer"), iterations=1)
        layer = np.zeros(b.shape + (4,), dtype=np.float32)
        rgba = matplotlib.colors.to_rgba(color)
        layer[b] = rgba
        ax.imshow(layer, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle("Effect of the chosen cut — the objects that vanished should all be rejects",
                 color=P["ink"], fontsize=11.5, fontweight="semibold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="STEP 2: apply a chosen p_soma cut and inspect the result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tiff", nargs="+", help="one TIFF per channel, in --channel-names order")
    src.add_argument("--stack", help="single multi-channel TIFF instead")
    ap.add_argument("--channel-names", nargs="+", required=True)
    ap.add_argument("--dna", default=None, help="default: first channel")
    ap.add_argument("--illum-dir", default=None)
    ap.add_argument("--diameter", type=float, default=100.0)
    ap.add_argument("--p-cut", type=float, required=True,
                    help="the cut you chose in step 1")
    ap.add_argument("--soma-model", default=None,
                    help="soma_model.joblib from step 1; omit to refit on this image")
    ap.add_argument("--nan-policy", choices=["drop", "keep"], default="drop",
                    help="what to do with objects too small to score")
    ap.add_argument("--rgb", nargs=3, default=None, metavar=("R", "G", "B"))
    ap.add_argument("--gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="gpu", action="store_false")
    ap.add_argument("--out-dir", default="./soma_test")
    args = ap.parse_args()

    names = list(args.channel_names)
    dna = args.dna or names[0]
    if dna not in names:
        sys.exit(f"--dna {dna!r} is not one of {names}")
    os.makedirs(args.out_dir, exist_ok=True)

    image = load_image(args)

    from cellpose import models
    model = models.CellposeModel(gpu=args.gpu)
    print(f"cellpose device: {model.device}")

    masks, flows, _ = sf.segment_permissive(model, image, diameter=args.diameter)
    if int(masks.max()) == 0:
        sys.exit("no objects found")
    df = sf.score_objects(masks, flows, image, names, device=model.device)

    if args.soma_model:
        soma_model = sf.SomaModel.load(args.soma_model)
        soma_model.score(df)
        print(f"scored with {args.soma_model}")
    else:
        soma_model, info = sf.fit_soma_probability(df, dna=dna)
        print("WARNING: refitted on this image; the cut is not comparable across sites.")

    masks_after, stats = sf.apply_cut(masks, df, args.p_cut, nan_policy=args.nan_policy)

    print("\n=== effect of the cut ===")
    print(pd.Series(stats).to_string())
    if stats["removal_rate"] > 0.6:
        print("\n!! more than 60% of objects removed. Either the cut is far too "
              "strict\n   or the mixture picked the wrong component -- check "
              "soma_model.report().")
    elif stats["removal_rate"] < 0.02:
        print("\n!! under 2% removed: this cut is doing essentially nothing.")

    df["keep"] = np.where(np.isfinite(df["p_soma"]), df["p_soma"] >= args.p_cut,
                          args.nan_policy == "keep")
    csv = os.path.join(args.out_dir, f"objects_pcut{args.p_cut:g}.csv")
    df.to_csv(csv, index=False)
    np.save(os.path.join(args.out_dir, f"masks_pcut{args.p_cut:g}.npy"), masks_after)

    rgb_order = args.rgb or ([names[2], names[1], names[0]] if len(names) >= 3
                             else [names[0]] * 3)
    fig = os.path.join(args.out_dir, f"fig_before_after_pcut{args.p_cut:g}.png")
    fig_before_after(composite_rgb(image, names, rgb_order), masks, masks_after,
                     args.p_cut, fig)

    print(f"\nwrote:\n  {csv}\n  {fig}\n"
          f"  {os.path.join(args.out_dir, f'masks_pcut{args.p_cut:g}.npy')}")
    print("\nIf the survivors look right, see the module docstring in this file "
          "for the\ntwo-line patch to Cellpose_GPU_s3fs_export_tiffs.py.")


if __name__ == "__main__":
    main()
