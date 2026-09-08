"""
Per-object soma filtering for Cellpose-SAM (cellpose >= 4.0).

WHY THIS EXISTS
---------------
Cellpose v4 has only four per-object filters -- min_size, max_size_fraction,
flow_threshold and cellprob_threshold -- and none of them can express "this
object has no DNA core" or "this is a flat neurite crossing, not a soma".
Area in particular cannot separate them: real somas span a wide size range that
overlaps the debris.

So we segment PERMISSIVELY (all of cellpose's own censoring switched off),
measure each object, assign it a soma probability, and then remove the rejects
from the label image with the exact same two fastremap calls cellpose uses
internally in utils.fill_holes_and_remove_small_masks. Downstream code sees a
normal cellpose label image and needs no changes.

WHAT GOES INTO THE PROBABILITY
------------------------------
Two families of features, both scale-invariant so they do not just re-encode area:

  Cellpose-internal (free -- already computed during eval)
    cellprob_fill  mean cellprob in the object's core minus its rim. Solid
                   objects score high; annulus-shaped "ring around a rosette"
                   artefacts score low or negative.
    flow_error     dynamics.flow_error -- disagreement between the network's
                   predicted flows and the flows implied by the mask. Catches
                   spindly fragments. NOTE: this is the *same* quantity that
                   flow_threshold cuts on, which is why eval must be run with
                   flow_threshold=0 or the distribution arrives pre-censored.

  Channel signal (needs the illumination-corrected image)
    <chan>_bgsub   object mean minus that object's OWN local background ring.
                   A global background makes the threshold depend on where in
                   the field the object sits; this does not.
    <chan>_central log2(core / rim) intensity. A soma has a centrally
                   concentrated DNA core; a neurite crossing is radially flat.
    clr_<chan>     centred log-ratio of the background-subtracted channels.
                   Raw intensity ratios are compositional and strongly skewed;
                   the CLR makes them roughly Gaussian, which is what the
                   2-component Gaussian mixture actually assumes.

Area is deliberately NOT in the default feature set -- it is kept in the table
for plotting only, so the classifier cannot quietly rediscover a size cut.

Author: added Aug 2026 for the neuronal-rosette troubleshooting.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

# --------------------------------------------------------------------------
# Palette -- values taken from the house data-viz reference palette.
# Only documented, pre-validated slots are used: categorical slots 1-2 and the
# documented blue<->red diverging pair with a neutral gray midpoint.
# Light mode only; a dark variant would need its own steps, not a flip.
# --------------------------------------------------------------------------
PALETTE = {
    "surface":  "#fcfcfb",
    "ink":      "#0b0b0b",
    "ink2":     "#52514e",
    "muted":    "#898781",
    "grid":     "#e1e0d9",
    "baseline": "#c3c2b7",
    "soma":     "#2a78d6",   # categorical slot 1 (blue)
    "reject":   "#eb6834",   # categorical slot 2 (orange)
    "div_lo":   "#e34948",   # diverging warm pole   -> p_soma = 0
    "div_mid":  "#f0efec",   # neutral gray midpoint -> p_soma = 0.5
    "div_hi":   "#2a78d6",   # diverging cool pole   -> p_soma = 1
}


def soma_cmap():
    """Diverging colormap for p_soma: reject <- neutral -> soma."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "p_soma", [PALETTE["div_lo"], PALETTE["div_mid"], PALETTE["div_hi"]]
    )


def apply_style():
    """Recessive grid, hairline axes, ink-token text. Call once per script."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor":  PALETTE["surface"],
        "axes.facecolor":    PALETTE["surface"],
        "savefig.facecolor": PALETTE["surface"],
        "axes.edgecolor":    PALETTE["baseline"],
        "axes.linewidth":    0.8,
        "axes.labelcolor":   PALETTE["ink2"],
        "axes.titlecolor":   PALETTE["ink"],
        "axes.titlesize":    10,
        "axes.titleweight":  "semibold",
        "axes.labelsize":    9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.6,
        "grid.linestyle":    "-",          # never dashed
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.frameon":    False,
        "legend.fontsize":   8,
        "font.size":         9,
    })


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------

def segment_permissive(model, image, diameter=100, **kw):
    """
    Run CellposeModel.eval with every per-object filter switched OFF.

    flow_threshold=0 disables the flow check and min_size=-1 disables the area
    check, so `flow_error` and `area` keep their full distributions and can be
    used as features. If you leave the defaults on, cellpose has already thrown
    away the objects you are trying to characterise.
    """
    kw.pop("flow_threshold", None)
    kw.pop("min_size", None)
    return model.eval(image, diameter=diameter, flow_threshold=0, min_size=-1, **kw)


# --------------------------------------------------------------------------
# Per-object measurement
# --------------------------------------------------------------------------

def _obj_mean(values, labelimg, index):
    """ndi.mean that yields NaN (not a warning) for labels absent from labelimg."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = ndi.mean(values, labelimg, index=index)
    return np.asarray(out, dtype=np.float64)


def score_objects(masks, flows, image, channel_names, inner=0.6, outer=0.3,
                  bg_lo=2.0, bg_hi=12.0, device=None, compute_flow_error=True):
    """
    Build the per-object feature table.

    Parameters
    ----------
    masks : (H, W) int label image straight out of eval.
    flows : the `flows` list eval returned; flows[1] is dP, flows[2] is cellprob.
    image : (H, W, C) float array, ILLUMINATION-CORRECTED, channels in the same
            order as `channel_names`. A 2D array is accepted as a single channel.
    channel_names : list of names, e.g. ['DNA', 'Neurite', 'Cyto'].
    inner, outer : normalised-radius cuts defining the object's core (>= inner)
            and rim (<= outer). Normalised per object, so they are scale-free.
    bg_lo, bg_hi : the local-background ring, in pixels outside the mask.
    device : torch device for flow_error; defaults to CPU. Pass model.device to
            use the GPU (much faster, but it is a real per-site cost).

    Returns
    -------
    DataFrame, one row per object, indexed 0..N-1 with a `label` column holding
    the cellpose label id. Objects too small to have core or rim pixels come
    back with NaN in the affected columns.
    """
    masks = np.asarray(masks)
    if masks.ndim != 2:
        raise ValueError(f"score_objects expects a 2D label image, got {masks.ndim}D")

    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[:2] != masks.shape:
        raise ValueError(f"image {image.shape[:2]} does not match masks {masks.shape}")
    if image.shape[-1] != len(channel_names):
        raise ValueError(f"{image.shape[-1]} channels but {len(channel_names)} names given")

    labs = np.unique(masks)
    labs = labs[labs != 0].astype(np.int64)
    if labs.size == 0:
        return pd.DataFrame(columns=["label", "area"])

    cellprob = np.asarray(flows[2], dtype=np.float32)
    dP = np.asarray(flows[1], dtype=np.float32)
    if cellprob.shape != masks.shape:
        raise ValueError(
            f"cellprob {cellprob.shape} != masks {masks.shape}. "
            "Was eval run with resample=False? This code assumes resample=True "
            "so that cellprob is at full resolution."
        )

    fg = masks > 0

    # --- normalised radial position: 0 at the boundary, 1 at the centre -----
    dist = ndi.distance_transform_edt(fg)
    dmax = np.asarray(ndi.maximum(dist, masks, labs), dtype=np.float64)
    lut = np.zeros(int(masks.max()) + 1, dtype=np.int64)   # label -> row position
    lut[labs] = np.arange(labs.size)
    dnorm = np.zeros_like(dist)
    dnorm[fg] = dist[fg] / np.maximum(dmax[lut[masks[fg]]], 1e-6)

    core = np.where(fg & (dnorm >= inner), masks, 0)
    rim = np.where(fg & (dnorm <= outer), masks, 0)

    # --- local background ring, assigned to its NEAREST object --------------
    bg_dist, inds = ndi.distance_transform_edt(~fg, return_indices=True)
    iy, ix = inds
    band = (~fg) & (bg_dist >= bg_lo) & (bg_dist <= bg_hi)
    bg_lab = np.where(band, masks[iy, ix], 0)

    cy_cx = ndi.center_of_mass(fg, masks, labs)
    cy = np.array([p[0] for p in cy_cx])
    cx = np.array([p[1] for p in cy_cx])

    cp_core = _obj_mean(cellprob, core, labs)
    cp_rim = _obj_mean(cellprob, rim, labs)

    df = pd.DataFrame({
        "label": labs,
        "centroid_y": cy,
        "centroid_x": cx,
        "area": np.asarray(ndi.sum(fg, masks, labs), dtype=np.float64),
        "cellprob_mean": _obj_mean(cellprob, masks, labs),
        "cellprob_core": cp_core,
        "cellprob_rim": cp_rim,
        "cellprob_fill": cp_core - cp_rim,
    })
    df["log_area"] = np.log10(df["area"].clip(lower=1))

    # --- cellpose flow error, one value per label ---------------------------
    if compute_flow_error:
        import torch
        from cellpose import dynamics
        dev = device if device is not None else torch.device("cpu")
        fe, _ = dynamics.flow_error(masks.astype(np.int32), dP, device=dev)
        df["flow_error"] = np.asarray(fe, dtype=np.float64)[labs - 1]

    # --- channel signal ----------------------------------------------------
    for c, name in enumerate(channel_names):
        ch = image[..., c].astype(np.float32)
        m_obj = _obj_mean(ch, masks, labs)
        m_bg = _obj_mean(ch, bg_lab, labs)
        m_core = _obj_mean(ch, core, labs)
        m_rim = _obj_mean(ch, rim, labs)
        df[f"{name}_mean"] = m_obj
        df[f"{name}_bg"] = m_bg
        df[f"{name}_bgsub"] = m_obj - m_bg
        df[f"{name}_integrated"] = m_obj * df["area"]
        df[f"{name}_central"] = np.log2((m_core + 1.0) / (m_rim + 1.0))

    # --- centred log-ratios across channels --------------------------------
    if len(channel_names) >= 2:
        sig = np.stack(
            [np.clip(df[f"{n}_bgsub"].to_numpy(dtype=np.float64), 1.0, None)
             for n in channel_names], axis=1)
        lg = np.log(sig)
        clr = lg - lg.mean(axis=1, keepdims=True)
        for i, n in enumerate(channel_names):
            df[f"clr_{n}"] = clr[:, i]

    return df


# --------------------------------------------------------------------------
# Soma probability
# --------------------------------------------------------------------------

class SomaModel:
    """
    Fitted 2-component Gaussian mixture giving P(soma) per object.

    Unsupervised on purpose: it needs no hand labels. The trade-off is that it
    assumes the rejects form ONE cluster. If your rejects are several unrelated
    populations (neurites AND debris AND pyknotic nuclei), check the ambiguous
    fraction it reports -- a large one means a single cut is not the right tool
    and you should carry p_soma forward as a weight instead.
    """

    def __init__(self, gm, features, mu, sd, soma_component, dna):
        self.gm = gm
        self.features = list(features)
        self.mu = mu
        self.sd = sd
        self.soma_component = int(soma_component)
        self.dna = dna

    def predict(self, df):
        """Return p_soma per row; NaN where any feature is missing."""
        X = df[self.features].replace([np.inf, -np.inf], np.nan)
        ok = X.notna().all(axis=1)
        p = np.full(len(df), np.nan)
        if ok.any():
            Z = (X[ok] - self.mu) / self.sd
            p[np.flatnonzero(ok.to_numpy())] = \
                self.gm.predict_proba(Z)[:, self.soma_component]
        return p

    def score(self, df):
        """Add a `p_soma` column in place and return the DataFrame."""
        df["p_soma"] = self.predict(df)
        return df

    def report(self):
        """Per-component feature means in ORIGINAL units, for sanity-checking."""
        means = self.gm.means_ * self.sd.to_numpy() + self.mu.to_numpy()
        out = pd.DataFrame(means, columns=self.features,
                           index=[f"component_{i}" for i in range(means.shape[0])])
        out.insert(0, "weight", self.gm.weights_)
        out.insert(0, "role", ["soma" if i == self.soma_component else "reject"
                               for i in range(means.shape[0])])
        return out

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)


def default_features(df, dna="DNA"):
    """
    The feature set used unless you override it.

    `area` / `log_area` are excluded on purpose: the whole point is a criterion
    that is NOT a size cut. Add 'log_area' explicitly if you decide you want it.
    """
    wanted = [f"clr_{dna}", f"{dna}_central", "cellprob_fill", "flow_error"]
    return [f for f in wanted if f in df.columns]


def fit_soma_probability(df, dna="DNA", features=None, random_state=0, n_init=8):
    """
    Fit the mixture and write `p_soma` into `df` (in place).

    The soma component is identified as the one with the higher mean clr_<DNA>
    (falling back to the first feature). ALWAYS eyeball model.report() to check
    that assignment -- if it picked the wrong component the probabilities are
    inverted, and that is the single most likely way this goes wrong.

    Returns (SomaModel, info_dict).
    """
    from sklearn.mixture import GaussianMixture

    feats = list(features) if features else default_features(df, dna)
    if len(feats) < 2:
        raise ValueError(f"need >= 2 usable features, got {feats}")

    X = df[feats].replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1)
    if ok.sum() < 20:
        raise ValueError(
            f"only {int(ok.sum())} objects have all features -- too few to fit a "
            "mixture. Use more sites, or drop the feature with the most NaNs."
        )

    Xo = X[ok]
    mu = Xo.mean()
    sd = Xo.std(ddof=0).replace(0.0, 1.0)
    Z = (Xo - mu) / sd

    gm = GaussianMixture(n_components=2, covariance_type="full",
                         random_state=random_state, n_init=n_init).fit(Z)

    key = f"clr_{dna}" if f"clr_{dna}" in feats else feats[0]
    k = int(np.argmax(gm.means_[:, feats.index(key)]))

    model = SomaModel(gm, feats, mu, sd, k, dna)
    model.score(df)

    p = df["p_soma"].to_numpy()
    finite = np.isfinite(p)
    ambiguous = float(np.mean((p[finite] > 0.1) & (p[finite] < 0.9))) if finite.any() else np.nan
    info = {
        "features": feats,
        "component_used_for_soma": k,
        "identified_by": key,
        "n_objects": int(len(df)),
        "n_scored": int(finite.sum()),
        "n_unscored_nan": int((~finite).sum()),
        "ambiguous_fraction_0.1_0.9": ambiguous,
        "bic": float(gm.bic(Z)),
    }
    return model, info


# --------------------------------------------------------------------------
# Applying the threshold
# --------------------------------------------------------------------------

def apply_cut(masks, df, p_cut, nan_policy="drop", renumber=True):
    """
    Remove rejected objects from the label image.

    Uses the same two fastremap calls cellpose itself uses in
    utils.fill_holes_and_remove_small_masks, so the result is indistinguishable
    from a label image cellpose produced.

    nan_policy : 'drop' (default) or 'keep' -- what to do with objects that
        could not be scored (usually objects too small to have core/rim pixels).
        Either way the count is reported rather than silently applied.

    WARNING: renumbering changes every surviving label id. Any downstream index
    derived from label order (e.g. Cell_Index) shifts. Do not retrofit this into
    a pipeline that must reproduce an existing single-cell table.

    Returns (filtered_masks, stats_dict).
    """
    import fastremap

    if "p_soma" not in df.columns:
        raise ValueError("df has no p_soma column -- run fit_soma_probability first")

    p = df["p_soma"].to_numpy()
    unscored = ~np.isfinite(p)
    keep = np.where(unscored, nan_policy == "keep", p >= p_cut)

    drop_labels = df.loc[~keep, "label"].to_numpy()
    out = np.asarray(masks).copy()
    if drop_labels.size:
        out = fastremap.mask(out, drop_labels.astype(out.dtype))
    if renumber:
        fastremap.renumber(out, in_place=True)

    stats = {
        "p_cut": float(p_cut),
        "n_in": int(len(df)),
        "n_kept": int(keep.sum()),
        "n_dropped": int((~keep).sum()),
        "n_unscored": int(unscored.sum()),
        "nan_policy": nan_policy,
        "removal_rate": float((~keep).sum() / max(len(df), 1)),
    }
    return out, stats


def eval_somas(model, image, channel_names, p_cut=0.5, soma_model=None,
               diameter=100, dna="DNA", nan_policy="drop", **kw):
    """
    Drop-in replacement for CellposeModel.eval that returns soma-only masks.

    Returns (masks, flows, styles) like eval does, plus the feature table and
    stats via the `.last` attribute set on this function (see below), so the
    call site in an existing pipeline changes by one word:

        masks, _, _ = cell_model.eval(image_4ch, diameter=100)
        masks, _, _ = eval_somas(cell_model, image_4ch, CHANNELS, p_cut=0.5,
                                 soma_model=SOMA_MODEL, diameter=100)

    Pass a `soma_model` fitted once on representative data. Fitting per-site is
    possible (soma_model=None) but the threshold then means something slightly
    different on every image, which is usually not what you want.
    """
    masks, flows, styles = segment_permissive(model, image, diameter=diameter, **kw)
    df = score_objects(masks, flows, image, channel_names, device=model.device)
    if df.empty:
        eval_somas.last = {"table": df, "stats": {"n_in": 0, "n_kept": 0, "n_dropped": 0}}
        return masks, flows, styles

    if soma_model is None:
        soma_model, _ = fit_soma_probability(df, dna=dna)
    else:
        soma_model.score(df)

    masks, stats = apply_cut(masks, df, p_cut, nan_policy=nan_policy)
    eval_somas.last = {"table": df, "stats": stats, "soma_model": soma_model}
    return masks, flows, styles


eval_somas.last = None
