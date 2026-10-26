"""Year-to-year checks, year status, QA table and QA figures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Year status, from the share of a reach's pixels with at least one clear look.
OBSERVED, SPARSE, INFERRED, NO_DATA = "observed", "sparse", "inferred", "no_data"
OBSERVED_MIN = 0.5
SPARSE_MIN = 0.1


def year_status(
    years: Sequence[int], coverage: Sequence[float], max_infer_gap: int
) -> dict[int, str]:
    """Classify each year of one reach.

    * ``observed``: ≥ 50% of pixels have a clear look;
    * ``sparse``: 10–50%;
    * ``inferred``: < 10%, but a year with ≥ 10% lies within
      ``max_infer_gap`` years on *both* sides (the HMM fills it from them);
    * ``no_data``: anything else, e.g. empty years at the start of the record.
    """
    cov = dict(zip(years, coverage, strict=True))
    seen = [y for y in years if cov[y] >= SPARSE_MIN]
    out = {}
    for y in years:
        c = cov[y]
        if c >= OBSERVED_MIN:
            out[y] = OBSERVED
        elif c >= SPARSE_MIN:
            out[y] = SPARSE
        else:
            before = any(0 < y - s <= max_infer_gap for s in seen)
            after = any(0 < s - y <= max_infer_gap for s in seen)
            out[y] = INFERRED if before and after else NO_DATA
    return out


MAD_FLOOR = 0.05  # MAD floor as a share of the median area


def outlier_scores(
    area: Sequence[float], window: int = 5, signed: bool = False
) -> np.ndarray:
    """Robust z-score of each value against its running median.

    score = (A − running median) / (1.4826 × MAD of those residuals). The
    MAD has a floor of 5% of the median area, so on a steady river only a
    change of roughly 15% or more (at the default cut of 3) is flagged.
    Positive scores are too much water, negative too little; ``signed=False``
    returns the absolute value. NaNs are ignored and score NaN.
    """
    a = np.asarray(area, float)
    ok = np.isfinite(a)
    scores = np.full(a.shape, np.nan)
    if ok.sum() < 3:
        return scores
    v = a[ok]
    h = window // 2
    med = np.array([np.median(v[max(0, i - h) : i + h + 1]) for i in range(len(v))])
    resid = v - med
    mad = np.median(np.abs(resid - np.median(resid)))
    mad = max(mad, MAD_FLOOR * float(np.median(v)), 1e-9)
    z = resid / (1.4826 * mad)
    scores[ok] = z if signed else np.abs(z)
    return scores


def reach_figure(rows, thumbs: dict, path, title: str) -> None:
    """Area by year with status/flags, plus mask thumbnails.

    Args:
        rows: QA rows (dicts) for one reach, sorted by year.
        thumbs: {year: (channel, core, inside) boolean arrays} for a few years.
        path: Output PNG.
        title: Figure title.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = [r["year"] for r in rows]
    ch = [r["channel_km2"] for r in rows]
    co = [r["core_km2"] for r in rows]
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, max(len(thumbs), 1), height_ratios=[1.1, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(
        years, ch, "-o", color="#1c5cab", ms=3, lw=1.6, label="channel (near-bankfull)"
    )
    ax.plot(years, co, "-", color="#86b6ef", lw=1.2, label="low-flow core")
    for r in rows:
        y = r["year"]
        if r["status"] == NO_DATA:
            ax.axvspan(y - 0.5, y + 0.5, color="#eeeeee", lw=0)
        elif r["status"] == INFERRED:
            ax.axvspan(y - 0.5, y + 0.5, color="#f9eeda", lw=0)
        elif r["status"] == SPARSE:
            ax.axvspan(y - 0.5, y + 0.5, color="#fbf5ea", lw=0)
        if "area_" in (r.get("flags") or ""):
            ax.plot(y, r["channel_km2"], "x", color="#c2410c", ms=8, mew=2)
    ax.set_ylabel("area (km²)")
    ax.set_ylim(0, None)
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    ax.grid(axis="y", color="#e6e4df")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", fontsize=10)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles, _ = ax.get_legend_handles_labels()
    handles += [
        Patch(color="#fbf5ea", label="sparse"),
        Patch(color="#f9eeda", label="inferred"),
        Patch(color="#eeeeee", label="no data"),
        Line2D([], [], marker="x", ls="", color="#c2410c", label="area outlier"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=6, loc="lower left")
    for j, (y, (chm, com, ins)) in enumerate(sorted(thumbs.items())):
        a = fig.add_subplot(gs[1, j])
        rgb = np.ones(chm.shape + (3,))
        rgb[ins] = (0.96, 0.94, 0.90)
        rgb[chm] = (0.43, 0.65, 0.93)
        rgb[com] = (0.05, 0.20, 0.50)
        a.imshow(rgb, interpolation="nearest")
        a.set_title(str(y), fontsize=9)
        a.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def overview_figure(table, path, title: str) -> None:
    """Channel area by reach and year (heatmap normalised per reach)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = table.pivot(index="ds_order", columns="year", values="channel_km2")
    norm = piv.div(piv.median(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(12, 0.25 * len(piv) + 2))
    im = ax.imshow(norm.values, cmap="RdBu_r", vmin=0.5, vmax=1.5, aspect="auto")
    ax.set_yticks(range(len(piv)), [str(i) for i in piv.index])
    step = max(1, len(piv.columns) // 15)
    ax.set_xticks(
        range(0, len(piv.columns), step), [str(c) for c in piv.columns[::step]]
    )
    ax.set_ylabel("reach")
    ax.set_title(title, loc="left", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("channel area / reach median")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)
