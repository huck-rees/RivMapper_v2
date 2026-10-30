"""Candidate pieces of a processed channel mask, for the learned cleaning step.

Each processed reach-year mask (``masks/processed/``) is split into

* the **main corridor**: the main path through the reach (least-cost pixel
  route between the points where the river crosses the reach's edges,
  keeping about one typical half-width from the bank) and the channel
  pixels within that half-width + 0.25 W of it: a tube of the river's
  usual width, so lobes and blobs fused to the bank fall outside; and
* **pieces**: connected groups of the remaining channel pixels, at least
  ``MIN_PIECE_W2`` W² in area. A piece touching the corridor once is a bank
  lobe or a dead-end arm (floodplain water, a tributary mouth, a bar head);
  touching it two or more times it is a loop (side channel, braid thread,
  cut-off loop, floodplain water draped along the bank).

The pieces are what a person labels (channel / not channel) and what the
classifier (``ml.py``) judges; the main corridor is never removed. Every
measurement is in units of the channel width W so the features carry
across rivers.

Feature columns (one row per piece):

===================  ===================================================
area_w2              area / W²
length_w             skeleton length / W
width_w              mean width (area / length) / W
max_width_w          2 × largest inscribed radius / W
width_rel_main       width_w / the main channel's width (same units)
compactness          4πA / P² (1 = disc; low = thin or ragged)
raggedness           perimeter / perimeter of a same-area disc
contacts             separate contacts with the main corridor
contact_frac         share of the piece's edge touching the corridor
max_dist_w           farthest pixel from the main corridor / W
core_frac            share on the low-flow core
p_env_mean, p_core_mean   mean probabilities (0–1)
prev1, next1         share of the piece that was channel 1 year before / after
prev2, next2         ... 2 years before / after
freq_all             mean channel frequency of its pixels over all years
area_rel_reach       area / the reach-year's total channel area
===================  ===================================================

Context columns (added by the caller): river, ds_order, year, status,
stage percentile, braided, W in pixels.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import topology

EIGHT = np.ones((3, 3), bool)
MIN_PIECE_W2 = 0.5  # smallest piece considered (W²); smaller ones are noise
CORRIDOR_CAP = 1.0  # corridor half-width cap, × the main channel's typical half-width
CORRIDOR_MARGIN_W = 0.25
GAP_COST = 50.0  # cost per pixel of crossing non-channel ground (bridges mask gaps)

FEATURES = (
    "area_w2",
    "length_w",
    "width_w",
    "max_width_w",
    "width_rel_main",
    "compactness",
    "raggedness",
    "contacts",
    "contact_frac",
    "max_dist_w",
    "core_frac",
    "p_env_mean",
    "p_core_mean",
    "prev1",
    "next1",
    "prev2",
    "next2",
    "freq_all",
    "area_rel_reach",
)


def edge_terminals(channel: np.ndarray, inside: np.ndarray, W: float) -> np.ndarray:
    """Zones where the channel crosses the reach polygon's edge (labelled).

    The raster border counts as an edge too (a channel running off the grid
    leaves the reach there).
    """
    outside = np.pad(~inside, 1, constant_values=True)
    edge = inside & ndi.binary_dilation(outside, EIGHT, iterations=2)[1:-1, 1:-1]
    zone = edge & ndi.binary_dilation(channel, EIGHT, iterations=1)
    lab, _ = ndi.label(ndi.binary_dilation(zone, EIGHT, iterations=max(1, int(W // 2))))
    return np.where(zone, lab, 0).astype(np.int32)


def main_corridor(
    channel: np.ndarray, inside: np.ndarray, W: float, core: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Main-path corridor pixels and the main channel's typical width (px).

    The typical half-width ``h`` is the median distance-to-bank along the
    channel's skeleton. The main path is the least-cost pixel route between
    the points where the channel crosses the reach edge, with a cost that is
    lowest one half-width from the bank (``1 + 4·|d − h| / h``, cheaper on
    the core): it follows the river's normal-width channel and does not
    detour through lobes or blobs fused to the bank, where pixels sit much
    farther from any edge. Non-channel ground inside the reach costs
    ``GAP_COST`` per pixel, so the path bridges short gaps in the mask (a
    cloud, a narrow reach below the resolution) rather than leaving the far
    side of the gap as a "piece", but does not cut across meander necks.
    The corridor is every channel pixel within ``h + 0.25 W`` of that path.
    Without two crossing zones, the largest component's skeleton stands in
    for the path.
    """
    from skimage.graph import MCP_Geometric
    from skimage.morphology import skeletonize

    if not channel.any():
        return np.zeros_like(channel), W
    dt = ndi.distance_transform_edt(channel)
    sk_all = skeletonize(channel)
    h = float(np.median(dt[sk_all])) if sk_all.any() else W / 2
    h = max(h, 0.5)
    term = edge_terminals(channel, inside, W)
    zones = [z for z in np.unique(term) if z]
    path = np.zeros_like(channel)
    if len(zones) >= 2:
        cost = 1.0 + 4.0 * np.abs(dt - h) / h
        if core is not None:
            cost = cost * np.where(core, 0.75, 1.0)
        cost = np.where(channel, cost, np.where(inside, GAP_COST, np.inf))
        mcp = MCP_Geometric(cost, fully_connected=True)
        cum, _ = mcp.find_costs(np.argwhere(term == zones[0]))
        for z in zones[1:]:
            pts = np.argwhere(term == z)
            k = int(np.argmin(cum[pts[:, 0], pts[:, 1]]))
            if np.isfinite(cum[pts[k, 0], pts[k, 1]]):
                tb = np.array(mcp.traceback(tuple(pts[k])))
                path[tb[:, 0], tb[:, 1]] = True
    if not path.any():
        lab, _ = ndi.label(channel, EIGHT)
        big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        path = skeletonize(lab == big)
    if not path.any():
        return channel.copy(), 2 * h
    d = ndi.distance_transform_edt(~path)
    corridor = channel & (d <= CORRIDOR_CAP * h + CORRIDOR_MARGIN_W * W)
    return corridor, 2 * h


def _perimeter(mask: np.ndarray) -> float:
    """Edge length in pixels (4-neighbour transitions)."""
    m = mask.astype(np.int8)
    return float(
        np.abs(np.diff(np.pad(m, ((1, 1), (0, 0))), axis=0)).sum()
        + np.abs(np.diff(np.pad(m, ((0, 0), (1, 1))), axis=1)).sum()
    )


def pieces(
    channel: np.ndarray,
    inside: np.ndarray,
    W: float,
    *,
    core: np.ndarray | None = None,
    p_env: np.ndarray | None = None,
    p_core: np.ndarray | None = None,
    others: dict[int, np.ndarray | None] | None = None,
    freq_all: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Split one reach-year mask into the main corridor and pieces.

    Args:
        channel, inside: this year's channel mask and the reach footprint.
        W: channel width in pixels.
        core: low-flow core mask; p_env, p_core: probabilities in 0–1.
        others: {offset: channel mask} for neighbouring years (−2, −1, 1, 2);
            None where the year has no data.
        freq_all: per-pixel share of observed years that were channel.

    Returns:
        (piece_id int32 array, 0 = not a piece; corridor mask; feature table
        indexed by piece id).
    """
    W = max(float(W), 1.0)
    corridor, main_w = main_corridor(channel, inside, W, core)
    rest = channel & ~corridor
    lab, n = ndi.label(rest, EIGHT)
    if n:
        size = np.bincount(lab.ravel())
        keep = size >= MIN_PIECE_W2 * W * W
        keep[0] = False
        remap = np.zeros(n + 1, np.int32)
        remap[keep] = np.arange(1, keep.sum() + 1)
        lab = remap[lab]
        n = int(keep.sum())
    rows = []
    if n == 0:
        return lab.astype(np.int32), corridor, pd.DataFrame(columns=FEATURES)
    from skimage.morphology import skeletonize

    d_corr = ndi.distance_transform_edt(~corridor) if corridor.any() else None
    corr_ring = ndi.binary_dilation(corridor, EIGHT)
    total = float(channel.sum()) or 1.0
    others = others or {}
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        # pad the slice by 2 px for edges and contacts
        r0, r1 = max(sl[0].start - 2, 0), min(sl[0].stop + 2, lab.shape[0])
        c0, c1 = max(sl[1].start - 2, 0), min(sl[1].stop + 2, lab.shape[1])
        s = (slice(r0, r1), slice(c0, c1))
        m = lab[s] == i
        area = float(m.sum())
        sk = skeletonize(m)
        length = max(float(sk.sum()), 1.0)
        dt = ndi.distance_transform_edt(m)
        perim = _perimeter(m)
        touch = ndi.binary_dilation(m, EIGHT) & corridor[s]
        nc = ndi.label(touch, EIGHT)[1] if touch.any() else 0
        edge_px = m & ~ndi.binary_erosion(m, EIGHT)
        cfrac = float((edge_px & corr_ring[s]).sum() / max(edge_px.sum(), 1))

        def share(arr):
            return float(arr[s][m].mean()) if arr is not None else np.nan

        def persist(k):
            a = others.get(k)
            return float(a[s][m].mean()) if a is not None else np.nan

        width = area / length
        rows.append(
            {
                "piece": i,
                "area_w2": area / (W * W),
                "length_w": length / W,
                "width_w": width / W,
                "max_width_w": 2 * float(dt.max()) / W,
                "width_rel_main": width / max(main_w, 1.0),
                "compactness": 4 * math.pi * area / max(perim, 1.0) ** 2,
                "raggedness": perim / (2 * math.sqrt(math.pi * area)),
                "contacts": int(nc),
                "contact_frac": cfrac,
                "max_dist_w": float(d_corr[s][m].max()) / W if d_corr is not None else np.nan,
                "core_frac": share(core),
                "p_env_mean": share(p_env),
                "p_core_mean": share(p_core),
                "prev1": persist(-1),
                "next1": persist(1),
                "prev2": persist(-2),
                "next2": persist(2),
                "freq_all": share(freq_all),
                "area_rel_reach": area / total,
                # a pixel inside the piece (its widest point): labels refer to
                # pieces by this pixel, so they survive renumbering
                "row": int(np.unravel_index(np.argmax(dt), dt.shape)[0] + r0),
                "col": int(np.unravel_index(np.argmax(dt), dt.shape)[1] + c0),
            }
        )
    table = pd.DataFrame(rows).set_index("piece")
    return lab.astype(np.int32), corridor, table
