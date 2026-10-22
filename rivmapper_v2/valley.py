"""Valley line, width model and valley-length reach breaks.

These functions turn an ordered set of SWORD nodes (position, width,
drainage area) into:

1. a **width model** ``W(A)``: a power law of channel width on drainage
   area, fitted per river, used to set reach length;
2. a **valley line**: the channel centreline smoothed at a scale tied to
   channel width and local sinuosity, so individual meander bends are
   removed but valley-scale bends are kept;
3. **reach breaks** spaced ``k × W(A)`` apart in down-valley distance.

See ``docs/methods/reaches.md`` for the rationale and citations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Rolling-median window (nodes) for the local width field. SWORD nodes are
# ~200 m apart, so 41 nodes ≈ 8 km: long enough to suppress anabranch /
# confluence width spikes, short enough to follow real downstream change.
WIDTH_WINDOW_NODES = 41
# Resolution of the uniform grid in width-normalised arc length (widths).
_GRID_STEP = 0.5


@dataclass
class WidthModel:
    """``W(A) = coef × A**exponent`` fitted to one river's SWORD widths."""

    coef: float
    exponent: float
    n_nodes: int
    method: str  # "power_law" or "constant" (too little drainage-area range)

    def __call__(self, area_km2: np.ndarray | float) -> np.ndarray:
        return self.coef * np.asarray(area_km2, dtype=float) ** self.exponent


def fit_width_model(width_m: np.ndarray, area_km2: np.ndarray) -> WidthModel:
    """Fit ``W = c A^b`` to node widths using binned medians in log space.

    Binning by drainage area (20 equal-log bins) and taking median widths
    makes the fit robust to noisy wetted widths, anabranches and the many
    nodes clustered at similar drainage area. The exponent is constrained to
    be non-negative so reach length never shrinks downstream. If drainage
    area barely changes along the river, a constant (median) width is used.
    """
    w = np.asarray(width_m, dtype=float)
    a = np.asarray(area_km2, dtype=float)
    ok = np.isfinite(w) & np.isfinite(a) & (w > 0) & (a > 0)
    w, a = w[ok], a[ok]
    if len(w) == 0:
        raise ValueError("no valid SWORD node widths / drainage areas")
    la, lw = np.log(a), np.log(w)
    if np.ptp(la) < np.log(1.2):  # < 20% change in drainage area
        return WidthModel(float(np.median(w)), 0.0, len(w), "constant")

    edges = np.linspace(la.min(), la.max(), 21)
    idx = np.clip(np.digitize(la, edges) - 1, 0, 19)
    bx, by = [], []
    for i in range(20):
        sel = idx == i
        if sel.sum() >= 5:
            bx.append(np.median(la[sel]))
            by.append(np.median(lw[sel]))
    if len(bx) < 3:
        return WidthModel(float(np.median(w)), 0.0, len(w), "constant")
    b, ln_c = np.polyfit(bx, by, 1)
    if b < 0:
        return WidthModel(float(np.median(w)), 0.0, len(w), "constant")
    return WidthModel(float(np.exp(ln_c)), float(b), len(w), "power_law")


def local_width(width_m: np.ndarray, window: int = WIDTH_WINDOW_NODES) -> np.ndarray:
    """Rolling median of node width in log space (gaps filled from neighbours)."""
    w = np.asarray(width_m, dtype=float).copy()
    w[~(w > 0)] = np.nan
    lw = pd.Series(np.log(w)).rolling(window, center=True, min_periods=3).median()
    lw = lw.interpolate(limit_direction="both")
    if lw.isna().all():
        raise ValueError("no valid SWORD node widths")
    return np.exp(lw.to_numpy())


def _smooth_xy(
    x: np.ndarray, y: np.ndarray, sstar: np.ndarray, sigma: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-smooth x, y as functions of width-normalised arc length.

    ``sigma`` (in channel widths) may vary along the river; it is applied by
    blending a ladder of fixed-sigma passes. Ends are reflected.
    """
    grid = np.arange(0.0, sstar[-1] + _GRID_STEP, _GRID_STEP)
    gx, gy = np.interp(grid, sstar, x), np.interp(grid, sstar, y)
    if np.isscalar(sigma):
        sx = gaussian_filter1d(gx, float(sigma) / _GRID_STEP, mode="reflect")
        sy = gaussian_filter1d(gy, float(sigma) / _GRID_STEP, mode="reflect")
    else:
        sig = np.interp(grid, sstar, np.asarray(sigma, dtype=float))
        levels = np.unique(np.round(np.geomspace(sig.min(), sig.max(), 8), 2))
        if len(levels) == 1:
            return _smooth_xy(x, y, sstar, float(levels[0]))
        bx = np.array(
            [gaussian_filter1d(gx, s / _GRID_STEP, mode="reflect") for s in levels]
        )
        by = np.array(
            [gaussian_filter1d(gy, s / _GRID_STEP, mode="reflect") for s in levels]
        )
        hi = np.clip(np.searchsorted(levels, sig), 1, len(levels) - 1)
        lo = hi - 1
        f = np.clip((sig - levels[lo]) / (levels[hi] - levels[lo]), 0.0, 1.0)
        cols = np.arange(len(grid))
        sx = bx[lo, cols] * (1 - f) + bx[hi, cols] * f
        sy = by[lo, cols] * (1 - f) + by[hi, cols] * f
    return np.interp(sstar, grid, sx), np.interp(sstar, grid, sy)


@dataclass
class ValleyLine:
    """Valley line sampled at each node, plus per-node distances."""

    x: np.ndarray  # valley-line coordinates at each node (projected CRS)
    y: np.ndarray
    valley_dist: np.ndarray  # cumulative valley distance at each node (m)
    channel_dist: np.ndarray  # cumulative channel distance at each node (m)
    width_local: np.ndarray  # smoothed local width (m)
    sinuosity_local: np.ndarray  # channel / valley length over ±25 W
    sigma_widths: np.ndarray  # smoothing scale used at each node (widths)


def valley_line(
    x: np.ndarray,
    y: np.ndarray,
    width_m: np.ndarray,
    sigma_widths: float = 15.0,
    sinuosity_factor: float = 6.0,
) -> ValleyLine:
    """Derive a valley line from ordered channel nodes.

    1. Width-normalised arc length ``s* = ∫ ds / W`` (W = local width), so
       a fixed smoothing scale in s* grows with river size.
    2. Pass 1: Gaussian smoothing with σ = ``sigma_widths`` (in widths).
    3. Local sinuosity from pass 1 over ±25 W; pass 2 re-smooths with
       σ = max(``sigma_widths``, ``sinuosity_factor`` × S_local) widths, so
       highly sinuous reaches (long along-channel meander wavelengths) are
       smoothed enough to remove their bends.
    4. Valley distance at node *i* is the cumulative length of the smoothed
       line up to node *i*, which is monotonic by construction.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    seg = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
    s = np.cumsum(seg)
    W = local_width(width_m)
    sstar = np.cumsum(np.r_[0.0, seg[1:] / W[1:]])

    vx, vy = _smooth_xy(x, y, sstar, sigma_widths)
    v = np.cumsum(np.r_[0.0, np.hypot(np.diff(vx), np.diff(vy))])

    half = 25.0 * W
    s_hi = np.interp(np.minimum(v + half, v[-1]), v, s)
    s_lo = np.interp(np.maximum(v - half, 0.0), v, s)
    v_span = np.minimum(v + half, v[-1]) - np.maximum(v - half, 0.0)
    S = np.where(v_span > 0, (s_hi - s_lo) / np.maximum(v_span, 1e-9), 1.0)
    S = (
        pd.Series(np.clip(S, 1.0, None))
        .rolling(WIDTH_WINDOW_NODES, center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    sig = np.maximum(sigma_widths, sinuosity_factor * S)
    vx, vy = _smooth_xy(x, y, sstar, sig)
    v = np.cumsum(np.r_[0.0, np.hypot(np.diff(vx), np.diff(vy))])
    return ValleyLine(vx, vy, v, s, W, S, sig)


def reach_breaks(
    valley_dist: np.ndarray,
    width_at_node: np.ndarray,
    widths_per_reach: float,
    final_merge_fraction: float = 0.5,
) -> list[float]:
    """Down-valley distances (m) where reaches start, spaced ``k × W`` apart.

    Walks downstream: each reach's valley length is ``widths_per_reach``
    times the modelled width at its upstream end (interpolated between
    nodes). A final remainder shorter than ``final_merge_fraction`` of its
    nominal length is merged into the reach above. The first start is 0 and
    the last reach ends at the end of the valley line.
    """
    v = np.asarray(valley_dist, dtype=float)
    W = np.asarray(width_at_node, dtype=float)
    total = float(v[-1])
    starts = [0.0]
    while True:
        nominal = widths_per_reach * float(np.interp(starts[-1], v, W))
        nxt = starts[-1] + nominal
        if nxt >= total:
            break
        starts.append(nxt)
    if len(starts) >= 2:
        last = starts[-1]
        remainder = total - last
        nominal = widths_per_reach * float(np.interp(last, v, W))
        if remainder < final_merge_fraction * nominal:
            starts.pop()
    return starts
