"""Synthetic rivers with a known channel every year.

Used as the unit-test fixture for mask processing and as its main
validation tool: real data never tells us the true channel, a synthetic
river does. A river is built in three layers:

1. **Truth.** A meandering channel of width W that migrates downstream
   every year, plus optional scenario events (cutoff, avulsion, braided
   threads). ``truth`` is the near-bankfull channel; ``truth_core`` the
   low-flow channel (``truth`` minus its high-stage margins).
2. **Distractors.** Water that is *not* channel and that processing must
   remove: an oxbow left by the cutoff, the abandoned course after an
   avulsion (wet for a few years), a floodplain lake, overbank flooding in
   one year, a tributary.
3. **Observation.** Landsat-like sampling of all water: per year a number
   of scenes with spatially coherent cloud, each scene at a random stage,
   each clear pixel classified into DSWE classes 0–4 with realistic
   confusion (mixed edge pixels, false water on land, occasional "bad
   scenes" that wet a large block of floodplain). Clear-look counts follow
   the Ok Tedi's year-by-year medians by default, including empty years.

The result is written in exactly the raw-file format (reach polygons,
reach lines and ``masks/raw/{river}_r{NNN}_{year}.tif``), so the
processing step reads it like real data.

Example::

    river = make_river(scenarios=("cutoff", "lake"), seed=1)
    cfg = river.write_project(tmp_path)        # rivmapper.yaml + data
    truth_2000 = river.truth[river.years.index(2000)]
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from shapely.geometry import LineString, box

from ..gee import COUNT_BANDS, COUNT_MAX, NODATA, Grid

SCENARIOS = ("cutoff", "avulsion", "braided", "lake", "flood", "tributary")

# Median clear looks per pixel by year on the Ok Tedi–Fly (20 reaches,
# raw extraction of Oct 2026). 0 = no imagery.
OKTEDI_MEDIAN_LOOKS: dict[int, int] = {
    1984: 0, 1985: 0, 1986: 0, 1987: 3, 1988: 5, 1989: 3, 1990: 4, 1991: 4,
    1992: 3, 1993: 5, 1994: 3, 1995: 3, 1996: 4, 1997: 5, 1998: 5, 1999: 6,
    2000: 6, 2001: 6, 2002: 5, 2003: 3, 2004: 5, 2005: 3, 2006: 5, 2007: 2,
    2008: 4, 2009: 3, 2010: 1, 2011: 1, 2012: 0, 2013: 1, 2014: 4, 2015: 6,
    2016: 3, 2017: 3, 2018: 3, 2019: 3, 2020: 5, 2021: 5, 2022: 11, 2023: 6,
    2024: 8, 2025: 9, 2026: 5,
}  # fmt: skip

# Per-look DSWE class probabilities [0, 1, 2, 3, 4].
WATER_CLASSES = np.array([0.02, 0.70, 0.20, 0.05, 0.03])
EDGE_CLASSES = np.array([0.25, 0.15, 0.25, 0.15, 0.20])  # mixed water/land pixel
LAND_CLASSES = np.array([0.965, 0.001, 0.004, 0.010, 0.020])
BAD_SCENE_LAND = np.array([0.30, 0.05, 0.45, 0.15, 0.05])  # haze / thin cloud


@dataclass
class ObservationModel:
    """How a synthetic river is observed by Landsat.

    Attributes:
        median_looks: Target median clear looks per pixel, by year.
        clear_prob: Mean chance a pixel is clear in one scene.
        cloud_scale_px: Spatial scale of cloud patches (pixels).
        high_stage_prob: Chance a scene catches high stage (margins wet).
        bad_scene_prob: Chance a scene has a haze block of false water.
        bad_block_fraction: Share of the scene a haze block covers.
    """

    median_looks: dict[int, int] = field(
        default_factory=lambda: dict(OKTEDI_MEDIAN_LOOKS)
    )
    clear_prob: float = 0.45
    cloud_scale_px: float = 25.0
    high_stage_prob: float = 0.5
    bad_scene_prob: float = 0.08
    bad_block_fraction: float = 0.3


@dataclass
class SyntheticRiver:
    """A synthetic river: truth, counts and enough geometry to write a project.

    Arrays are (years, rows, cols) on ``grid``; rows run north to south,
    columns west (upstream) to east (downstream).
    """

    years: list[int]
    grid: Grid
    width_px: float
    truth: np.ndarray
    truth_core: np.ndarray
    water: np.ndarray
    counts: dict[str, np.ndarray]
    centerline: LineString
    n_reaches: int = 3
    events: dict[str, int] = field(default_factory=dict)

    # ── geometry ────────────────────────────────────────────────────
    def reach_polygons(self) -> gpd.GeoDataFrame:
        """Equal reaches split across the valley (columns), one pixel inset."""
        g = self.grid
        # Boundaries on pixel edges, so every pixel centre is in one reach.
        edges = np.round(np.linspace(0, g.width, self.n_reaches + 1))
        xs = g.x0 + edges * g.res
        y_top, y_bot = g.y0 - g.res, g.y0 - (g.height - 1) * g.res
        polys = [box(xs[i], y_bot, xs[i + 1], y_top) for i in range(self.n_reaches)]
        return gpd.GeoDataFrame(
            {
                "ds_order": list(range(1, self.n_reaches + 1)),
                "width_ref": [self.width_px * g.res] * self.n_reaches,
            },
            geometry=polys,
            crs=g.crs,
        )

    def reach_lines(self) -> gpd.GeoDataFrame:
        """The first year's centreline cut by reach (stands in for SWORD)."""
        polys = self.reach_polygons()
        return gpd.GeoDataFrame(
            {"ds_order": polys["ds_order"]},
            geometry=[self.centerline.intersection(p) for p in polys.geometry],
            crs=self.grid.crs,
        )

    # ── writing ─────────────────────────────────────────────────────
    def write_project(self, root: Path, river: str = "Synth") -> Path:
        """Write rivmapper.yaml, rivers.csv, reach files and raw counts.

        Returns the path of rivmapper.yaml. Data go to ``root/output``.
        """
        from ..masks import reach_grid, river_origin, write_counts
        from ..paths import RiverPaths
        from .windows import offset

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        paths = RiverPaths(root / "output", river)
        paths.reaches_dir.mkdir(parents=True, exist_ok=True)
        polys = self.reach_polygons()
        polys.to_file(paths.reach_polygons, driver="GPKG")
        self.reach_lines().to_file(paths.reach_lines, driver="GPKG")

        origin = river_origin(tuple(polys.total_bounds), self.grid.res)
        tags = {
            "rivmapper_params": {"product": "dswe_counts_v1", "synthetic": True},
            "river": river,
        }
        for o, geom in zip(polys["ds_order"], polys.geometry, strict=True):
            g = reach_grid(tuple(geom.bounds), origin, self.grid.crs, self.grid.res)
            r, c = offset(g, self.grid)
            inside = _inside(geom, g)
            for i, y in enumerate(self.years):
                bands = {}
                for b in COUNT_BANDS:
                    a = self.counts[b][i, r : r + g.height, c : c + g.width].copy()
                    a[~inside] = NODATA
                    bands[b] = a
                write_counts(
                    paths.raw_mask(int(o), y),
                    g,
                    bands,
                    {**tags, "ds_order": str(int(o)), "year": str(y)},
                )

        (root / "rivers.csv").write_text(
            "river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id\n"
            f"{river},oc,51000000011,51000000021\n",
            encoding="utf-8",
        )
        cfg = root / "rivmapper.yaml"
        cfg.write_text(
            "data_root: ./output\nrivers_table: rivers.csv\n"
            "reaches:\n  buffer_width: 0.15\n"
            f"masks:\n  year_start: {self.years[0]}\n  year_end: {self.years[-1]}\n",
            encoding="utf-8",
        )
        return cfg


def _inside(geom, grid: Grid) -> np.ndarray:
    from ..masks import rasterize_inside

    return rasterize_inside(geom, grid)


# ─────────────────────────────────────────────────────────────────────
# Building a river
# ─────────────────────────────────────────────────────────────────────
def _channel(ys_of_x: np.ndarray, shape: tuple[int, int], half_width: float):
    """Boolean channel around a centreline given as row position per column."""
    h, w = shape
    xs = np.arange(0, w, 0.25)
    yc = np.interp(xs, np.arange(w), ys_of_x)
    line = np.zeros(shape, bool)
    rr = np.clip(np.round(yc).astype(int), 0, h - 1)
    cc = np.clip(np.round(xs).astype(int), 0, w - 1)
    line[rr, cc] = True
    return distance_transform_edt(~line) <= half_width


def _smooth_field(rng, shape, scale):
    """Gaussian random field, standardised. Built at reduced resolution for
    large scales (much faster; the field is smooth anyway)."""
    k = max(1, int(scale // 4))
    small = (max(2, -(-shape[0] // k)), max(2, -(-shape[1] // k)))
    f = gaussian_filter(rng.standard_normal(small), scale / k)
    if k > 1:
        f = np.kron(f, np.ones((k, k)))[: shape[0], : shape[1]]
        f = gaussian_filter(f, k / 2)
    return (f - f.mean()) / (f.std() + 1e-12)


def make_river(
    scenarios: Sequence[str] = (),
    shape: tuple[int, int] = (300, 450),
    width_px: float = 8.0,
    years: Sequence[int] | None = None,
    seed: int = 0,
    obs: ObservationModel | None = None,
    crs: str = "EPSG:32754",
    origin: tuple[float, float] = (500_010.0, 9_100_020.0),
    n_reaches: int = 3,
) -> SyntheticRiver:
    """Build a synthetic river.

    Args:
        scenarios: Any of :data:`SCENARIOS`. Meander migration is always on.
        shape: Grid (rows, cols); flow runs west to east.
        width_px: Channel width W in pixels (3 = barely resolved).
        years: Calendar years; default 1984–2026.
        seed: Random seed; the same seed always gives the same river.
        obs: Observation model; default mimics the Ok Tedi.
        crs, origin: Where the grid sits (origin = top-left, multiples of 30 m).
        n_reaches: Reaches the valley is split into when written.
    """
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        raise ValueError(
            f"unknown scenario(s) {sorted(unknown)}; choose from {SCENARIOS}"
        )
    obs = obs or ObservationModel()
    years = list(years) if years is not None else list(range(1984, 2027))
    rng = np.random.default_rng(seed)
    h, w = shape
    T = len(years)
    hw = width_px / 2.0
    x = np.arange(w, dtype=float)
    amp, wavelength = 0.18 * h, 0.4 * w
    # Downstream migration: meanders translate ~0.6 px/yr, lateral ~1 px/yr.
    phase_rate = 2 * np.pi * 0.6 / wavelength
    events: dict[str, int] = {}

    truth = np.zeros((T, h, w), bool)
    other = np.zeros((T, h, w), bool)  # non-channel water
    centre_y = np.zeros((T, w))
    for t in range(T):
        centre_y[t] = h / 2 + amp * np.sin(2 * np.pi * x / wavelength + phase_rate * t)

    # Avulsion: from t_a, the downstream third jumps 6 W north, eased in.
    if "avulsion" in scenarios:
        t_a = T // 2
        events["avulsion"] = years[t_a]
        xa = int(0.6 * w)
        ramp = np.clip((x - xa) / (0.1 * w), 0, 1)
        for t in range(t_a, T):
            centre_y[t] = centre_y[t] - 6 * width_px * ramp

    for t in range(T):
        truth[t] = _channel(centre_y[t], shape, hw)

    if "avulsion" in scenarios:
        t_a = years.index(events["avulsion"])
        old = _channel(centre_y[t_a - 1], shape, hw) & ~truth[t_a]
        for t in range(t_a, min(T, t_a + 3)):  # abandoned course stays wet 3 years
            other[t] |= old & ~truth[t]

    # Cutoff: from t_c the loop around the middle crest is cut by a straight
    # chute; the loop stays as an oxbow lake (its ends silted, 1.5 W).
    if "cutoff" in scenarios:
        t_c = int(0.35 * T)
        events["cutoff"] = years[t_c]
        crest = int(wavelength * (0.25 - phase_rate * t_c / (2 * np.pi)) % wavelength)
        crest += int(wavelength) * int((w / 2) // wavelength)
        x1, x2 = (
            max(1, crest - int(0.3 * wavelength)),
            min(w - 2, crest + int(0.3 * wavelength)),
        )
        for t in range(t_c, T):
            ys = centre_y[t].copy()
            ys[x1:x2] = np.interp(x[x1:x2], [x1, x2], [ys[x1], ys[x2]])
            new = _channel(ys, shape, hw)
            loop = truth[t] & ~new
            loop &= distance_transform_edt(~new) > 1.5 * width_px
            truth[t] = new
            if t == t_c:
                oxbow = loop
            other[t] |= oxbow & ~new

    # Braided reach: a second thread in the middle third, switching on and
    # off every 6 years.
    if "braided" in scenarios:
        xs0, xs1 = int(w / 3), int(2 * w / 3)
        for t in range(T):
            if (t // 6) % 2 == 0:
                ys = centre_y[t] + 3 * width_px * np.sin(
                    np.pi * (x - xs0) / (xs1 - xs0)
                )
                ys[:xs0] = centre_y[t][:xs0]
                ys[xs1:] = centre_y[t][xs1:]
                truth[t] |= (
                    _channel(ys, shape, hw * 0.75)
                    & (x >= xs0)[None, :]
                    & (x <= xs1)[None, :]
                )
        events["braided"] = years[0]

    # Floodplain lake: a permanent ellipse well away from the channel.
    if "lake" in scenarios:
        yy, xx = np.mgrid[0:h, 0:w]
        far_row = (
            int(0.1 * h) if centre_y[:, int(0.75 * w)].mean() > h / 2 else int(0.9 * h)
        )
        lake = ((xx - 0.75 * w) / (4 * width_px)) ** 2 + (
            (yy - far_row) / (2 * width_px)
        ) ** 2 <= 1
        other |= lake[None] & ~truth
        events["lake"] = years[0]

    # Flood: one year with overbank water spreading 5 W from the channel in
    # the downstream half.
    if "flood" in scenarios:
        t_f = int(0.6 * T)
        events["flood"] = years[t_f]
        spread = distance_transform_edt(~truth[t_f]) <= 5 * width_px
        blob = spread & (x > w / 2)[None, :] & (_smooth_field(rng, shape, 6) > 0.3)
        other[t_f] |= blob & ~truth[t_f]

    # Tributary: a narrow channel entering from the north edge at mid-valley.
    if "tributary" in scenarios:
        xt = int(0.45 * w)
        for t in range(T):
            ys = np.arange(0, int(centre_y[t][xt]))
            trib = np.zeros(shape, bool)
            trib[ys, xt] = True
            other[t] |= (distance_transform_edt(~trib) <= max(1.0, hw / 2)) & ~truth[t]
        events["tributary"] = years[0]

    margin = max(1, round(0.15 * width_px)) if width_px >= 5 else 0
    truth_core = np.stack(
        [binary_erosion(tt, iterations=margin) if margin else tt for tt in truth]
    )
    water = truth | other
    counts = _observe(rng, years, truth, truth_core, other, obs)

    grid = Grid(crs, origin[0], origin[1], 30.0, w, h)
    xs_geo = grid.x0 + (x + 0.5) * grid.res
    ys_geo = grid.y0 - (centre_y[0] + 0.5) * grid.res
    centerline = LineString(np.column_stack([xs_geo, ys_geo]))
    return SyntheticRiver(
        years=years,
        grid=grid,
        width_px=width_px,
        truth=truth,
        truth_core=truth_core,
        water=water,
        counts=counts,
        centerline=centerline,
        n_reaches=n_reaches,
        events=events,
    )


def _observe(rng, years, truth, truth_core, other, obs: ObservationModel):
    """Simulate annual DSWE counts for all water (channel and distractors)."""
    T, h, w = truth.shape
    counts = {b: np.zeros((T, h, w), np.uint16) for b in COUNT_BANDS}
    for t, year in enumerate(years):
        looks = obs.median_looks.get(year, 0)
        n_scenes = round(looks / obs.clear_prob) if looks else 0
        edge_hi = truth[t] & ~binary_erosion(truth[t])
        edge_lo = truth_core[t] & ~binary_erosion(truth_core[t])
        wet_hi, wet_lo = truth[t] | other[t], truth_core[t] | other[t]
        for _ in range(n_scenes):
            cloud_free = obs.clear_prob + 0.25 * rng.standard_normal()
            cloud_free = float(np.clip(cloud_free, 0.05, 0.95))
            f = _smooth_field(rng, (h, w), obs.cloud_scale_px)
            clear = f <= np.quantile(f, cloud_free)
            high = rng.random() < obs.high_stage_prob
            wet, edge = (wet_hi, edge_hi) if high else (wet_lo, edge_lo)
            probs = np.where(wet[..., None], WATER_CLASSES, LAND_CLASSES)
            probs = np.where((wet & edge)[..., None], EDGE_CLASSES, probs)
            if rng.random() < obs.bad_scene_prob:
                g = _smooth_field(rng, (h, w), 2 * obs.cloud_scale_px)
                block = g >= np.quantile(g, 1 - obs.bad_block_fraction)
                probs = np.where((block & ~wet)[..., None], BAD_SCENE_LAND, probs)
            u = rng.random((h, w, 1))
            cls = (u > np.cumsum(probs, axis=-1)).sum(-1)
            counts["n_obs"][t] += clear
            for k in (1, 2, 3, 4):
                counts[f"n_wet_{k}"][t] += clear & (cls >= 1) & (cls <= k)
    return {b: np.minimum(a, COUNT_MAX).astype(np.uint8) for b, a in counts.items()}
