"""Reach windows on the river grid, and reading raw counts into them.

Mask processing needs spatial context beyond one reach (connectivity,
through-flow), but loading a whole river at once does not scale. Instead
each reach is processed inside a *window*: the reach plus its upstream and
downstream neighbours, on the river's fixed 30 m grid. Results are kept
only for the window's centre reach, so every pixel is decided exactly once
and neighbours only provide context.

Water may enter or leave a window only across its *cut lines*: the shared
boundaries with the reaches just outside it (the nearest-valley partition
lines, which run across the valley). At the river's ends there is no cut
line; the SWORD endpoints are the terminals instead.

Everything here works from the reach polygons and the raw files'
conventions (see docs/methods/conventions.md): one grid per river, origin
at the top-left of the union of reach polygons snapped to 30 m, each reach
raster the smallest window of that grid covering its polygon, 255 outside
the polygon.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry.base import BaseGeometry

from ..gee import COUNT_BANDS, NODATA, Grid
from ..masks import RES, reach_grid, river_origin
from ..paths import RiverPaths

log = logging.getLogger(__name__)


def crs_string(gdf: gpd.GeoDataFrame) -> str:
    """The CRS string extraction used for this river ("EPSG:n" or WKT)."""
    if gdf.crs is None or gdf.crs.is_geographic:
        raise ValueError("reach polygons must be in a projected CRS")
    epsg = gdf.crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else gdf.crs.to_wkt()


def offset(inner: Grid, outer: Grid) -> tuple[int, int]:
    """(row, col) of ``inner``'s top-left pixel within ``outer``.

    Both grids must share the river grid (same resolution, origin on the
    same 30 m lattice).
    """
    r = (outer.y0 - inner.y0) / outer.res
    c = (inner.x0 - outer.x0) / outer.res
    if abs(r - round(r)) > 1e-6 or abs(c - round(c)) > 1e-6:
        raise ValueError("grids are not aligned to the same pixel lattice")
    return round(r), round(c)


def union_grid(grids: Iterable[Grid]) -> Grid:
    """Smallest grid on the same lattice covering all ``grids``."""
    grids = list(grids)
    g0 = grids[0]
    x0 = min(g.x0 for g in grids)
    y0 = max(g.y0 for g in grids)
    x1 = max(g.x0 + g.width * g.res for g in grids)
    y1 = min(g.y0 - g.height * g.res for g in grids)
    return Grid(
        g0.crs,
        x0,
        y0,
        g0.res,
        round((x1 - x0) / g0.res),
        round((y0 - y1) / g0.res),
    )


@dataclass(frozen=True)
class Window:
    """A reach plus its neighbours, on one grid.

    Attributes:
        center: ``ds_order`` of the reach whose results are kept.
        members: ``ds_order`` of every reach loaded, upstream first.
        grid: Grid covering all members.
        upstream_cut: Line where water enters the window, or None when the
            window starts at the river's upstream end.
        downstream_cut: Line where water leaves, or None at the river mouth.
    """

    center: int
    members: tuple[int, ...]
    grid: Grid
    upstream_cut: BaseGeometry | None
    downstream_cut: BaseGeometry | None


class RiverLayout:
    """Reach polygons, their grids and windows for one river."""

    def __init__(self, polygons: gpd.GeoDataFrame, res: float = RES):
        if "ds_order" not in polygons.columns:
            raise ValueError("reach polygons need a ds_order column")
        self.polygons = polygons.sort_values("ds_order").set_index("ds_order")
        self.crs = crs_string(polygons)
        self.res = res
        self.origin = river_origin(tuple(polygons.total_bounds), res)
        self.grids: dict[int, Grid] = {
            int(o): reach_grid(tuple(geom.bounds), self.origin, self.crs, res)
            for o, geom in self.polygons.geometry.items()
        }
        self.ds_orders: list[int] = sorted(self.grids)

    @classmethod
    def from_paths(cls, paths: RiverPaths) -> RiverLayout:
        if not paths.reach_polygons.exists():
            raise FileNotFoundError(
                f"reach polygons not found at {paths.reach_polygons} — run the "
                f"reaches step"
            )
        return cls(gpd.read_file(paths.reach_polygons))

    # ── geometry ────────────────────────────────────────────────────
    def width_m(self, ds_order: int) -> float:
        """Reference channel width W of a reach (``width_ref``), in metres."""
        if "width_ref" not in self.polygons.columns:
            raise ValueError(
                "reach polygons have no width_ref column — rebuild reaches with "
                "length_method: valley_widths"
            )
        return float(self.polygons.loc[ds_order, "width_ref"])

    def width_px(self, ds_order: int) -> float:
        return self.width_m(ds_order) / self.res

    def cut_line(self, a: int, b: int) -> BaseGeometry | None:
        """Shared boundary of reaches ``a`` and ``b`` (None if they don't touch)."""
        if a not in self.grids or b not in self.grids:
            return None
        line = self.polygons.geometry[a].boundary.intersection(
            self.polygons.geometry[b].boundary
        )
        return None if line.is_empty else line

    # ── windows ─────────────────────────────────────────────────────
    def window(self, center: int, halo: int = 1) -> Window:
        """Window centred on reach ``center`` with ``halo`` neighbours each side."""
        i = self.ds_orders.index(center)
        members = tuple(self.ds_orders[max(0, i - halo) : i + halo + 1])
        first, last = members[0], members[-1]
        j_first, j_last = self.ds_orders.index(first), self.ds_orders.index(last)
        up = self.cut_line(self.ds_orders[j_first - 1], first) if j_first > 0 else None
        down = (
            self.cut_line(last, self.ds_orders[j_last + 1])
            if j_last + 1 < len(self.ds_orders)
            else None
        )
        return Window(
            center=center,
            members=members,
            grid=union_grid(self.grids[o] for o in members),
            upstream_cut=up,
            downstream_cut=down,
        )

    def windows(self, halo: int = 1) -> list[Window]:
        """One window per reach, upstream to downstream."""
        return [self.window(o, halo) for o in self.ds_orders]

    def owner(self, grid: Grid, members: Sequence[int] | None = None) -> np.ndarray:
        """int16 raster: ``ds_order`` of the reach owning each pixel, 0 = none.

        A pixel belongs to a reach when its centre lies inside the reach
        polygon — the same rule extraction used to set 255 outside it.
        """
        from rasterio.features import rasterize
        from rasterio.transform import from_origin

        members = list(members) if members is not None else self.ds_orders
        shapes = [(self.polygons.geometry[o], int(o)) for o in members]
        return rasterize(
            shapes,
            out_shape=(grid.height, grid.width),
            transform=from_origin(grid.x0, grid.y0, grid.res, grid.res),
            fill=0,
            dtype="int16",
        )

    def rasterize_line(
        self, geom: BaseGeometry | None, grid: Grid, width_px: float = 1.0
    ) -> np.ndarray:
        """Boolean raster of pixels whose centre is within ``width_px`` + ½ pixel
        of a line.

        Cut lines often lie exactly on a window's outer edge, where plain
        line rasterisation would miss them; the buffer catches the pixels
        just inside.
        """
        from rasterio.features import rasterize
        from rasterio.transform import from_origin

        shape = (grid.height, grid.width)
        if geom is None or geom.is_empty:
            return np.zeros(shape, bool)
        zone = geom.buffer((width_px + 0.5) * grid.res)
        return rasterize(
            [(zone, 1)],
            out_shape=shape,
            transform=from_origin(grid.x0, grid.y0, grid.res, grid.res),
            fill=0,
            dtype="uint8",
        ).astype(bool)


# ─────────────────────────────────────────────────────────────────────
# Reading raw counts
# ─────────────────────────────────────────────────────────────────────
def _band_index(src) -> dict[str, int]:
    """Band name → 1-based index, from band descriptions ("n_obs: …")."""
    names = [(d or "").split(":")[0].strip() for d in src.descriptions]
    if all(names):
        return {n: i + 1 for i, n in enumerate(names)}
    return {n: i + 1 for i, n in enumerate(COUNT_BANDS[: src.count])}


def read_raw(path: Path, bands: Sequence[str]) -> tuple[Grid, dict[str, np.ndarray]]:
    """Read named count bands from one raw reach-year file."""
    import rasterio

    with rasterio.open(path) as src:
        index = _band_index(src)
        missing = [b for b in bands if b not in index]
        if missing:
            raise ValueError(f"{path.name}: no band(s) {missing}")
        t = src.transform
        crs = src.crs
        epsg = crs.to_epsg() if crs else None
        grid = Grid(
            f"EPSG:{epsg}" if epsg else (crs.to_wkt() if crs else ""),
            t.c,
            t.f,
            t.a,
            src.width,
            src.height,
        )
        return grid, {b: src.read(index[b]) for b in bands}


def read_reach_stack(
    paths: RiverPaths,
    ds_order: int,
    years: Sequence[int],
    bands: Sequence[str] = ("n_obs", "n_wet_2", "n_wet_3"),
) -> tuple[Grid, dict[str, np.ndarray]]:
    """All years of one reach: {band: uint8 array (years, rows, cols)}.

    Raises:
        FileNotFoundError: listing every missing reach-year file.
    """
    files = [paths.raw_mask(ds_order, y) for y in years]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} raw file(s) missing for reach {ds_order}, e.g. "
            f"{missing[:3]} — run extract_masks"
        )
    grid, first = read_raw(files[0], bands)
    stack = {
        b: np.empty((len(years), grid.height, grid.width), np.uint8) for b in bands
    }
    for b in bands:
        stack[b][0] = first[b]
    for i, f in enumerate(files[1:], start=1):
        g, arr = read_raw(f, bands)
        if (g.width, g.height, g.x0, g.y0) != (
            grid.width,
            grid.height,
            grid.x0,
            grid.y0,
        ):
            raise ValueError(f"{f.name}: grid differs from {files[0].name}")
        for b in bands:
            stack[b][i] = arr[b]
    return grid, stack


def read_window_year(
    paths: RiverPaths,
    window: Window,
    year: int,
    bands: Sequence[str] = ("n_obs", "n_wet_2", "n_wet_3"),
) -> dict[str, np.ndarray]:
    """One year of every member reach, mosaicked onto the window grid.

    Pixels outside all member polygons are 255. Raises ValueError if two
    reach files claim the same pixel (overlapping reach polygons).
    """
    shape = (window.grid.height, window.grid.width)
    out = {b: np.full(shape, NODATA, np.uint8) for b in bands}
    claimed = np.zeros(shape, bool)
    for o in window.members:
        g, arr = read_raw(paths.raw_mask(o, year), bands)
        r, c = offset(g, window.grid)
        inside = arr[bands[0]] != NODATA
        sub = (slice(r, r + g.height), slice(c, c + g.width))
        clash = claimed[sub] & inside
        if clash.any():
            raise ValueError(
                f"reach {o} overlaps another reach in window {window.center} "
                f"({int(clash.sum())} px, year {year})"
            )
        claimed[sub] |= inside
        for b in bands:
            out[b][sub][inside] = arr[b][inside]
    return out
