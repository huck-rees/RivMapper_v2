"""Step 03 — extract raw annual DSWE observation counts from Earth Engine.

For each reach polygon and calendar year, counts — over every clear
Landsat scene that year — how often each pixel was observed and how often
it was classified as water at each DSWE level (Jones, 2019), and downloads
the counts on the river's fixed 30 m grid in the river's analysis CRS (see
docs/methods/conventions.md).

Raw files are an exact record of what Earth Engine returned: single year,
no compositing window, no thresholds. Choosing a DSWE level and percentile
happens later, in mask processing, from these counts.

Bands (uint8; 255 = outside the reach polygon):

* ``n_obs``   clear observations;
* ``n_wet_1`` … ``n_wet_4``  observations with DSWE class 1..k.

Outputs: ``{data_root}/{river}/masks/raw/{river}_r{NNN}_{year}.tif`` plus
``{river}_extract_masks_log.csv`` (one row per reach-year) and the run
record. Existing files made with the same settings are skipped, so an
interrupted run resumes where it stopped.

Downloads are tiled in ``chunk_px`` squares. A tile that Earth Engine
cannot compute (memory or time limit — typically a large reach in a year
with many overlapping Landsat 8 + 9 scenes) is split into quadrants and
retried, down to ``min_chunk_px``. Tiles sit on the same 30 m grid, so the
result is identical to a single download. Each tile only pulls the scenes
that touch it.

Earth Engine calls go through an ``EEBackend`` so everything else here can
be tested offline with a fake backend.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Protocol

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import ConfigError, Project, River
from .gee import COUNT_BANDS, NODATA, QA_MASK_BITS, Grid
from .provenance import recorded_parameters, write_run_record
from .results import FAILED, OK, STALE, StepResult, check_existing

log = logging.getLogger(__name__)

STEP = "extract_masks"
RES = 30.0
LOG_COLUMNS = [
    "ds_order",
    "year",
    "status",
    "n_scenes",
    "n_tiles",
    "clear_fraction",
    "n_obs_median",
    "n_obs_p10",
    "reason",
]


# ─────────────────────────────────────────────────────────────────────
# Grid helpers
# ─────────────────────────────────────────────────────────────────────
def river_origin(bounds: tuple[float, float, float, float], res: float = RES):
    """Top-left corner of the river grid, snapped to multiples of ``res``."""
    minx, _, _, maxy = bounds
    return math.floor(minx / res) * res, math.ceil(maxy / res) * res


def reach_grid(
    bounds: tuple[float, float, float, float],
    origin: tuple[float, float],
    crs: str,
    res: float = RES,
) -> Grid:
    """Smallest window of the river grid covering ``bounds``."""
    ox, oy = origin
    minx, miny, maxx, maxy = bounds
    c0 = math.floor((minx - ox) / res)
    r0 = math.floor((oy - maxy) / res)
    c1 = math.ceil((maxx - ox) / res)
    r1 = math.ceil((oy - miny) / res)
    return Grid(crs, ox + c0 * res, oy - r0 * res, res, c1 - c0, r1 - r0)


def chunks(grid: Grid, size: int):
    """Split ``grid`` into sub-grids of at most ``size`` × ``size`` pixels.

    Yields (row_offset, col_offset, sub_grid).
    """
    for r in range(0, grid.height, size):
        for c in range(0, grid.width, size):
            yield (
                r,
                c,
                Grid(
                    grid.crs,
                    grid.x0 + c * grid.res,
                    grid.y0 - r * grid.res,
                    grid.res,
                    min(size, grid.width - c),
                    min(size, grid.height - r),
                ),
            )


def split_quadrants(grid: Grid):
    """Split ``grid`` into up to four sub-grids (halving each side > 1 px).

    Yields (row_offset, col_offset, sub_grid).
    """
    rows = [(0, grid.height)]
    cols = [(0, grid.width)]
    if grid.height > 1:
        h = math.ceil(grid.height / 2)
        rows = [(0, h), (h, grid.height - h)]
    if grid.width > 1:
        w = math.ceil(grid.width / 2)
        cols = [(0, w), (w, grid.width - w)]
    for r, nh in rows:
        for c, nw in cols:
            yield (
                r,
                c,
                Grid(
                    grid.crs,
                    grid.x0 + c * grid.res,
                    grid.y0 - r * grid.res,
                    grid.res,
                    nw,
                    nh,
                ),
            )


# Earth Engine error messages that mean "this request was too big", which
# smaller tiles can fix (anything else — auth, bad geometry — is re-raised).
_RESOURCE_ERRORS = (
    "memory limit",
    "out of memory",
    "too large",
    "too many pixels",
    "request size",
    "exceeds",
    "timed out",
    "timeout",
    "deadline",
)


def is_resource_error(exc: BaseException) -> bool:
    """True if ``exc`` is an Earth Engine memory/size/time-limit error."""
    msg = str(exc).lower()
    return any(m in msg for m in _RESOURCE_ERRORS)


def fetch_adaptive(
    backend: EEBackend,
    comp,
    grid: Grid,
    min_px: int,
    stats: dict | None = None,
) -> dict[str, np.ndarray]:
    """Fetch ``grid``; on a resource error, split into quadrants and recurse.

    Gives up (re-raises) only when a tile no larger than ``min_px`` on its
    longest side still fails, or the error is not resource-related.
    ``stats`` (if given) counts ``tiles`` fetched and ``splits`` made.
    """
    stats = stats if stats is not None else {}
    try:
        arr = backend.fetch(comp, grid)
        stats["tiles"] = stats.get("tiles", 0) + 1
        return arr
    except Exception as e:  # classified below
        if not is_resource_error(e) or max(grid.width, grid.height) <= min_px:
            raise
        log.debug(
            "    %d x %d px tile hit an Earth Engine limit (%s); splitting",
            grid.width,
            grid.height,
            str(e)[:80],
        )
    stats["splits"] = stats.get("splits", 0) + 1
    out = {b: np.zeros((grid.height, grid.width), np.uint8) for b in COUNT_BANDS}
    for r, c, sub in split_quadrants(grid):
        part = fetch_adaptive(backend, comp, sub, min_px, stats)
        for b in COUNT_BANDS:
            out[b][r : r + sub.height, c : c + sub.width] = part[b]
    return out


def rasterize_inside(geom, grid: Grid) -> np.ndarray:
    """Boolean array: pixel centre inside ``geom``."""
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin

    transform = from_origin(grid.x0, grid.y0, grid.res, grid.res)
    return geometry_mask(
        [geom], out_shape=(grid.height, grid.width), transform=transform, invert=True
    )


# ─────────────────────────────────────────────────────────────────────
# Earth Engine backend (real one in gee.py; tests pass a fake)
# ─────────────────────────────────────────────────────────────────────
class EEBackend(Protocol):
    def counts(self, geom_lonlat, year: int): ...
    def n_scenes(self, comp) -> int: ...
    def fetch(self, comp, grid: Grid) -> dict[str, np.ndarray]: ...


class EarthEngine:
    """``EEBackend`` backed by the real Earth Engine API."""

    def __init__(self):
        from . import gee

        self.gee = gee

    def counts(self, geom_lonlat, year):
        geom = self.gee.ee_geometry(geom_lonlat)
        _, size = self.gee.dswe_counts(geom, year)
        return {"year": year, "size": size}

    def n_scenes(self, comp) -> int:
        """Scenes touching the whole reach polygon (reported in the log)."""
        return int(comp["size"].getInfo())

    def fetch(self, comp, grid):
        # Only the scenes that touch this tile: identical counts (other
        # scenes add zero here) for much less Earth Engine work.
        tile = self.gee.grid_geometry(grid)
        image, _ = self.gee.dswe_counts(tile, comp["year"])
        return self.gee.compute_pixels(image, grid, COUNT_BANDS)


# ─────────────────────────────────────────────────────────────────────
# Per reach-year extraction
# ─────────────────────────────────────────────────────────────────────
def _fingerprint(river: River, crs: str) -> dict:
    """Settings that determine a raw file's content."""
    return {
        "product": "dswe_counts_v1",
        "bands": COUNT_BANDS,
        "res_m": RES,
        "crs": crs,
        "qa_mask_bits": list(QA_MASK_BITS),
        "landsat": "LT05+LE07(1999-04-15..2003-05-30)+LC08+LC09 C2 L2 T1",
    }


_FINGERPRINT_KEYS = ("product", "bands", "res_m", "crs", "qa_mask_bits", "landsat")


def _fingerprint_from(params: dict) -> dict:
    return {k: params.get(k) for k in _FINGERPRINT_KEYS}


def _read_fingerprint(path: Path) -> dict | None:
    import rasterio

    try:
        with rasterio.open(path) as src:
            tag = src.tags().get("rivmapper_params")
        return json.loads(tag) if tag else None
    except Exception:  # noqa: BLE001 — unreadable/partial file → redo
        return None


def write_counts(path: Path, grid: Grid, bands: dict[str, np.ndarray], tags: dict):
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.tif")
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": len(COUNT_BANDS),
        "dtype": "uint8",
        "crs": grid.crs,
        "transform": from_origin(grid.x0, grid.y0, grid.res, grid.res),
        "nodata": NODATA,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    descriptions = {
        "n_obs": "clear Landsat observations in the year",
        "n_wet_1": "observations with DSWE class 1",
        "n_wet_2": "observations with DSWE class 1-2",
        "n_wet_3": "observations with DSWE class 1-3",
        "n_wet_4": "observations with DSWE class 1-4",
    }
    with rasterio.open(tmp, "w", **profile) as dst:
        for i, name in enumerate(COUNT_BANDS, start=1):
            dst.write(bands[name].astype(np.uint8), i)
            dst.set_band_description(i, f"{name}: {descriptions[name]}")
        dst.update_tags(
            **{k: (v if isinstance(v, str) else json.dumps(v)) for k, v in tags.items()}
        )
    tmp.replace(path)


def extract_reach_year(
    backend: EEBackend,
    river: River,
    reach: pd.Series,
    polygon_lonlat,
    grid: Grid,
    year: int,
    out_path: Path,
    fingerprint: dict,
) -> dict:
    """Download one reach-year of DSWE counts on ``grid`` and write it."""
    comp = backend.counts(polygon_lonlat, year)
    n_scenes = backend.n_scenes(comp)
    shape = (grid.height, grid.width)
    bands = {b: np.zeros(shape, dtype=np.uint8) for b in COUNT_BANDS}
    stats: dict = {}
    if n_scenes > 0:
        for r, c, sub in chunks(grid, river.masks.chunk_px):
            arr = fetch_adaptive(backend, comp, sub, river.masks.min_chunk_px, stats)
            for b in COUNT_BANDS:
                bands[b][r : r + sub.height, c : c + sub.width] = arr[b]
    inside = rasterize_inside(reach.geometry, grid)
    for b in COUNT_BANDS:
        bands[b][~inside] = NODATA

    n_obs = bands["n_obs"][inside]
    n_in = max(int(inside.sum()), 1)
    write_counts(
        out_path,
        grid,
        bands,
        {
            "rivmapper_params": fingerprint,
            "river": river.name,
            "ds_order": str(int(reach["ds_order"])),
            "year": str(year),
            "n_scenes": str(n_scenes),
        },
    )
    return {
        "ds_order": int(reach["ds_order"]),
        "year": year,
        "n_scenes": n_scenes,
        "n_tiles": stats.get("tiles", 0),
        "clear_fraction": round(float((n_obs > 0).sum() / n_in), 4),
        "n_obs_median": float(np.median(n_obs)) if n_obs.size else 0.0,
        "n_obs_p10": float(np.percentile(n_obs, 10)) if n_obs.size else 0.0,
        "status": "ok" if n_scenes > 0 else "no_imagery",
    }


# ─────────────────────────────────────────────────────────────────────
# Step entry points
# ─────────────────────────────────────────────────────────────────────
def _parameters(river: River, crs: str) -> dict:
    m = river.masks
    return {
        **_fingerprint(river, crs),
        "year_start": m.year_start,
        "year_end": m.year_end,
        "reach_start": m.reach_start,
        "reach_end": m.reach_end,
    }


def extract_masks(
    river: River,
    backend: EEBackend,
    config_path: Path | None = None,
) -> StepResult:
    """Run step ``extract_masks`` for one river."""
    river.require("masks", "year_start", "year_end")
    p = river.paths
    if not p.reach_polygons.exists():
        return StepResult(
            river.name,
            STEP,
            FAILED,
            f"reach polygons not found at {p.reach_polygons} — run the reaches step",
        )
    polys = gpd.read_file(p.reach_polygons)
    if polys.crs is None or polys.crs.is_geographic:
        raise ConfigError([f"{river.name}: reach polygons must be in a projected CRS"])
    epsg = polys.crs.to_epsg()
    crs = f"EPSG:{epsg}" if epsg else polys.crs.to_wkt()

    m = river.masks
    lo = m.reach_start or 1
    hi = m.reach_end or int(polys["ds_order"].max())
    reaches = polys[polys["ds_order"].between(lo, hi)].sort_values("ds_order")
    years = list(range(m.year_start, m.year_end + 1))
    outputs = [p.raw_mask(int(o), y) for o in reaches["ds_order"] for y in years]

    params = _parameters(river, crs)
    existing = check_existing(
        river.name, STEP, outputs, p.run_record(STEP), params, river.overwrite
    )
    if existing:
        return existing
    # Partly done with *different* settings: don't silently mix old and new.
    previous = recorded_parameters(p.run_record(STEP))
    if (
        not river.overwrite
        and previous is not None
        and _fingerprint_from(previous) != _fingerprint(river, crs)
        and any(o.exists() for o in outputs)
    ):
        return StepResult(
            river.name,
            STEP,
            STALE,
            "existing masks were made with different settings; use --overwrite "
            "to rebuild them",
        )

    origin = river_origin(tuple(polys.total_bounds))
    fp = _fingerprint(river, crs)
    polys_ll = reaches.to_crs(4326)
    rows = []
    n_done = n_skip = 0
    for (_, reach), geom_ll in zip(reaches.iterrows(), polys_ll.geometry):
        grid = reach_grid(tuple(reach.geometry.bounds), origin, crs)
        geom_ll = geom_ll.simplify(0.0003)  # ~30 m; keeps EE requests small
        statuses = []
        for y in years:
            out = p.raw_mask(int(reach["ds_order"]), y)
            if out.exists() and not river.overwrite and _read_fingerprint(out) == fp:
                n_skip += 1
                statuses.append("skip")
                continue
            try:
                row = extract_reach_year(
                    backend, river, reach, geom_ll, grid, y, out, fp
                )
            except Exception as e:  # noqa: BLE001 — record and continue
                log.warning("  r%03d %d failed: %s", reach["ds_order"], y, e)
                row = {
                    "ds_order": int(reach["ds_order"]),
                    "year": y,
                    "status": "failed",
                    "reason": str(e)[:300],
                }
            rows.append(row)
            statuses.append(row["status"])
            n_done += 1
        counts = pd.Series(statuses).value_counts().to_dict()
        log.info(
            "  reach %d (%d x %d px): %s",
            reach["ds_order"],
            grid.width,
            grid.height,
            ", ".join(f"{v} {k}" for k, v in counts.items()),
        )

    log_path = p.masks_log
    new = pd.DataFrame(rows, columns=LOG_COLUMNS)
    if log_path.exists() and not new.empty:
        # Columns from older versions of this step are dropped.
        old = pd.read_csv(log_path).reindex(columns=LOG_COLUMNS)
        keys = set(zip(new["ds_order"], new["year"]))
        old = old[[k not in keys for k in zip(old["ds_order"], old["year"])]]
        new = pd.concat([old, new], ignore_index=True)
    elif log_path.exists():
        new = pd.read_csv(log_path).reindex(columns=LOG_COLUMNS)
    if not new.empty:
        new.sort_values(["ds_order", "year"]).to_csv(log_path, index=False)

    status_counts = new["status"].value_counts().to_dict() if not new.empty else {}
    missing = [o for o in outputs if not o.exists()]
    summary = {
        "analysis_crs": crs,
        "grid_origin": list(origin),
        "n_reaches": int(len(reaches)),
        "years": [m.year_start, m.year_end],
        "downloaded_this_run": n_done,
        "skipped_existing": n_skip,
        "status_counts": status_counts,
        "missing_outputs": len(missing),
    }
    failed = status_counts.get("failed", 0)
    write_run_record(
        p.run_record(STEP),
        step=STEP,
        river=river.name,
        config_path=config_path,
        parameters=params,
        inputs={"reach_polygons": p.reach_polygons},
        outputs={"raw_masks": p.raw_masks_dir, "log": log_path},
        summary=summary,
    )
    msg = (
        f"{len(outputs) - len(missing)}/{len(outputs)} masks "
        f"({n_done} downloaded, {n_skip} already done"
        + (
            f", {status_counts.get('no_imagery', 0)} no imagery"
            if status_counts.get("no_imagery")
            else ""
        )
        + (f", {failed} failed" if failed else "")
        + ")"
    )
    return StepResult(river.name, STEP, FAILED if failed else OK, msg, summary)


def run(project: Project, rivers: list[River], backend: EEBackend | None = None):
    """Run step ``extract_masks`` for ``rivers``."""
    if backend is None:
        from . import gee

        try:
            gee.initialize(project.settings.gee)
        except gee.GeeError as e:
            return [StepResult(r.name, STEP, FAILED, str(e)) for r in rivers]
        backend = EarthEngine()
    results = []
    for r in rivers:
        log.info("%s", r.name)
        try:
            res = extract_masks(r, backend, project.config_path)
        except Exception as e:  # noqa: BLE001
            log.debug("%s: error", r.name, exc_info=True)
            res = StepResult(r.name, STEP, FAILED, f"{type(e).__name__}: {e}")
        results.append(res)
    return results
