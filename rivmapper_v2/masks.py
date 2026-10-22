"""Step 03 — extract raw Landsat DSWE water masks from Earth Engine.

For each reach polygon and year, builds a DSWE water composite (Jones,
2019) in Earth Engine and downloads it on the river's fixed 30 m grid in
the river's analysis CRS (see docs/methods/conventions.md):

* **Grid:** origin snapped to multiples of 30 m in the analysis CRS (the CRS
  of the reach polygons), so every reach and year aligns pixel for pixel.
* **Values:** band 1 ``water`` = 0 dry, 1 water, 255 no data (outside the
  reach polygon or no clear observation); band 2 ``n_obs`` = number of
  clear Landsat observations in the composite (0–254).
* **Composite window:** the year alone first; if too little of the polygon
  has a clear observation, or detected water is far below the expected
  wetted area (``width_ref × reach_len``), the window widens to ±1, then
  ±2 years (``masks: composite_windows``). The window used is recorded.

Outputs: ``{data_root}/{river}/masks/raw/{river}_r{NNN}_{year}.tif`` plus
``{river}_extract_masks_log.csv`` (one row per reach-year) and the run
record. Existing files made with the same settings are skipped, so an
interrupted run resumes where it stopped.

Earth Engine calls go through an ``EEBackend`` so everything else here can
be tested offline with a fake backend.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import ConfigError, Project, River
from .gee import NODATA, Grid
from .provenance import recorded_parameters, write_run_record
from .results import FAILED, OK, STALE, StepResult, check_existing

log = logging.getLogger(__name__)

STEP = "extract_masks"
RES = 30.0
STATS_SCALE = 60.0  # m; coarse scale for the window-selection check


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
    def composite(
        self, geom_lonlat, year: int, window: int, level: int, pct: float
    ): ...
    def n_scenes(self, comp) -> int: ...
    def stats(self, comp, geom_lonlat, scale: float) -> dict[str, float]: ...
    def fetch(self, comp, grid: Grid) -> dict[str, np.ndarray]: ...


class EarthEngine:
    """``EEBackend`` backed by the real Earth Engine API."""

    def __init__(self):
        from . import gee

        self.gee = gee

    def composite(self, geom_lonlat, year, window, level, pct):
        geom = self.gee.ee_geometry(geom_lonlat)
        image, size = self.gee.water_composite(geom, year, window, level, pct)
        return {"image": image, "size": size, "geom": geom}

    def n_scenes(self, comp) -> int:
        return int(comp["size"].getInfo())

    def stats(self, comp, geom_lonlat, scale):
        return self.gee.region_stats(comp["image"], comp["geom"], scale)

    def fetch(self, comp, grid):
        return self.gee.compute_pixels(comp["image"], grid, ["water", "n_obs"])


# ─────────────────────────────────────────────────────────────────────
# Window selection
# ─────────────────────────────────────────────────────────────────────
@dataclass
class WindowChoice:
    window: int | None  # None = no imagery at any window
    n_scenes: int
    clear_fraction: float
    water_km2: float
    expected_water_km2: float | None
    passed: bool
    reason: str


def choose_window(
    backend: EEBackend,
    geom_lonlat,
    year: int,
    river: River,
    expected_water_km2: float | None,
) -> tuple[WindowChoice, object | None]:
    """Try ``composite_windows`` in order; return the first that passes.

    If none passes, returns the candidate with the largest clear fraction
    (flagged ``passed=False``) so the year is still exported and can be
    judged in processing.
    """
    m = river.masks
    best: tuple[WindowChoice, object] | None = None
    for w in m.composite_windows:
        comp = backend.composite(geom_lonlat, year, w, m.dswe_level, m.percentile)
        n = backend.n_scenes(comp)
        if n == 0:
            continue
        s = backend.stats(comp, geom_lonlat, STATS_SCALE)
        total = max(s.get("all", 0.0), 1.0)
        clear = s.get("valid", 0.0) / total
        water_km2 = s.get("water", 0.0) * STATS_SCALE**2 / 1e6
        ok_clear = clear >= m.min_clear_fraction
        ok_water = (
            expected_water_km2 is None
            or water_km2 >= m.min_water_fraction * expected_water_km2
        )
        reasons = []
        if not ok_clear:
            reasons.append(f"clear {clear:.0%} < {m.min_clear_fraction:.0%}")
        if not ok_water:
            reasons.append(
                f"water {water_km2:.2f} km2 < {m.min_water_fraction:.0%} of "
                f"expected {expected_water_km2:.2f} km2"
            )
        choice = WindowChoice(
            w,
            n,
            clear,
            water_km2,
            expected_water_km2,
            ok_clear and ok_water,
            "; ".join(reasons) or "ok",
        )
        if choice.passed:
            return choice, comp
        if best is None or choice.clear_fraction > best[0].clear_fraction:
            best = (choice, comp)
    if best is None:
        return WindowChoice(
            None, 0, 0.0, 0.0, expected_water_km2, False, "no imagery"
        ), None
    return best


# ─────────────────────────────────────────────────────────────────────
# Per reach-year extraction
# ─────────────────────────────────────────────────────────────────────
def _fingerprint(river: River, crs: str) -> dict:
    m = river.masks
    return {
        "dswe_level": m.dswe_level,
        "percentile": m.percentile,
        "composite_windows": list(m.composite_windows),
        "min_clear_fraction": m.min_clear_fraction,
        "min_water_fraction": m.min_water_fraction,
        "res_m": RES,
        "crs": crs,
        "qa_mask_bits": [0, 1, 2, 3, 4, 5],
    }


def _fingerprint_from(params: dict) -> dict:
    keys = (
        "dswe_level",
        "percentile",
        "composite_windows",
        "min_clear_fraction",
        "min_water_fraction",
        "res_m",
        "crs",
        "qa_mask_bits",
    )
    return {k: params.get(k) for k in keys}


def _read_fingerprint(path: Path) -> dict | None:
    import rasterio

    try:
        with rasterio.open(path) as src:
            tag = src.tags().get("rivmapper_params")
        return json.loads(tag) if tag else None
    except Exception:  # noqa: BLE001 — unreadable/partial file → redo
        return None


def write_mask(
    path: Path, grid: Grid, water: np.ndarray, n_obs: np.ndarray, tags: dict
):
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.tif")
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 2,
        "dtype": "uint8",
        "crs": grid.crs,
        "transform": from_origin(grid.x0, grid.y0, grid.res, grid.res),
        "nodata": NODATA,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(water.astype(np.uint8), 1)
        dst.write(n_obs.astype(np.uint8), 2)
        dst.set_band_description(1, "water (0 dry, 1 water, 255 no data)")
        dst.set_band_description(2, "n_obs (clear Landsat observations)")
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
    """Choose a window, download on ``grid``, mask outside the polygon, write."""
    expected = None
    if pd.notna(reach.get("width_ref")) and pd.notna(reach.get("reach_len")):
        expected = float(reach["width_ref"]) * float(reach["reach_len"]) / 1e6
    choice, comp = choose_window(backend, polygon_lonlat, year, river, expected)
    row = {"ds_order": int(reach["ds_order"]), "year": year, **asdict(choice)}
    if comp is None:
        row["status"] = "no_imagery"
        return row

    water = np.full((grid.height, grid.width), NODATA, dtype=np.uint8)
    n_obs = np.zeros((grid.height, grid.width), dtype=np.uint8)
    for r, c, sub in chunks(grid, river.masks.chunk_px):
        arr = backend.fetch(comp, sub)
        water[r : r + sub.height, c : c + sub.width] = arr["water"]
        n_obs[r : r + sub.height, c : c + sub.width] = arr["n_obs"]
    inside = rasterize_inside(reach.geometry, grid)
    water[~inside] = NODATA
    n_obs[~inside] = 0

    valid = water != NODATA
    row["clear_fraction_full_res"] = float(valid.sum() / max(inside.sum(), 1))
    row["water_km2_full_res"] = float((water == 1).sum() * RES * RES / 1e6)
    write_mask(
        out_path,
        grid,
        water,
        n_obs,
        {
            "rivmapper_params": fingerprint,
            "river": river.name,
            "ds_order": str(int(reach["ds_order"])),
            "year": str(year),
            "composite_window": str(choice.window),
            "composite_years": f"{year - choice.window}-{year + choice.window}",
            "n_scenes": str(choice.n_scenes),
            "window_check": choice.reason,
        },
    )
    row["status"] = "ok" if choice.passed else "flagged"
    return row


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
    river.require("masks", "year_start", "year_end", "dswe_level")
    p = river.paths
    if not p.reach_polygons.exists():
        return StepResult(
            river.name,
            STEP,
            FAILED,
            f"reach polygons not found at {p.reach_polygons} — run the reaches step",
        )
    polys = gpd.read_file(p.reach_polygons)
    lines = gpd.read_file(p.reach_lines) if p.reach_lines.exists() else None
    if polys.crs is None or polys.crs.is_geographic:
        raise ConfigError([f"{river.name}: reach polygons must be in a projected CRS"])
    epsg = polys.crs.to_epsg()
    crs = f"EPSG:{epsg}" if epsg else polys.crs.to_wkt()

    m = river.masks
    lo = m.reach_start or 1
    hi = m.reach_end or int(polys["ds_order"].max())
    reaches = polys[polys["ds_order"].between(lo, hi)].sort_values("ds_order")
    if lines is not None and "width_ref" in lines.columns:
        reaches = reaches.drop(
            columns=["width_ref", "reach_len"], errors="ignore"
        ).merge(
            lines[["ds_order", "width_ref", "reach_len"]], on="ds_order", how="left"
        )
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
    new = pd.DataFrame(rows)
    if log_path.exists() and not new.empty:
        old = pd.read_csv(log_path)
        keys = set(zip(new["ds_order"], new["year"]))
        old = old[[k not in keys for k in zip(old["ds_order"], old["year"])]]
        new = pd.concat([old, new], ignore_index=True)
    elif log_path.exists():
        new = pd.read_csv(log_path)
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
            f", {status_counts.get('flagged', 0)} flagged"
            if status_counts.get("flagged")
            else ""
        )
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
