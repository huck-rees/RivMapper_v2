"""Step ``export_rgb``: yearly true-colour composites for labelling and review.

One uint8 RGB GeoTIFF per reach-year, on exactly the same 30 m river grid
as the raw counts and the masks (``masks/rgb/{river}_r{NNN}_{year}.tif``):
the per-pixel median of the year's cloud-masked Landsat surface
reflectance (the scenes kept by scene screening when the manifest has
them; Landsat 7 after its scan-line failure only in ``masks.le07_gap_years``,
as for the masks), red / green / blue stretched linearly from 0–``RGB_MAX`` reflectance
to 0–255. Pixels with no clear look are 0 in all bands. An existing file
that is entirely 0 is redone on the next run.

Not part of a default run (``rivmapper run … --steps export_rgb``): it is
for people, not for the masks. Existing files are kept unless --overwrite.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import Project, River
from .crs import crs_to_string
from .masks import RES, chunks, reach_grid, river_origin, with_rate_retry
from .results import FAILED, OK, StepResult

log = logging.getLogger(__name__)

STEP = "export_rgb"
RGB_MAX = 0.3  # reflectance mapped to 255
TILE_PX = 1024


def _kept_scenes(manifest: pd.DataFrame | None, o: int, year: int):
    if manifest is None:
        return None
    m = manifest[(manifest.ds_order == o) & (manifest.year == year)]
    if m.empty:
        return None
    ids = m.loc[m.kept.astype(str).str.lower().isin(["true", "1"]), "scene_id"]
    return list(ids) if len(ids) else None


def composite(tile_geom, year: int, scene_ids=None, le07_slc_off: bool = False):
    """ee.Image: uint8 R, G, B median composite for ``year``.

    ``le07_slc_off``: include Landsat 7 after its scan-line failure (the
    gap years the masks also use, e.g. 2012).
    """
    from . import gee

    ee = gee._ee()
    col = gee.landsat_collection(
        tile_geom, f"{year}-01-01", f"{year + 1}-01-01", le07_slc_off=le07_slc_off
    )
    if scene_ids:
        col = col.filter(ee.Filter.inList("scene_id", scene_ids))
    bands = ["Red", "Green", "Blue"]
    img = col.select(bands).median()
    rgb = img.divide(RGB_MAX).multiply(255).clamp(0, 255).toUint8().unmask(0)
    # a year with no usable scene: median() has no bands -> all-zero image
    zero = ee.Image.constant([0, 0, 0]).rename(bands).toUint8()
    return ee.Image(ee.Algorithms.If(col.size().gt(0), rgb, zero))


def _all_zero(path) -> bool:
    """An existing composite with no data (e.g. a gap year exported before
    Landsat 7 was included): redone without --overwrite."""
    import rasterio

    with rasterio.open(path) as s:
        return not s.read(1).any()


def write_rgb(path, grid, arr: np.ndarray, tags: dict) -> None:
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 3,
        "dtype": "uint8",
        "crs": grid.crs,
        "transform": from_origin(grid.x0, grid.y0, grid.res, grid.res),
        "compress": "deflate",
        "photometric": "RGB",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    tmp = path.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(arr)
        for i, b in enumerate(("red", "green", "blue"), start=1):
            dst.set_band_description(i, f"{b}: median surface reflectance x 255/{RGB_MAX}")
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    tmp.replace(path)


def export_rgb(river: River, years: list[int] | None = None) -> StepResult:
    from . import gee

    p = river.paths
    if not p.reach_polygons.exists():
        return StepResult(river.name, STEP, FAILED, "no reach polygons — run reaches")
    polys = gpd.read_file(p.reach_polygons).sort_values("ds_order")
    crs = crs_to_string(polys.crs)
    origin = river_origin(tuple(polys.total_bounds))
    m = river.masks
    years = years or list(range(m.year_start, m.year_end + 1))
    manifest = pd.read_csv(p.scene_manifest) if p.scene_manifest.exists() else None
    done = skipped = failed = 0
    for _, reach in polys.iterrows():
        o = int(reach["ds_order"])
        grid = reach_grid(tuple(reach.geometry.bounds), origin, crs, RES)
        for y in years:
            out = p.rgb(o, y)
            if out.exists() and not river.overwrite and not _all_zero(out):
                skipped += 1
                continue
            arr = np.zeros((3, grid.height, grid.width), np.uint8)
            ids = _kept_scenes(manifest, o, y)
            try:
                for r, c, sub in chunks(grid, TILE_PX):
                    img = composite(
                        gee.grid_geometry(sub), y, ids, le07_slc_off=y in m.le07_gap_years
                    )
                    px = with_rate_retry(
                        gee.compute_pixels, img, sub, ["Red", "Green", "Blue"]
                    )
                    for i, b in enumerate(("Red", "Green", "Blue")):
                        arr[i, r : r + sub.height, c : c + sub.width] = px[b]
            except Exception as e:  # noqa: BLE001
                failed += 1
                log.warning("  r%03d %d failed: %s", o, y, str(e)[:120])
                continue
            write_rgb(
                out,
                grid,
                arr,
                {"river": river.name, "ds_order": o, "year": y,
                 "scenes": "screened (kept)" if ids else "all clear"},
            )
            done += 1
        log.info("  reach %d: RGB written", o)
    msg = f"{done} RGB composites written, {skipped} already there"
    if failed:
        msg += f", {failed} failed (rerun to retry them)"
    return StepResult(river.name, STEP, FAILED if failed and not done else OK, msg)


def run(project: Project, rivers: list[River]) -> list[StepResult]:
    from . import gee

    try:
        gee.initialize(project.settings.gee)
    except gee.GeeError as e:
        return [StepResult(r.name, STEP, FAILED, str(e)) for r in rivers]
    out = []
    for r in rivers:
        log.info("%s", r.name)
        try:
            out.append(export_rgb(r))
        except Exception as e:  # noqa: BLE001
            log.debug("%s: error", r.name, exc_info=True)
            out.append(StepResult(r.name, STEP, FAILED, f"{type(e).__name__}: {e}"))
    return out
