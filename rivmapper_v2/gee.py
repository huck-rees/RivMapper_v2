"""Google Earth Engine: credentials, Landsat collection, DSWE, pixel download.

Everything that talks to Earth Engine lives here, so the rest of the
package can be tested without it. ``import ee`` happens lazily.

Authentication
--------------
* Personal machine: run ``rivmapper gee auth`` once. It opens a browser,
  stores a refresh token in your user profile, and checks it works.
* Headless server: set ``gee: service_account:`` and ``gee: key_file:`` in
  rivmapper.yaml (a service account registered for Earth Engine).

Either way ``gee: project:`` must name a Google Cloud project registered
for Earth Engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import GeeSettings

log = logging.getLogger(__name__)

_INITIALIZED: str | None = None

# ── Landsat Collection 2, Level 2 ────────────────────────────────────
STANDARD_BANDS = ["uBlue", "Blue", "Green", "Red", "Nir", "Swir1", "Swir2", "QA"]
REFLECTANCE_BANDS = STANDARD_BANDS[:-1]
_BANDS_L89 = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "QA_PIXEL"]
# Landsat 5/7 have no coastal band: B1 stands in for "uBlue" (unused by DSWE).
_BANDS_L57 = ["SR_B1", "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "QA_PIXEL"]
COLLECTIONS = {
    "LT05": "LANDSAT/LT05/C02/T1_L2",
    "LE07": "LANDSAT/LE07/C02/T1_L2",
    "LC08": "LANDSAT/LC08/C02/T1_L2",
    "LC09": "LANDSAT/LC09/C02/T1_L2",
}
# Landsat 7 is used only before the scan-line-corrector failure (31 May 2003).
LE07_DATES = ("1999-04-15", "2003-05-30")

# QA_PIXEL bits (Collection 2): 0 fill, 1 dilated cloud, 2 cirrus,
# 3 cloud, 4 cloud shadow, 5 snow, 6 clear, 7 water.
QA_MASK_BITS = (0, 1, 2, 3, 4, 5)

NODATA = 255


class GeeError(RuntimeError):
    """Earth Engine is not installed, not authenticated or not configured."""


def _ee():
    try:
        import ee
    except ImportError as e:  # pragma: no cover
        raise GeeError(
            "earthengine-api is not installed: conda install -c conda-forge "
            "earthengine-api"
        ) from e
    return ee


# ─────────────────────────────────────────────────────────────────────
# Credentials
# ─────────────────────────────────────────────────────────────────────
def authenticate(force: bool = False) -> None:
    """Interactive browser login; stores credentials in the user profile."""
    _ee().Authenticate(force=force)


def initialize(settings: GeeSettings) -> None:
    """Initialise Earth Engine once per process from the project's settings.

    Raises:
        GeeError: With a message saying exactly what to fix.
    """
    global _INITIALIZED
    ee = _ee()
    if not settings.project:
        raise GeeError(
            "gee.project is not set in rivmapper.yaml. Use the Google Cloud "
            "project registered for Earth Engine (e.g. gee: project: my-project)."
        )
    key = f"{settings.project}|{settings.service_account}"
    if _INITIALIZED == key:
        return
    try:
        if settings.service_account:
            if not settings.key_file or not settings.key_file.exists():
                raise GeeError(
                    f"gee.key_file not found for service account "
                    f"{settings.service_account}: {settings.key_file}"
                )
            creds = ee.ServiceAccountCredentials(
                settings.service_account, str(settings.key_file)
            )
            ee.Initialize(creds, project=settings.project)
        else:
            ee.Initialize(project=settings.project)
    except GeeError:
        raise
    except Exception as e:  # noqa: BLE001 — ee raises several types
        msg = str(e)
        hint = ""
        if "credentials" in msg.lower() or "authenticate" in msg.lower():
            hint = " Run `rivmapper gee auth` first."
        elif "project" in msg.lower() or "permission" in msg.lower():
            hint = (
                f" Check that project '{settings.project}' exists and is "
                f"registered for Earth Engine (code.earthengine.google.com/register)."
            )
        raise GeeError(f"Earth Engine initialisation failed: {msg}{hint}") from None
    _INITIALIZED = key
    log.info("Earth Engine initialised (project %s)", settings.project)


def check_connection() -> dict[str, Any]:
    """Round-trip a trivial computation; returns basic facts for display."""
    ee = _ee()
    n = (
        ee.ImageCollection(COLLECTIONS["LC08"])
        .filterDate("2020-01-01", "2020-01-02")
        .size()
        .getInfo()
    )
    return {"ok": True, "landsat8_scenes_2020_01_01": int(n)}


# ─────────────────────────────────────────────────────────────────────
# Landsat + DSWE
# ─────────────────────────────────────────────────────────────────────
def _qa_mask(image):
    qa = image.select("QA")
    bad = 0
    for b in QA_MASK_BITS:
        bad |= 1 << b
    return image.updateMask(qa.bitwiseAnd(bad).eq(0))


def _rescale(image):
    sr = image.select(REFLECTANCE_BANDS).multiply(0.0000275).add(-0.2)
    return image.addBands(sr, None, True)


def landsat_collection(geometry, start: str, end: str):
    """Landsat 5/7/8/9 C2 L2 over ``geometry`` in [start, end), cloud-masked,
    rescaled to reflectance, with harmonised band names."""
    ee = _ee()
    parts = []
    for key, cid in COLLECTIONS.items():
        bands = _BANDS_L57 if key in ("LT05", "LE07") else _BANDS_L89
        col = ee.ImageCollection(cid).filterBounds(geometry).filterDate(start, end)
        if key == "LE07":
            col = col.filterDate(*LE07_DATES)
        parts.append(col.select(bands, STANDARD_BANDS))
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p)
    return merged.map(_qa_mask).map(_rescale)


def dswe(image):
    """DSWE classes (Jones, 2019): 0 none, 1 high, 2 moderate, 3 wetland, 4 low."""
    mndwi = image.normalizedDifference(["Green", "Swir1"])
    mbsrv = image.select("Green").add(image.select("Red"))
    mbsrn = image.select("Nir").add(image.select("Swir1"))
    ndvi = image.normalizedDifference(["Nir", "Red"])
    awesh = image.expression(
        "Blue + 2.5 * Green - 1.5 * mbsrn - 0.25 * Swir2",
        {
            "Blue": image.select("Blue"),
            "Green": image.select("Green"),
            "mbsrn": mbsrn,
            "Swir2": image.select("Swir2"),
        },
    )
    blue, nir = image.select("Blue"), image.select("Nir")
    swir1, swir2 = image.select("Swir1"), image.select("Swir2")
    t1 = mndwi.gt(0.124)
    t2 = mbsrv.gt(mbsrn)
    t3 = awesh.gt(0)
    t4 = mndwi.gt(-0.44).And(swir1.lt(0.09)).And(nir.lt(0.15)).And(ndvi.lt(0.7))
    t5 = (
        mndwi.gt(-0.5)
        .And(blue.lt(0.1))
        .And(swir1.lt(0.3))
        .And(swir2.lt(0.1))
        .And(nir.lt(0.25))
    )
    t = (
        t1.add(t2.multiply(10))
        .add(t3.multiply(100))
        .add(t4.multiply(1000))
        .add(t5.multiply(10000))
    )
    ee = _ee()
    codes = {
        1: [1111, 10111, 11011, 11101, 11110, 11111],
        2: [111, 1011, 1101, 1110, 10011, 10101, 10110, 11001, 11010, 11100],
        3: [11000],
        4: [11, 101, 110, 1001, 1010, 1100, 10000, 10001, 10010, 10100],
    }
    out = ee.Image(0)
    for cls, vals in codes.items():
        hit = ee.Image(0)
        for v in vals:
            hit = hit.Or(t.eq(v))
        out = out.where(hit, cls)
    return out.rename("dswe").updateMask(t.mask())


def water_composite(
    geometry, year: int, window: int, dswe_level: int, percentile: float
):
    """Two-band composite for ``year`` ± ``window`` years.

    Band ``water``: 1 where the per-pixel ``percentile`` of per-scene binary
    water (DSWE class 1..``dswe_level``) is ≥ 0.5, else 0; ``NODATA`` where
    there is no clear observation. Band ``n_obs``: clear observations
    (capped at 254). Both uint8.
    """
    ee = _ee()
    start, end = f"{year - window}-01-01", f"{year + window + 1}-01-01"
    col = landsat_collection(geometry, start, end)

    def _binary(img):
        d = dswe(img)
        return d.gte(1).And(d.lte(dswe_level)).rename("w")

    binary = col.map(_binary)
    pct = binary.reduce(ee.Reducer.percentile([percentile])).gte(0.5)
    n_obs = binary.count().unmask(0)
    water = (
        ee.Image(NODATA)
        .where(n_obs.gt(0), pct.unmask(NODATA))
        .rename("water")
        .toUint8()
    )
    n_obs = n_obs.min(254).rename("n_obs").toUint8()
    return water.addBands(n_obs), col.size()


# ─────────────────────────────────────────────────────────────────────
# Pixel download on an explicit grid
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Grid:
    """A north-up pixel grid: ``crs`` + top-left corner + pixel size + shape."""

    crs: str  # "EPSG:xxxx" or WKT
    x0: float
    y0: float
    res: float
    width: int
    height: int

    def pixel_grid(self) -> dict:
        g = {
            "dimensions": {"width": int(self.width), "height": int(self.height)},
            "affineTransform": {
                "scaleX": self.res,
                "shearX": 0,
                "translateX": self.x0,
                "shearY": 0,
                "scaleY": -self.res,
                "translateY": self.y0,
            },
        }
        if self.crs.upper().startswith("EPSG:"):
            g["crsCode"] = self.crs
        else:
            g["crsWkt"] = self.crs
        return g


def compute_pixels(image, grid: Grid, bands: list[str]) -> dict[str, np.ndarray]:
    """Fetch ``bands`` of ``image`` on ``grid``; returns {band: 2-D array}."""
    ee = _ee()
    arr = ee.data.computePixels(
        {
            "expression": image.select(bands),
            "fileFormat": "NUMPY_NDARRAY",
            "grid": grid.pixel_grid(),
        }
    )
    return {b: np.asarray(arr[b]) for b in bands}


def region_stats(image, geometry, scale: float) -> dict[str, float]:
    """Clear-pixel and water fractions of ``image`` over ``geometry`` (coarse)."""
    ee = _ee()
    w = image.select("water")
    valid = w.neq(NODATA)
    stats = (
        ee.Image.cat(
            valid.rename("valid"),
            w.eq(1).rename("water"),
            ee.Image(1).rename("all"),
        )
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,
        )
        .getInfo()
    )
    return {k: float(v or 0.0) for k, v in stats.items()}


def ee_geometry(geom_lonlat) -> Any:
    """shapely (EPSG:4326) Polygon/MultiPolygon → ee.Geometry."""
    from shapely.geometry import mapping

    return _ee().Geometry(mapping(geom_lonlat), proj="EPSG:4326", geodesic=False)
