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


COUNT_BANDS = ["n_obs", "n_wet_1", "n_wet_2", "n_wet_3", "n_wet_4"]
COUNT_MAX = 254  # counts are capped so 255 stays free for "no data"


def dswe_counts(geometry, year: int):
    """Annual DSWE observation counts for ``year`` (calendar year).

    Returns (image, n_scenes). The image has five uint8 bands, each a count
    over all clear Landsat scenes in the year:

    * ``n_obs``   clear observations (pixel not masked by QA_PIXEL);
    * ``n_wet_k`` observations with DSWE class 1..k, for k = 1..4.

    Any DSWE level and percentile can be applied later from these counts.
    """
    ee = _ee()
    col = landsat_collection(geometry, f"{year}-01-01", f"{year + 1}-01-01")

    def _per_scene(img):
        d = dswe(img)
        return (
            ee.Image.cat(
                d.gte(0),
                d.eq(1),
                d.gte(1).And(d.lte(2)),
                d.gte(1).And(d.lte(3)),
                d.gte(1).And(d.lte(4)),
            )
            .rename(COUNT_BANDS)
            .unmask(0)
        )

    counts = col.map(_per_scene).sum().min(COUNT_MAX).toUint8().rename(COUNT_BANDS)
    return counts, col.size()


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


def ee_geometry(geom_lonlat) -> Any:
    """shapely (EPSG:4326) Polygon/MultiPolygon → ee.Geometry."""
    from shapely.geometry import mapping

    return _ee().Geometry(mapping(geom_lonlat), proj="EPSG:4326", geodesic=False)
