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


def landsat_collection(geometry, start: str, end: str, le07_slc_off: bool = False):
    """Landsat 5/7/8/9 C2 L2 over ``geometry`` in [start, end), cloud-masked,
    rescaled to reflectance, with harmonised band names.

    Each image carries ``scene_id`` (its Landsat scene name, e.g.
    ``LC08_096063_20200101``) and ``sensor`` (LT05/LE07/LC08/LC09).
    Landsat 7 is only used before the scan-line-corrector failure, unless
    ``le07_slc_off`` (gap years with too few other scenes).
    """
    ee = _ee()
    parts = []
    for key, cid in COLLECTIONS.items():
        bands = _BANDS_L57 if key in ("LT05", "LE07") else _BANDS_L89
        col = ee.ImageCollection(cid).filterBounds(geometry).filterDate(start, end)
        if key == "LE07" and not le07_slc_off:
            col = col.filterDate(*LE07_DATES)
        col = col.map(
            lambda img, key=key: img.set(
                {"scene_id": img.get("system:index"), "sensor": key}
            )
        )
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
# Raw product v2 adds the clear looks that came from Landsat 7.
COUNT_BANDS_V2 = [*COUNT_BANDS, "n_obs_le07"]
COUNT_MAX = 254  # counts are capped so 255 stays free for "no data"
JRC_GSW = "JRC/GSW1_4/GlobalSurfaceWater"


def dswe_counts(
    geometry,
    year: int,
    scene_ids: list[str] | None = None,
    le07_slc_off: bool = False,
    bands: list[str] = COUNT_BANDS,
):
    """Annual DSWE observation counts for ``year`` (calendar year).

    Returns (image, n_scenes). The image has uint8 bands, each a count over
    the clear Landsat scenes in the year (all of them, or only
    ``scene_ids`` when given):

    * ``n_obs``   clear observations (pixel not masked by QA_PIXEL);
    * ``n_wet_k`` observations with DSWE class 1..k, for k = 1..4;
    * ``n_obs_le07`` (if requested) clear observations from Landsat 7.

    Any DSWE level and percentile can be applied later from these counts.
    """
    ee = _ee()
    col = landsat_collection(
        geometry, f"{year}-01-01", f"{year + 1}-01-01", le07_slc_off=le07_slc_off
    )
    if scene_ids is not None:
        col = col.filter(ee.Filter.inList("scene_id", ee.List(list(scene_ids))))

    def _per_scene(img):
        d = dswe(img)
        layers = [
            d.gte(0),
            d.eq(1),
            d.gte(1).And(d.lte(2)),
            d.gte(1).And(d.lte(3)),
            d.gte(1).And(d.lte(4)),
        ]
        if "n_obs_le07" in bands:
            is_le07 = ee.Number(
                ee.Algorithms.If(ee.String(img.get("sensor")).equals("LE07"), 1, 0)
            )
            layers.append(d.gte(0).multiply(ee.Image.constant(is_le07)))
        # Explicit cast: a layer multiplied by a constant 0 (non-Landsat-7
        # scenes) gets a narrower pixel type than the same layer for a
        # Landsat 7 scene, and summing mixed types fails EE's
        # homogeneous-collection check.
        return ee.Image.cat(*layers).rename(bands).unmask(0).toUint8()

    # When no scene touches ``geometry`` (possible for a single tile), summing
    # an empty collection gives an image with no bands; use all-zero counts
    # instead. (Chosen server-side with If: merging a constant image into the
    # collection fails EE's same-pixel-type check.)
    zero = ee.Image.constant([0] * len(bands)).rename(bands)
    summed = ee.Image(
        ee.Algorithms.If(col.size().gt(0), col.map(_per_scene).sum(), zero)
    )
    counts = summed.min(COUNT_MAX).toUint8().rename(bands)
    return counts, col.size()


SCENE_FIELDS = [
    "scene_id",
    "sensor",
    "time_ms",
    "clear_frac",
    "ref_px",
    "ref_wet2_px",
    "ref_wet3_px",
]


def scene_stats(
    geometry,
    ref_geometry,
    year: int,
    le07_slc_off: bool = False,
    scale_m: float = 60.0,
) -> list[dict]:
    """Per-scene screening statistics for one reach-year (one EE request).

    For every scene touching ``geometry`` in ``year``:

    * ``clear_frac``: share of ``geometry`` that is clear in the scene;
    * ``ref_px``: clear *reference-dry* pixels — inside ``ref_geometry``
      (the reach polygon away from the channel) and never water in JRC
      Global Surface Water (``max_extent`` = 0, 1984–2021);
    * ``ref_wet2_px`` / ``ref_wet3_px``: of those, classified DSWE 1–2 /
      1–3, i.e. false water.

    All scenes are stacked into one multi-band image and reduced once over
    the reach and once over the reference zone, so the request costs two
    aggregations however many scenes there are (Earth Engine limits
    concurrent aggregations per project). Counted at ``scale_m`` (a rate,
    so a coarser scale is fine and cheaper).
    """
    ee = _ee()
    col = landsat_collection(
        geometry, f"{year}-01-01", f"{year + 1}-01-01", le07_slc_off=le07_slc_off
    )
    ref_dry = ee.Image(JRC_GSW).select("max_extent").unmask(0).eq(0)

    def _layers(img):
        d = dswe(img)
        clear = d.gte(0).unmask(0)
        wet2 = d.gte(1).And(d.lte(2)).unmask(0)
        wet3 = d.gte(1).And(d.lte(3)).unmask(0)
        return (
            ee.Image.cat(
                clear,
                clear.multiply(ref_dry),
                wet2.multiply(ref_dry),
                wet3.multiply(ref_dry),
            )
            .rename(["clear", "ref", "wet2", "wet3"])
            .toFloat()
        )

    n = col.size()
    layers = col.map(_layers)
    stack = layers.toBands()  # bands "{system:index}_{layer}"
    keys = layers.aggregate_array("system:index")
    clear_bands = keys.map(lambda k: ee.String(k).cat("_clear"))
    ref_bands = keys.map(
        lambda k: ee.List(["_ref", "_wet2", "_wet3"]).map(lambda s: ee.String(k).cat(s))
    ).flatten()
    out = ee.Dictionary(
        {
            "n": n,
            "keys": keys,
            "ids": col.aggregate_array("scene_id"),
            "sensor": col.aggregate_array("sensor"),
            "time": col.aggregate_array("system:time_start"),
            "clear": stack.select(clear_bands).reduceRegion(
                ee.Reducer.mean(), geometry, scale_m, maxPixels=1e10, tileScale=4
            ),
            "ref": stack.select(ref_bands).reduceRegion(
                ee.Reducer.sum(), ref_geometry, scale_m, maxPixels=1e10, tileScale=4
            ),
        }
    )
    # toBands() on an empty collection fails; ask only when there are scenes
    info = ee.Algorithms.If(n.gt(0), out, ee.Dictionary({"n": 0})).getInfo()
    return scene_rows(info)


def scene_rows(info: dict) -> list[dict]:
    """Turn the ``scene_stats`` response into one dict per scene."""
    rows = []
    if not info or not info.get("n"):
        return rows
    clear, ref = info.get("clear") or {}, info.get("ref") or {}
    for k, sid, sensor, t in zip(
        info["keys"], info["ids"], info["sensor"], info["time"], strict=True
    ):
        rows.append(
            {
                "scene_id": sid,
                "sensor": sensor,
                "time_ms": t,
                "clear_frac": clear.get(f"{k}_clear") or 0.0,
                "ref_px": ref.get(f"{k}_ref") or 0.0,
                "ref_wet2_px": ref.get(f"{k}_wet2") or 0.0,
                "ref_wet3_px": ref.get(f"{k}_wet3") or 0.0,
            }
        )
    return rows


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

    def bounds(self) -> tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) in ``crs`` units."""
        return (
            self.x0,
            self.y0 - self.height * self.res,
            self.x0 + self.width * self.res,
            self.y0,
        )

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


def grid_geometry(grid: Grid, margin_m: float = 300.0) -> Any:
    """The footprint of ``grid`` (plus ``margin_m``) as an ee.Geometry in the
    grid's own CRS. Used to select only the scenes that touch one tile."""
    ee = _ee()
    minx, miny, maxx, maxy = grid.bounds()
    return ee.Geometry.Rectangle(
        [minx - margin_m, miny - margin_m, maxx + margin_m, maxy + margin_m],
        ee.Projection(grid.crs),
        False,
    )


def ee_geometry(geom_lonlat) -> Any:
    """shapely (EPSG:4326) Polygon/MultiPolygon → ee.Geometry."""
    from shapely.geometry import mapping

    return _ee().Geometry(mapping(geom_lonlat), proj="EPSG:4326", geodesic=False)
