"""Choose each river's analysis CRS (see docs/methods/conventions.md).

Default is the UTM zone containing the centre of the river's bounding box.
If the river extends more than ``LAEA_THRESHOLD_DEG`` of longitude from
that zone's central meridian, a Lambert azimuthal equal-area projection
centred on the river is used instead, so areas stay exact.
"""

from __future__ import annotations

from pyproj import CRS

# Switch to LAEA when the river reaches further than this (degrees of
# longitude) from the UTM zone's central meridian.
LAEA_THRESHOLD_DEG = 3.0


def utm_zone(lon: float) -> int:
    return int((lon + 180.0) // 6.0) % 60 + 1


def utm_crs(lon: float, lat: float) -> CRS:
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + utm_zone(lon))


def choose_analysis_crs(
    bounds_lonlat: tuple[float, float, float, float], override: str = "auto"
) -> tuple[CRS, str]:
    """Return (crs, reason) for a river with lon/lat bounds (minx, miny, maxx, maxy).

    ``override`` is "auto" or anything pyproj understands (e.g. "EPSG:32754").
    """
    if override and override.lower() != "auto":
        return CRS.from_user_input(override), "override"
    minx, miny, maxx, maxy = bounds_lonlat
    lon_c, lat_c = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    utm = utm_crs(lon_c, lat_c)
    cm = (utm_zone(lon_c) - 1) * 6 - 180 + 3
    reach_deg = max(abs(minx - cm), abs(maxx - cm))
    if reach_deg <= LAEA_THRESHOLD_DEG:
        return utm, f"utm (max {reach_deg:.2f} deg from central meridian)"
    laea = CRS.from_proj4(
        f"+proj=laea +lat_0={lat_c:.4f} +lon_0={lon_c:.4f} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return laea, (
        f"laea (river spans {reach_deg:.2f} deg from the UTM central meridian)"
    )


def crs_to_string(crs) -> str:
    """"EPSG:n" if the CRS has an EPSG code, otherwise WKT1 (GDAL dialect).

    Earth Engine parses only WKT1; pyproj's default ``to_wkt()`` is WKT2,
    which EE rejects ("The CRS of a map projection could not be parsed").
    Rasterio and GeoPandas read either. Accepts a pyproj or rasterio CRS.
    """
    if crs is None:
        return ""
    crs = CRS.from_user_input(crs)
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else crs.to_wkt("WKT1_GDAL")
