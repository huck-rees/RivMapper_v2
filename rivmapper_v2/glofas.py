"""GloFAS historical reanalysis: a discharge source for step ``discharge``.

GloFAS (Global Flood Awareness System, Copernicus Emergency Management
Service) runs the LISFLOOD hydrological model forced by ERA5 on a global
0.05° grid. Its historical reanalysis (dataset ``cems-glofas-historical``
on the CEMS Early Warning Data Store, EWDS) gives daily river discharge
from 1979 to near present.

What this module does for one river:

* **Cells** (``Glofas.links``): every cell of the GloFAS upstream-area map
  inside a bounding box, with its centre, upstream area (km²) and its
  row/column on the global 0.05° grid. The join itself (area-matched,
  nearest) is shared with GEOGLOWS and lives in ``discharge.py``.
* **Daily discharge** (``Glofas.daily``): for each year, from the first of
  these that has it:

  1. NetCDF files already on disk in ``glofas_dir`` (a lab archive),
     if they cover the cells and the whole year;
  2. this river's download cache, ``{river}/discharge/glofas/``;
  3. a new EWDS request for that year, clipped to the cells' area plus a
     margin, saved to the cache.

  So a rerun, or a run resumed after an interruption, downloads only the
  years it does not have yet.

Time convention: GloFAS values are "discharge in the last 24 hours"; the
time stamp (``valid_time``, 00 UTC) is the *end* of the 24-hour averaging
period (Harrigan et al., 2020). The value stamped 2005-07-02 00:00 is the
mean over 2005-07-01, so dates are shifted back one day (``TIME_SHIFT``)
to label each value with the UTC day it averages, the same calendar as
the scene dates.

Heavy or optional packages (xarray, a NetCDF backend, the EWDS client)
are imported inside functions.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import River

log = logging.getLogger(__name__)

EWDS_URL = "https://ewds.climate.copernicus.eu/api"
DATASET = "cems-glofas-historical"
DATASET_PAGE = "https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical"
LICENCE_PAGE = DATASET_PAGE + "?tab=download#manage-licences"
PROFILE_PAGE = "https://ewds.climate.copernicus.eu/profile"
# EWDS request keys changed with the move from the CDS: "current" is the
# form in ECMWF's EWDS API examples (year/month/day, "average_..." variable,
# timespan); "legacy" is the older CDS form. Requests try current first and
# fall back to legacy if EWDS rejects the combination.
VARIABLE = "average_river_discharge_in_the_last_24_hours"
LEGACY_VARIABLE = "river_discharge_in_the_last_24_hours"
SCHEMAS = ("current", "legacy")

CELL_DEG = 0.05  # GloFAS v4/v5 grid spacing
TIME_SHIFT = pd.Timedelta(days=-1)  # valid_time = end of the 24 h mean
MARGIN_DEG = 0.1  # download area = cells' extent + this margin ...
SNAP_DEG = 0.5  # ... snapped outward to this grid (stable across reruns)

_LAT_NAMES = ("latitude", "lat", "y")
_LON_NAMES = ("longitude", "lon", "x")
_TIME_NAMES = ("valid_time", "time")
_Q_NAMES = ("dis24", "avg_dis24", "dis", "avg_dis", VARIABLE, LEGACY_VARIABLE)

Retrieve = Callable[[dict, Path], None]


class GlofasAccessError(RuntimeError):
    """EWDS cannot be used: client missing, no API key, or licence not accepted."""


# ─────────────────────────────────────────────────────────────────────
# Grid cells
# ─────────────────────────────────────────────────────────────────────
def cell_index(lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
    """Row and column on the global 0.05° grid (row 0 at 90° N, col 0 at 180° W)."""
    row = np.floor((90.0 - np.asarray(lat, dtype=float)) / CELL_DEG).astype(int)
    col = np.floor((np.asarray(lon, dtype=float) + 180.0) / CELL_DEG).astype(int)
    return row, col


def cell_center(row: Any, col: Any) -> tuple[np.ndarray, np.ndarray]:
    """(lon, lat) of cell centres."""
    lat = 90.0 - (np.asarray(row, dtype=float) + 0.5) * CELL_DEG
    lon = -180.0 + (np.asarray(col, dtype=float) + 0.5) * CELL_DEG
    return lon, lat


def cell_id(row: int, col: int) -> str:
    """Stable text id of a cell, e.g. ``r1916c6425``."""
    return f"r{int(row):04d}c{int(col):04d}"


def parse_cell_id(text: str) -> tuple[int, int]:
    """Inverse of ``cell_id``."""
    r, c = str(text).lstrip("r").split("c")
    return int(r), int(c)


def request_area(lons: Iterable[float], lats: Iterable[float]) -> list[float]:
    """EWDS ``area`` [N, W, S, E] around points: margin, snapped outward."""
    lons, lats = np.asarray(list(lons)), np.asarray(list(lats))

    def snap(v: float, up: bool) -> float:
        f = math.ceil if up else math.floor
        return round(f(v / SNAP_DEG) * SNAP_DEG, 4)

    return [
        min(snap(lats.max() + MARGIN_DEG, True), 90.0),
        max(snap(lons.min() - MARGIN_DEG, False), -180.0),
        max(snap(lats.min() - MARGIN_DEG, False), -60.0),
        min(snap(lons.max() + MARGIN_DEG, True), 180.0),
    ]


def build_request(
    system_version: str,
    product_type: str,
    year: int,
    area: list[float],
    schema: str = "current",
) -> dict:
    """One EWDS request: one year of daily discharge over ``area``, NetCDF.

    ``schema`` picks the key names (see ``SCHEMAS``).
    """
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]
    req = {
        "system_version": [system_version],
        "hydrological_model": ["lisflood"],
        "product_type": [product_type],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }
    if schema == "current":
        req.update(
            variable=[VARIABLE],
            timespan=["time_mean"],
            year=[str(year)],
            month=months,
            day=days,
        )
    elif schema == "legacy":
        req.update(
            variable=[LEGACY_VARIABLE],
            hyear=[str(year)],
            hmonth=months,
            hday=days,
        )
    else:
        raise ValueError(f"unknown request schema {schema!r}; choose from {SCHEMAS}")
    return req


def request_year(request: dict) -> int:
    """The year a request asks for, whichever schema it uses."""
    return int((request.get("year") or request.get("hyear"))[0])


def is_invalid_combination(exc: BaseException) -> bool:
    """EWDS's answer to a request whose keys/values it does not accept."""
    low = str(exc).lower()
    return "valid combination" in low or ("400" in low and "invalid request" in low)


# ─────────────────────────────────────────────────────────────────────
# NetCDF reading
# ─────────────────────────────────────────────────────────────────────
def _xr():
    try:
        import xarray as xr
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "GloFAS needs xarray and a NetCDF backend: "
            "conda install -c conda-forge xarray netcdf4"
        ) from e
    return xr


def _name(ds, names: tuple[str, ...], what: str, path: Path) -> str:
    for n in names:
        if n in ds.variables or n in ds.dims:
            return n
    raise ValueError(f"{path.name}: no {what} coordinate (looked for {names})")


def _standardise(ds, path: Path):
    """Rename to (time, lat, lon) with time = valid time; lon in [-180, 180)."""
    if "step" in ds.dims and ds.sizes["step"] == 1:
        ds = ds.squeeze("step", drop=False)
    t = None
    if "valid_time" in ds.dims:
        t = "valid_time"
    elif "valid_time" in ds.coords and ds["valid_time"].ndim == 1:
        ds = ds.swap_dims({ds["valid_time"].dims[0]: "valid_time"})
        t = "valid_time"
    elif "time" in ds.dims:
        t = "time"
    if t == "valid_time" and "time" in ds.variables:
        ds = ds.drop_vars("time")  # forecast reference time, not needed
    lat = _name(ds, _LAT_NAMES, "latitude", path)
    lon = _name(ds, _LON_NAMES, "longitude", path)
    ren = {lat: "lat", lon: "lon"}
    if t is not None:
        ren[t] = "time"
    ds = ds.rename({k: v for k, v in ren.items() if k != v})
    if float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180)
    return ds


def _discharge_var(ds) -> str | None:
    """Name of the discharge variable: a known name, else the only gridded one."""
    name = next((n for n in _Q_NAMES if n in ds.data_vars), None)
    if name is not None:
        return name
    lat = {"lat", "latitude"}
    gridded = [
        v
        for v in ds.data_vars
        if ds[v].ndim >= 2 and lat & set(ds[v].dims) and "upArea" not in v
    ]
    return gridded[0] if len(gridded) == 1 else None


@dataclass(frozen=True)
class GlofasFile:
    """What one discharge NetCDF file holds (from its coordinates only)."""

    path: Path
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    times: pd.DatetimeIndex  # valid times

    def covers(self, lons: np.ndarray, lats: np.ndarray) -> bool:
        """True if every point is within half a cell of the file's extent."""
        h = CELL_DEG / 2 + 1e-6
        return bool(
            (lats >= self.lat_min - h).all()
            and (lats <= self.lat_max + h).all()
            and (lons >= self.lon_min - h).all()
            and (lons <= self.lon_max + h).all()
        )

    def n_days(self, year: int) -> int:
        """Distinct valid dates in ``year``."""
        t = self.times[self.times.year == year]
        return int(t.normalize().nunique())


# HDF5/netCDF4 is not thread-safe: downloads run in parallel threads, but
# every NetCDF open happens under this lock.
NC_LOCK = __import__("threading").RLock()


def inspect_file(path: Path) -> GlofasFile | None:
    """Describe a GloFAS discharge file; None if it is not one."""
    with NC_LOCK:
        return _inspect_file(path)


def _inspect_file(path: Path) -> GlofasFile | None:
    xr = _xr()
    try:
        with xr.open_dataset(path) as ds:
            if _discharge_var(ds) is None:
                return None
            ds = _standardise(ds, path)
            lat, lon = ds["lat"].values, ds["lon"].values
            times = pd.DatetimeIndex(np.atleast_1d(ds["time"].values))
    except Exception as e:  # unreadable or not a grid: skip it
        log.debug("skipping %s: %s", path, e)
        return None
    return GlofasFile(
        Path(path),
        float(lat.min()),
        float(lat.max()),
        float(lon.min()),
        float(lon.max()),
        times,
    )


def _extract_points(
    path: Path,
    ids: list[str],
    lons: np.ndarray,
    lats: np.ndarray,
    year: int | None = None,
) -> pd.DataFrame:
    """Daily discharge at cell centres: index = valid time, one column per id.

    Each centre is matched to the nearest grid point of the file, which
    must lie within half a cell. With ``year``, only the valid times that
    can belong to that year's days are read (1 Jan 00 UTC to 1 Jan next
    year 00 UTC).
    """
    xr = _xr()
    with xr.open_dataset(path) as ds:
        var = _discharge_var(ds)
        ds = _standardise(ds, path)
        da = ds[var]
        if year is not None:
            da = da.sel(time=slice(f"{year}-01-01", f"{year + 1}-01-01T00:00"))
        lat_v, lon_v = da["lat"].values, da["lon"].values
        iy = np.abs(lat_v[None, :] - lats[:, None]).argmin(axis=1)
        ix = np.abs(lon_v[None, :] - lons[:, None]).argmin(axis=1)
        off = np.maximum(np.abs(lat_v[iy] - lats), np.abs(lon_v[ix] - lons))
        if (off > CELL_DEG / 2 + 1e-6).any():
            bad = [ids[k] for k in np.flatnonzero(off > CELL_DEG / 2 + 1e-6)]
            raise ValueError(f"{path.name} does not contain cells {bad}")
        pts = da.isel(
            lat=xr.DataArray(iy, dims="pt"), lon=xr.DataArray(ix, dims="pt")
        ).transpose("time", "pt")
        values = np.asarray(pts.values, dtype=float)
        times = pd.DatetimeIndex(np.atleast_1d(pts["time"].values))
    return pd.DataFrame(values, index=times, columns=list(ids))


def extract_points(*args, **kwargs):
    """Daily series at points from one file (see ``_extract_points``)."""
    with NC_LOCK:
        return _extract_points(*args, **kwargs)


def to_day_of_mean(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Valid time (end of the 24 h mean) → the UTC day the mean covers."""
    return (pd.DatetimeIndex(index) + TIME_SHIFT).normalize()


def read_uparea(path: Path, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """Cells of an upstream-area map inside ``bbox`` (lon/lat).

    Returns:
        DataFrame with ``lon``, ``lat`` (cell centres) and ``area_km2``;
        cells without a value (sea) are dropped. Areas in m² (GloFAS
        convention) are converted to km².
    """
    xr = _xr()
    x0, y0, x1, y1 = bbox
    with xr.open_dataset(path) as ds:
        ds = _standardise(ds, Path(path))
        grids = [v for v in ds.data_vars if set(ds[v].dims) == {"lat", "lon"}]
        if not grids:
            raise ValueError(f"{Path(path).name}: no lat/lon grid variable")
        var = next((v for v in grids if "area" in v.lower()), grids[0])
        lat, lon = ds["lat"].values, ds["lon"].values
        iy = np.flatnonzero((lat >= y0) & (lat <= y1))
        ix = np.flatnonzero((lon >= x0) & (lon <= x1))
        if not len(iy) or not len(ix):
            return pd.DataFrame(columns=["lon", "lat", "area_km2"])
        sub = ds[var].isel(lat=iy, lon=ix).transpose("lat", "lon")
        a = np.asarray(sub.values, dtype=float)
        units = str(ds[var].attrs.get("units", "")).lower()
    glon, glat = np.meshgrid(lon[ix], lat[iy])
    out = pd.DataFrame(
        {"lon": glon.ravel(), "lat": glat.ravel(), "area_km2": a.ravel()}
    )
    out = out[np.isfinite(out["area_km2"]) & (out["area_km2"] > 0)]
    # GloFAS upstream area is in m² (a 0.05° cell alone is ~10-30 km²,
    # i.e. > 1e7 m²); accept a km² map too.
    in_m2 = "km" not in units and (units.startswith("m") or out["area_km2"].max() > 1e8)
    if in_m2:
        out["area_km2"] = out["area_km2"] / 1e6
    return out.reset_index(drop=True)


@cache
def _scan_folder(folder: str) -> tuple[GlofasFile, ...]:
    files = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in (".nc", ".nc4", ".netcdf") and p.is_file():
            if "uparea" in p.name.lower():
                continue
            info = inspect_file(p)
            if info is not None:
                files.append(info)
    log.info("  glofas_dir: %d GloFAS discharge file(s) in %s", len(files), folder)
    return tuple(files)


def find_uparea_file(folder: Path | None) -> Path | None:
    """An upstream-area map (name containing 'uparea') in ``folder``, if any."""
    if folder is None or not Path(folder).is_dir():
        return None
    hits = sorted(
        p
        for p in Path(folder).rglob("*")
        if "uparea" in p.name.lower() and p.suffix.lower() in (".nc", ".nc4")
    )
    return hits[0] if hits else None


# ─────────────────────────────────────────────────────────────────────
# EWDS access
# ─────────────────────────────────────────────────────────────────────
def rc_file() -> Path:
    """The EWDS client's config file (``ECMWF_DATASTORES_RC_FILE`` or home)."""
    env = os.environ.get("ECMWF_DATASTORES_RC_FILE")
    return Path(env).expanduser() if env else Path.home() / ".ecmwfdatastoresrc"


def _setup_text() -> str:
    return (
        f"create the text file {rc_file()} with these two lines:\n"
        f"    url: {EWDS_URL}\n"
        f"    key: <your personal access token>\n"
        f"The token is on your EWDS profile page ({PROFILE_PAGE}) once you are "
        f"logged in. See docs/methods/discharge.md, 'Access'."
    )


def api_key_problem() -> str | None:
    """Why no EWDS API key can be found, or None if one can."""
    if os.environ.get("ECMWF_DATASTORES_KEY"):
        return None
    rc = rc_file()
    if not rc.exists():
        return f"No EWDS API key found: {_setup_text()}"
    for line in rc.read_text(encoding="utf-8", errors="replace").splitlines():
        k, _, v = line.partition(":")
        if k.strip() == "key" and v.strip():
            return None
    return f"{rc} has no 'key:' line: {_setup_text()}"


def _unzip_if_needed(path: Path) -> None:
    """If EWDS sent a zip (it sometimes ignores 'unarchived'), keep the .nc."""
    import zipfile

    if not zipfile.is_zipfile(path):
        return
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".nc", ".nc4"))]
        if len(members) != 1:
            raise RuntimeError(
                f"{path.name}: expected one NetCDF file in the zip, found {members}"
            )
        data = zf.read(members[0])
    path.write_bytes(data)


def ewds_retrieve(request: dict, target: Path) -> None:
    """Download one request from EWDS to ``target`` (blocking).

    Raises:
        GlofasAccessError: Client not installed, no API key, key rejected,
            or the dataset licence not accepted.
    """
    try:
        from ecmwf.datastores import Client
    except ImportError as e:
        raise GlofasAccessError(
            "Downloading GloFAS needs the EWDS client: "
            "conda install -c conda-forge ecmwf-datastores-client"
        ) from e
    problem = api_key_problem()
    if problem:
        raise GlofasAccessError(problem)
    client = Client(url=EWDS_URL, progress=False)
    try:
        client.retrieve(DATASET, request, target=str(target))
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if "licen" in low:
            raise GlofasAccessError(
                f"EWDS says the dataset licence has not been accepted. Log in, "
                f"open {LICENCE_PAGE} and accept the CEMS-FLOODS licence at the "
                f"bottom of the download form, then rerun. (EWDS: {msg.strip()})"
            ) from e
        if "401" in msg or "unauthori" in low or "authenticat" in low:
            raise GlofasAccessError(
                f"EWDS rejected the API key. Check the key in {rc_file()} "
                f"against {PROFILE_PAGE}. (EWDS: {msg.strip()})"
            ) from e
        raise


# ─────────────────────────────────────────────────────────────────────
# Source
# ─────────────────────────────────────────────────────────────────────
class Glofas:
    """GloFAS historical reanalysis for one river (see module docstring).

    Args:
        cache_file: ``year -> path`` of this river's cached download.
        version: GloFAS version, ``"4.0"`` or ``"5.0"``.
        product_type: ``"consolidated"`` or ``"intermediate"``.
        glofas_dir: Folder of GloFAS NetCDF files already on disk.
        uparea_path: Upstream-area map; default: one in ``glofas_dir``,
            else the registered dataset ``glofas_uparea`` (downloaded once
            into the shared datasets folder).
        retrieve: Download function ``(request, target)``; default EWDS.
    """

    def __init__(
        self,
        cache_file: Callable[[int], Path],
        version: str = "4.0",
        product_type: str = "consolidated",
        glofas_dir: Path | None = None,
        uparea_path: Path | None = None,
        retrieve: Retrieve | None = None,
        workers: int = 1,
    ):
        self.workers = max(1, int(workers))
        self.cache_file = cache_file
        self._schema: str | None = None  # request form EWDS accepted
        try:  # remembered from an earlier run (lets parallel requests start at once)
            f = self._schema_file()
            if f is not None and f.exists() and f.read_text().strip() in SCHEMAS:
                self._schema = f.read_text().strip()
        except Exception:  # noqa: BLE001 — a hint only
            pass
        self.version = version
        self.product_type = product_type
        self.glofas_dir = Path(glofas_dir) if glofas_dir else None
        self._uparea_path = Path(uparea_path) if uparea_path else None
        self.retrieve = retrieve or ewds_retrieve
        self.used: dict[str, list[str]] = {"local": [], "cached": [], "downloaded": []}

    @classmethod
    def from_river(cls, river: River, **kw) -> Glofas:
        d = river.discharge
        return cls(
            cache_file=lambda y: river.paths.glofas_cache_file(
                d.glofas_version, d.glofas_product_type, y
            ),
            version=d.glofas_version,
            product_type=d.glofas_product_type,
            glofas_dir=d.glofas_dir,
            workers=d.glofas_workers,
            **kw,
        )

    @property
    def system_version(self) -> str:
        return "version_" + self.version.replace(".", "_")

    # ── cells ───────────────────────────────────────────────────────
    def uparea_path(self) -> Path:
        if self._uparea_path is None:
            local = find_uparea_file(self.glofas_dir)
            if local is not None:
                self._uparea_path = local
            else:
                from .data import fetch, get_dataset

                ds = get_dataset("glofas_uparea", self.version)
                self._uparea_path = (
                    fetch("glofas_uparea", self.version) / ds.expected_files[0]
                )
        return self._uparea_path

    def links(self, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
        """GloFAS cells in ``bbox``: id, lon, lat, area_km2, cell_row, cell_col."""
        cells = read_uparea(self.uparea_path(), bbox)
        frac = ((cells["lon"].to_numpy() + 180.0) / CELL_DEG) % 1.0
        if len(cells) and np.abs(frac - 0.5).max() > 0.1:
            log.warning(
                "  %s: grid coordinates are not at 0.05° cell centres; "
                "check the file matches GloFAS v%s",
                self.uparea_path().name,
                self.version,
            )
        row, col = cell_index(cells["lon"], cells["lat"])
        cells["cell_row"], cells["cell_col"] = row, col
        cells["id"] = [cell_id(r, c) for r, c in zip(row, col, strict=True)]
        return cells[["id", "lon", "lat", "area_km2", "cell_row", "cell_col"]]

    # ── daily discharge ─────────────────────────────────────────────
    def daily(self, river_ids: list[str], start: str, end: str) -> pd.DataFrame:
        """Daily mean discharge (m³/s) for cells: index = UTC day, column = id."""
        ids = [str(i) for i in river_ids]
        rc = np.array([parse_cell_id(i) for i in ids])
        lons, lats = cell_center(rc[:, 0], rc[:, 1])
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        years = list(range(t0.year, t1.year + 1))
        files = self._all_year_files(years, lons, lats)
        frames = [
            extract_points(path, ids, lons, lats, year)
            for year in years
            for path in files[year]
        ]
        days = pd.date_range(t0, t1, freq="D", name="date")
        if not frames:
            return pd.DataFrame(index=days, columns=ids, dtype=float)
        q = pd.concat(frames)
        q.index = to_day_of_mean(q.index)
        q = q[~q.index.duplicated(keep="first")].sort_index()
        out = q.reindex(days)
        out.index.name = "date"
        return out

    def _schema_file(self) -> Path | None:
        try:
            return Path(self.cache_file(2000)).parent / "ewds_request_form.txt"
        except Exception:  # noqa: BLE001
            return None

    def _all_year_files(self, years, lons, lats) -> dict[int, list[Path]]:
        """Files for every year; missing years are requested ``workers`` at a
        time. Years run one by one until a download has shown which request
        form EWDS accepts, then the rest go in parallel."""
        from concurrent.futures import ThreadPoolExecutor

        out: dict[int, list[Path]] = {}
        todo = list(years)
        while todo and (self._schema is None or self.workers == 1):
            y = todo.pop(0)
            out[y] = self._year_files(y, lons, lats)
            if self.workers > 1 and self._schema is not None:
                break
        if todo:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for y, files in zip(
                    todo,
                    pool.map(lambda y: self._year_files(y, lons, lats), todo),
                    strict=True,
                ):
                    out[y] = files
        return out

    def _year_files(self, year: int, lons: np.ndarray, lats: np.ndarray) -> list[Path]:
        """Files that together hold ``year`` for these cells (see module doc)."""
        need = 365 + (pd.Timestamp(year=year, month=12, day=31).dayofyear == 366)
        if self.glofas_dir is not None:
            local = [
                f
                for f in _scan_folder(str(self.glofas_dir))
                if f.n_days(year) and f.covers(lons, lats)
            ]
            n = len(set().union(*(set(f.times[f.times.year == year]) for f in local)))
            if local and n >= need - 1:  # -1: one end day may sit in the next file
                self.used["local"] += [f.path.name for f in local]
                return [f.path for f in local]
        path = self.cache_file(year)
        if path.exists():
            info = inspect_file(path)
            if info is not None and info.covers(lons, lats):
                self.used["cached"].append(path.name)
                return [path]
            log.info("  %s does not cover the cells; downloading again", path.name)
        self._download(year, lons, lats, path)
        info = inspect_file(path)
        if info is None or not info.covers(lons, lats):
            raise RuntimeError(f"downloaded {path.name} does not cover the cells")
        self.used["downloaded"].append(path.name)
        return [path]

    def _download(self, year: int, lons, lats, path: Path) -> None:
        area = request_area(lons, lats)
        log.info(
            "  GloFAS v%s %s %d, area %s: requesting from EWDS",
            self.version,
            self.product_type,
            year,
            area,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        schemas = [self._schema] if self._schema else list(SCHEMAS)
        for i, schema in enumerate(schemas):
            request = build_request(
                self.system_version, self.product_type, year, area, schema
            )
            part.unlink(missing_ok=True)
            try:
                self.retrieve(request, part)
            except Exception as e:
                if i + 1 < len(schemas) and is_invalid_combination(e):
                    log.info(
                        "  EWDS rejected the %s request form; trying %s",
                        schema,
                        schemas[i + 1],
                    )
                    continue
                raise
            self._schema = schema  # remember what works for later years
            try:
                f = self._schema_file()
                if f is not None:
                    f.write_text(schema)
            except OSError:
                pass
            break
        _unzip_if_needed(part)
        part.replace(path)  # only complete downloads get the final name

    def describe(self) -> dict:
        """Provenance for the run record."""
        return {
            "dataset": DATASET,
            "doi": "10.24381/cds.a4fdd6b9",
            "system_version": self.system_version,
            "product_type": self.product_type,
            "request_form": self._schema,
            "variable": VARIABLE,
            "time_convention": (
                "valid_time is the end of the 24 h mean; dates = valid_time - 1 day"
            ),
            "uparea_file": str(self.uparea_path()) if self._uparea_path else None,
            "glofas_dir": str(self.glofas_dir) if self.glofas_dir else None,
            "files": {k: sorted(set(v)) for k, v in self.used.items()},
        }
