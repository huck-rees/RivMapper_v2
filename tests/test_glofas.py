"""GloFAS source, offline: grid join, download cache, time alignment, local files.

A fake upstream-area map and a fake EWDS (writing small NetCDF files for
the requested year and area) stand in for the real services.
"""

from __future__ import annotations

import sys
import types

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import yaml

from rivmapper_v2 import glofas, run_pipeline
from rivmapper_v2.discharge import downstream_end, join_reaches
from rivmapper_v2.glofas import (
    Glofas,
    GlofasAccessError,
    build_request,
    request_year,
    cell_center,
    cell_id,
    cell_index,
    parse_cell_id,
    request_area,
    to_day_of_mean,
)

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

REGION = (-9.0, -61.0, -11.0, -58.5)  # N, W, S, E of the fake upstream-area map
TRIB_KM2 = 50.0
BACKGROUND_KM2 = 10.0


# ─────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────
def grid(n: float, w: float, s: float, e: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre lats (descending, as GloFAS) and lons inside a box."""
    lats = np.round(np.arange(n - 0.025, s, -0.05), 4)
    lons = np.round(np.arange(w + 0.025, e, 0.05), 4)
    return lats, lons


def q_true(days: pd.DatetimeIndex) -> np.ndarray:
    """Daily mean discharge of day D: seasonal (low Jan, high Jul) + a small
    day-of-month term so that every day differs from the day before."""
    doy = days.dayofyear.to_numpy()
    return (
        100 + 80 * np.sin(2 * np.pi * (doy - 100) / 365.25) + days.day.to_numpy() * 1e-3
    )


def write_discharge(
    path, valid: pd.DatetimeIndex, box, names=("valid_time", "latitude", "longitude")
) -> None:
    """A GloFAS-like NetCDF: value at valid time T = mean of day T - 1."""
    from rivmapper_v2.glofas import NC_LOCK

    with NC_LOCK:  # fakes write NetCDF from download threads
        _write_discharge(path, valid, box, names)


def _write_discharge(path, valid, box, names) -> None:
    t, la, lo = names
    lats, lons = grid(*box)
    q = q_true(valid - pd.Timedelta(days=1))
    data = np.broadcast_to(q[:, None, None], (len(valid), len(lats), len(lons)))
    xr.Dataset(
        {"dis24": ((t, la, lo), data.astype("float64"))},
        coords={t: valid, la: lats, lo: lons},
    ).to_netcdf(path)


def designated_cells(lines: gpd.GeoDataFrame) -> dict[int, dict]:
    """Per reach: the cell holding its downstream end gets a small tributary
    area; the cell one row south (~5.5 km away) gets the reach's area."""
    lines = lines.sort_values("ds_order").set_index("ds_order")
    out = {}
    for o in lines.index:
        end = gpd.GeoSeries([downstream_end(lines, o)], crs=lines.crs).to_crs(4326)
        r, c = cell_index(end.x.to_numpy(), end.y.to_numpy())
        out[int(o)] = {
            "trib": (int(r[0]), int(c[0])),
            "main": (int(r[0]) + 1, int(c[0])),
            "area": float(lines.loc[o, "us_drain_ar"]),
        }
    return out


def write_uparea(path, lines: gpd.GeoDataFrame) -> dict[int, dict]:
    lats, lons = grid(*REGION)
    a = np.full((len(lats), len(lons)), BACKGROUND_KM2 * 1e6)
    a[:, :3] = np.nan  # some sea
    r0, c0 = cell_index([lons[0]], [lats[0]])
    cells = designated_cells(lines)
    for d in cells.values():
        for key, km2 in (("trib", TRIB_KM2), ("main", d["area"])):
            r, c = d[key]
            a[r - r0[0], c - c0[0]] = km2 * 1e6
    xr.Dataset(
        {"uparea": (("lat", "lon"), a, {"units": "m2"})},
        coords={"lat": lats, "lon": lons},
    ).to_netcdf(path)
    return cells


class FakeEWDS:
    """Writes the requested year over the requested area.

    ``valid_end=True``: valid times 2 Jan .. 1 Jan next year (time stamp at
    the end of each day's mean); False: 1 Jan .. 31 Dec.
    """

    def __init__(self, valid_end: bool = True):
        self.valid_end = valid_end
        self.calls: list[dict] = []

    def __call__(self, request: dict, target) -> None:
        self.calls.append(request)
        assert request["variable"] == ["average_river_discharge_in_the_last_24_hours"]
        assert request["timespan"] == ["time_mean"]
        year = request_year(request)
        n, w, s, e = request["area"]
        first = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=int(self.valid_end))
        valid = pd.date_range(first, periods=366 if year % 4 == 0 else 365, freq="D")
        write_discharge(target, valid, (n, w, s, e))


@pytest.fixture
def lines(stage_project):
    river = stage_project.river("Synth")
    return gpd.read_file(river.paths.reach_lines)


def no_download(request, target):
    raise AssertionError(f"unexpected download: {request_year(request)}")


# ─────────────────────────────────────────────────────────────────────
# Grid helpers, request, time convention
# ─────────────────────────────────────────────────────────────────────
def test_cells_and_request():
    r, c = cell_index([141.275, -59.81], [-5.825, -10.01])
    lon, lat = cell_center(r, c)
    assert np.allclose(lon, [141.275, -59.825]) and np.allclose(lat, [-5.825, -10.025])
    assert parse_cell_id(cell_id(r[0], c[0])) == (r[0], c[0])
    area = request_area([-59.81, -59.31], [-10.01, -9.92])
    assert area == [-9.5, -60.0, -10.5, -59.0]  # N, W, S, E, snapped to 0.5°
    req = build_request("version_4_0", "consolidated", 2005, area)
    assert req["year"] == ["2005"] and len(req["month"]) == 12
    assert req["day"][0] == "01" and req["day"][-1] == "31"
    old = build_request("version_4_0", "consolidated", 2005, area, schema="legacy")
    assert old["hyear"] == ["2005"] and "year" not in old
    assert old["variable"] == ["river_discharge_in_the_last_24_hours"]
    assert req["data_format"] == "netcdf" and req["hydrological_model"] == ["lisflood"]


def test_valid_time_is_end_of_daily_mean():
    idx = pd.DatetimeIndex(["2005-07-02T00:00", "2005-01-01T00:00"])
    assert list(to_day_of_mean(idx)) == [
        pd.Timestamp("2005-07-01"),
        pd.Timestamp("2004-12-31"),
    ]


# ─────────────────────────────────────────────────────────────────────
# Join on the grid
# ─────────────────────────────────────────────────────────────────────
def test_grid_join_rejects_nearer_tributary_cell(lines, tmp_path):
    cells = write_uparea(tmp_path / "uparea.nc", lines)
    src = Glofas(
        cache_file=lambda y: tmp_path / f"{y}.nc",
        uparea_path=tmp_path / "uparea.nc",
        retrieve=no_download,
    )
    join = join_reaches(lines, src, search_km=10, tol=2, source_name="glofas")
    assert (join.method == "area_matched").all()
    for _, row in join.iterrows():
        d = cells[int(row.ds_order)]
        assert row.river_id == cell_id(*d["main"])
        assert (row.cell_row, row.cell_col) == d["main"]
        assert row.model_area_km2 == pytest.approx(d["area"], rel=1e-3)
        assert row.area_ratio == pytest.approx(1.0, abs=1e-3)
    # the tributary cell holds the reach end, so it is always the nearer one
    links = src.links((-61, -11, -58.5, -9))
    trib = links[links.area_km2 == TRIB_KM2]
    assert len(trib) == len(join)


def test_manual_lon_lat_picks_containing_cell(stage_project, tmp_path):
    from rivmapper_v2.config import DischargeSettings
    from rivmapper_v2.discharge import discharge

    river = stage_project.river("Synth")
    lines = gpd.read_file(river.paths.reach_lines)
    cells = write_uparea(tmp_path / "uparea.nc", lines)
    r, c = cells[3]["main"]
    lon, lat = cell_center(r, c)
    settings = DischargeSettings(manual_lon=float(lon) + 0.02, manual_lat=float(lat))
    river = river.model_copy(update={"discharge": settings})
    src = Glofas.from_river(
        river, uparea_path=tmp_path / "uparea.nc", retrieve=FakeEWDS()
    )
    res = discharge(river, src)
    assert res.status == "ok", res.message
    join = pd.read_csv(river.paths.discharge_join)
    assert (join.river_id == cell_id(r, c)).all() and (join.method == "manual").all()


# ─────────────────────────────────────────────────────────────────────
# Daily discharge: downloads, cache, time alignment, local files
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("valid_end", [True, False])
def test_daily_downloads_once_per_year_then_uses_cache(lines, tmp_path, valid_end):
    cells = write_uparea(tmp_path / "uparea.nc", lines)
    ids = [cell_id(*d["main"]) for d in cells.values()]
    ewds = FakeEWDS(valid_end)

    def make(retrieve):
        return Glofas(
            cache_file=lambda y: tmp_path / "cache" / f"Synth_{y}.nc",
            uparea_path=tmp_path / "uparea.nc",
            retrieve=retrieve,
        )

    src = make(ewds)
    q = src.daily(ids, "2002-01-01", "2004-12-31")
    assert [request_year(c) for c in ewds.calls] == [2002, 2003, 2004]
    assert sorted(p.name for p in (tmp_path / "cache").glob("*.nc")) == [
        "Synth_2002.nc",
        "Synth_2003.nc",
        "Synth_2004.nc",
    ]
    assert list(q.columns) == ids and q.index[0] == pd.Timestamp("2002-01-01")
    # each date carries the mean *of that day*
    ok = q.notna().all(axis=1)
    assert np.allclose(q[ok].iloc[:, 0], q_true(q.index[ok]))
    missing = list(q.index[~ok])
    # stamps at the end of the mean: complete; stamps at the start: the
    # last day's mean is in next year's file, which is not fetched
    assert missing == ([] if valid_end else [pd.Timestamp("2004-12-31")])

    again = make(no_download)  # a rerun: everything from the cache
    q2 = again.daily(ids, "2002-01-01", "2004-12-31")
    pd.testing.assert_frame_equal(q, q2)
    assert len(again.describe()["files"]["cached"]) == 3


def test_local_glofas_dir_is_read_first(lines, tmp_path):
    lab = tmp_path / "lab"
    lab.mkdir()
    cells = write_uparea(lab / "uparea_glofas_v4_0.nc", lines)  # found by name
    # a lab file with older coordinate names, 2002-2003 (valid 2 Jan 2002 ..
    # 1 Jan 2004), larger than the river
    valid = pd.date_range("2002-01-02", "2004-01-01", freq="D")
    write_discharge(
        lab / "glofas_v4_2002_2003.nc", valid, REGION, ("time", "lat", "lon")
    )
    (lab / "notes.nc").write_text("not a netcdf")  # skipped quietly
    ids = [cell_id(*d["main"]) for d in cells.values()]
    ewds = FakeEWDS()
    src = Glofas(
        cache_file=lambda y: tmp_path / "cache" / f"{y}.nc",
        glofas_dir=lab,
        retrieve=ewds,
    )
    q = src.daily(ids, "2002-01-01", "2004-12-31")
    assert [request_year(c) for c in ewds.calls] == [2004]  # only the missing year
    assert src.uparea_path().name == "uparea_glofas_v4_0.nc"
    used = src.describe()["files"]
    assert used["local"] == ["glofas_v4_2002_2003.nc"] and len(used["downloaded"]) == 1
    assert q.notna().all().all()
    assert np.allclose(q.iloc[:, 0], q_true(q.index))


# ─────────────────────────────────────────────────────────────────────
# Step end to end
# ─────────────────────────────────────────────────────────────────────
def test_glofas_step_end_to_end(stage_project, tmp_path, monkeypatch):
    river = stage_project.river("Synth")
    lines = gpd.read_file(river.paths.reach_lines)
    # the upstream-area map as if fetched into the shared datasets folder
    monkeypatch.setenv("RIVMAPPER_DATA_DIR", str(tmp_path / "datasets"))
    (tmp_path / "datasets" / "GloFAS").mkdir(parents=True)
    cells = write_uparea(tmp_path / "datasets" / "GloFAS" / "upArea_repaired.nc", lines)
    ewds = FakeEWDS()
    monkeypatch.setattr(glofas, "ewds_retrieve", ewds)
    river.discharge.search_km = 10  # the fake mainstem cell is up to ~9 km away

    table = run_pipeline(stage_project, steps=["discharge"])
    assert table.iloc[0]["status"] == "ok", table.iloc[0]["message"]
    assert "GloFAS cell" in table.iloc[0]["message"]
    p = river.paths
    join = pd.read_csv(p.discharge_join)
    assert (join.source == "glofas").all() and (join.method == "area_matched").all()
    assert set(join.river_id) == {cell_id(*d["main"]) for d in cells.values()}
    daily = pd.read_csv(p.discharge_daily, index_col="date", parse_dates=True)
    assert set(daily.columns) == set(join.river_id)
    stage = pd.read_csv(p.stage_table, keep_default_na=False, na_values=[""])
    by_year = stage.groupby("year").stage_flag.agg(lambda s: set(s.fillna("")))
    assert by_year[2002] == {"low_stage"} and by_year[2003] == {"high_stage"}
    assert len(list(p.glofas_cache_dir.glob("*.nc"))) == 3
    record = yaml.safe_load(p.run_record("discharge").read_text(encoding="utf-8"))
    assert "version_4_0" in str(record)

    # outputs removed: the step runs again, from the cache only
    p.stage_table.unlink()
    n_calls = len(ewds.calls)
    again = run_pipeline(stage_project, steps=["discharge"])
    assert again.iloc[0]["status"] == "ok", again.iloc[0]["message"]
    assert len(ewds.calls) == n_calls


# ─────────────────────────────────────────────────────────────────────
# Access errors
# ─────────────────────────────────────────────────────────────────────
def test_missing_api_key_names_the_file(tmp_path, monkeypatch):
    rc = tmp_path / ".ecmwfdatastoresrc"
    monkeypatch.setenv("ECMWF_DATASTORES_RC_FILE", str(rc))
    monkeypatch.delenv("ECMWF_DATASTORES_KEY", raising=False)
    msg = glofas.api_key_problem()
    assert str(rc) in msg and "url: https://ewds.climate.copernicus.eu/api" in msg
    rc.write_text("url: https://ewds.climate.copernicus.eu/api\n")
    assert "no 'key:' line" in glofas.api_key_problem()
    rc.write_text("url: https://ewds.climate.copernicus.eu/api\nkey: abc-123\n")
    assert glofas.api_key_problem() is None


def _fake_client_module(monkeypatch, error: Exception | None):
    class Client:
        def __init__(self, url=None, **kw):
            assert url == glofas.EWDS_URL

        def retrieve(self, collection_id, request, target=None):
            assert collection_id == "cems-glofas-historical"
            if error:
                raise error

    mod = types.ModuleType("ecmwf.datastores")
    mod.Client = Client
    pkg = types.ModuleType("ecmwf")
    pkg.datastores = mod
    monkeypatch.setitem(sys.modules, "ecmwf", pkg)
    monkeypatch.setitem(sys.modules, "ecmwf.datastores", mod)


def test_licence_and_client_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("ECMWF_DATASTORES_KEY", "abc-123")
    _fake_client_module(
        monkeypatch,
        RuntimeError("403 Client Error: Forbidden\nrequired licences not accepted"),
    )
    with pytest.raises(GlofasAccessError, match="licence has not been accepted") as e:
        glofas.ewds_retrieve({}, tmp_path / "x.nc")
    assert glofas.LICENCE_PAGE in str(e.value)

    _fake_client_module(monkeypatch, RuntimeError("401 Client Error: Unauthorized"))
    with pytest.raises(GlofasAccessError, match="rejected the API key"):
        glofas.ewds_retrieve({}, tmp_path / "x.nc")

    monkeypatch.setitem(sys.modules, "ecmwf.datastores", None)  # not installed
    with pytest.raises(GlofasAccessError, match="ecmwf-datastores-client"):
        glofas.ewds_retrieve({}, tmp_path / "x.nc")


def test_falls_back_to_legacy_request_form(tmp_path):
    """If EWDS rejects the current key names, the old CDS form is tried once
    and then used for the remaining years; a zipped reply is unpacked."""
    import zipfile

    from rivmapper_v2.glofas import Glofas

    calls = []

    def ewds(request, target):
        calls.append(request)
        if "year" in request:
            raise RuntimeError(
                "400 Client Error: invalid request. Request has not produced a "
                "valid combination of values, please check your selection."
            )
        year = request_year(request)
        nc = tmp_path / "inner.nc"
        valid = pd.date_range(f"{year}-01-02", periods=365, freq="D")
        write_discharge(nc, valid, tuple(request["area"]))
        with zipfile.ZipFile(target, "w") as zf:
            zf.write(nc, "data_version-5.nc")

    src = Glofas(
        version="5.0",
        product_type="consolidated",
        glofas_dir=None,
        cache_file=lambda y: tmp_path / "cache" / f"g_{y}.nc",
        retrieve=ewds,
    )
    lon, lat = 141.275, -5.825
    for y in (2003, 2004):
        src._year_files(y, np.array([lon]), np.array([lat]))
    assert ["year" in c for c in calls] == [True, False, False]
    assert src.describe()["request_form"] == "legacy"


def test_years_are_requested_in_parallel_after_the_first(tmp_path):
    import threading
    import time

    from rivmapper_v2.glofas import Glofas

    active, peak, lock = [0], [0], threading.Lock()

    def ewds(request, target):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        year = request_year(request)
        valid = pd.date_range(f"{year}-01-02", periods=365, freq="D")
        write_discharge(target, valid, tuple(request["area"]))
        with lock:
            active[0] -= 1

    src = Glofas(
        version="5.0",
        cache_file=lambda y: tmp_path / "cache" / f"g_{y}.nc",
        retrieve=ewds,
        workers=4,
    )
    q = src.daily(
        [cell_id(*[int(v[0]) for v in cell_index([141.275], [-5.825])])],
        "2001-01-01",
        "2006-12-31",
    )
    assert q.notna().mean().iloc[0] > 0.99
    assert peak[0] >= 3  # after the first year, several at once
    assert (tmp_path / "cache" / "ewds_request_form.txt").read_text() == "current"
