"""Step discharge with a fake GEOGLOWS source (no network needed).

Shared join / stage logic is tested here; GloFAS-specific parts in
test_glofas.py.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from rivmapper_v2 import run_pipeline
from rivmapper_v2.config import ConfigError, DischargeSettings
from rivmapper_v2.discharge import (
    downstream_end,
    join_reaches,
    stage_table,
    weighted_median,
)

YEARS = [2002, 2003, 2004]


class FakeGeoglows:
    """At each reach's downstream end: the mainstem link (area = reach area)
    1 km away, and a small tributary link 300 m away. Discharge: a seasonal
    cycle, 100 m³/s on average, highest in mid-year."""

    def __init__(self, lines: gpd.GeoDataFrame):
        lines = lines.sort_values("ds_order").set_index("ds_order")
        rows = []
        for o in lines.index:
            end = downstream_end(lines, o)
            for k, (dx, area) in enumerate(
                [(1000.0, float(lines.loc[o, "us_drain_ar"])), (300.0, 50.0)]
            ):
                pt = gpd.GeoSeries(
                    gpd.points_from_xy([end.x + dx], [end.y]), crs=lines.crs
                ).to_crs(4326)
                rows.append(
                    {
                        "id": 700000000 + 10 * int(o) + k,
                        "lon": pt.x.iloc[0],
                        "lat": pt.y.iloc[0],
                        "area_km2": area,
                    }
                )
        self.table = pd.DataFrame(rows)
        self.requested: list[int] = []

    def links(self, bbox):
        x0, y0, x1, y1 = bbox
        t = self.table
        return t[t.lon.between(x0, x1) & t.lat.between(y0, y1)].reset_index(drop=True)

    def daily(self, river_ids, start, end):
        self.requested = list(river_ids)
        idx = pd.date_range(start, end, freq="D")
        doy = idx.dayofyear.to_numpy()
        q = 100 + 80 * np.sin(2 * np.pi * (doy - 100) / 365.25)
        return pd.DataFrame({i: q for i in river_ids}, index=idx)


def test_weighted_median():
    assert weighted_median(np.array([1.0, 2, 3]), np.array([1.0, 1, 1])) == 2
    assert weighted_median(np.array([1.0, 2, 3]), np.array([0.1, 0.1, 5])) == 3
    assert np.isnan(weighted_median(np.array([np.nan]), np.array([1.0])))


def test_join_prefers_area_match_over_nearest(stage_project):
    river = stage_project.river("Synth")
    lines = gpd.read_file(river.paths.reach_lines)
    src = FakeGeoglows(lines)
    join = join_reaches(lines, src, search_km=5, tol=2, source_name="geoglows_v2")
    # the tributary is nearer, but the mainstem link matches on area
    assert (join.river_id % 10 == 0).all()
    assert (join.source == "geoglows_v2").all()
    assert join.cell_row.isna().all()  # grid columns are GloFAS only
    assert (join.method == "area_matched").all()
    assert join.dist_km.between(0.9, 1.1).all()
    # a tight radius leaves no area match: best area within 2x radius
    join2 = join_reaches(lines, src, search_km=0.6, tol=2)
    assert (join2.method == "best_area_2x_radius").all()


def test_stage_table_flags_low_and_high_years():
    manifest = pd.DataFrame(
        dict(
            ds_order=1,
            year=[2002, 2002, 2003, 2003, 2003],
            date=["2002-01-05", "2002-01-10", "2003-07-01", "2003-07-05", "2003-04-01"],
            clear_frac=[0.8, 0.8, 0.8, 0.8, 0.9],
            kept=[True, True, True, True, False],
        )
    )
    join = pd.DataFrame(dict(ds_order=[1], river_id=[7]))
    idx = pd.date_range("2002-01-01", "2003-12-31")
    q = 100 + 80 * np.sin(2 * np.pi * (idx.dayofyear.to_numpy() - 100) / 365.25)
    daily = pd.DataFrame({7: q}, index=idx)
    st = stage_table(manifest, join, daily, [2002, 2003], 20, 80).set_index("year")
    assert st.loc[2002, "stage_flag"] == "low_stage"
    assert st.loc[2003, "stage_flag"] == "high_stage"
    assert st.loc[2003, "n_scene_dates"] == 2  # the dropped scene is ignored
    assert st.loc[2002, "q_scene_median"] < st.loc[2002, "q_record_median"]


def test_discharge_step_end_to_end(stage_project, monkeypatch):
    from rivmapper_v2 import discharge

    river = stage_project.river("Synth")
    river.discharge.source = "geoglows_v2"
    src = FakeGeoglows(gpd.read_file(river.paths.reach_lines))
    monkeypatch.setattr(discharge, "Geoglows", lambda: src)
    table = run_pipeline(stage_project, steps=["discharge"])
    assert table.iloc[0]["status"] == "ok", table.iloc[0]["message"]
    p = river.paths
    join = pd.read_csv(p.discharge_join)
    stage = pd.read_csv(p.stage_table, keep_default_na=False, na_values=[""])
    daily = pd.read_csv(p.discharge_daily, index_col="date", parse_dates=True)
    assert p.discharge_join.name == "Synth_discharge_join.csv"
    assert sorted(src.requested) == sorted(join.river_id.unique())
    assert daily.index.min().year == 2002 and daily.index.max().year == 2004
    assert len(stage) == len(join) * 3
    by_year = stage.groupby("year").stage_flag.agg(lambda s: set(s.fillna("")))
    assert by_year[2002] == {"low_stage"} and by_year[2003] == {"high_stage"}
    assert by_year[2004] == {""}
    again = run_pipeline(stage_project, steps=["discharge"])
    assert again.iloc[0]["status"] == "skipped"


def test_discharge_failure_does_not_block_processing(stage_project, monkeypatch):
    from rivmapper_v2 import discharge

    class Offline:
        def links(self, bbox):
            raise OSError("no network")

    monkeypatch.setattr(discharge, "make_source", lambda river: Offline())
    table = run_pipeline(stage_project, steps=["discharge", "process_masks"])
    st = dict(zip(table.step, table.status, strict=True))
    assert st["discharge"] == "failed"
    assert st["process_masks"] != "blocked"  # fails on its own (no raw masks)


def test_manual_river_id(stage_project, monkeypatch):
    from rivmapper_v2 import discharge

    river = stage_project.river("Synth")
    river.discharge.source = "geoglows_v2"
    river.discharge.river_id = 123456789
    src = FakeGeoglows(gpd.read_file(river.paths.reach_lines))
    res = discharge.discharge(river, src)
    assert res.status == "ok", res.message
    join = pd.read_csv(river.paths.discharge_join)
    assert (join.river_id == 123456789).all() and (join.method == "manual").all()


def test_manual_lon_lat_geoglows(stage_project):
    """manual_lon/lat picks the link outlet nearest to the point."""
    from rivmapper_v2 import discharge

    river = stage_project.river("Synth")
    lines = gpd.read_file(river.paths.reach_lines)
    src = FakeGeoglows(lines)
    target = src.table.iloc[4]
    settings = DischargeSettings(
        source="geoglows_v2",
        manual_lon=float(target.lon) + 0.001,
        manual_lat=float(target.lat),
    )
    river = river.model_copy(update={"discharge": settings})
    res = discharge.discharge(river, src)
    assert res.status == "ok", res.message
    join = pd.read_csv(river.paths.discharge_join)
    assert (join.river_id == target["id"]).all() and (join.method == "manual").all()
    assert join.dist_km.notna().all() and join.area_ratio.notna().all()


def test_discharge_settings_validation():
    assert DischargeSettings().source == "glofas"
    for v in (4, 4.0, "4", "4.0", "v4.0", "version_4_0"):
        assert DischargeSettings(glofas_version=v).glofas_version == "4.0"
    s = DischargeSettings(glofas_version="5")
    assert s.glofas_system_version == "version_5_0"
    with pytest.raises(ValueError, match="glofas_version"):
        DischargeSettings(glofas_version="3.1")
    with pytest.raises(ValueError, match="both manual_lon and manual_lat"):
        DischargeSettings(manual_lon=141.2)
    with pytest.raises(ValueError, match="river_id is a GEOGLOWS"):
        DischargeSettings(river_id=5)
    DischargeSettings(source="geoglows_v2", river_id=5)  # fine


def test_per_river_discharge_overrides(tmp_path):
    """Discharge settings as rivers.csv columns, incl. a relative glofas_dir."""
    from rivmapper_v2 import Project

    from .conftest import CHAIN, make_sword, write_project

    make_sword(tmp_path)
    cfg = write_project(
        tmp_path,
        f"""
        river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id,source,glofas_version,manual_lon,manual_lat,glofas_dir,river_id
        A,sa,{CHAIN[0]},{CHAIN[-1]},,5.0,-59.8,-10.0,lab_glofas,
        B,sa,{CHAIN[0]},{CHAIN[-1]},geoglows_v2,,,,,712345678
    """,
        "discharge:\n  glofas_version: 4.0\n",
    )
    p = Project.load(cfg)
    a, b = p.river("A").discharge, p.river("B").discharge
    assert a.source == "glofas" and a.glofas_version == "5.0"
    assert (a.manual_lon, a.manual_lat) == (-59.8, -10.0)
    assert a.glofas_dir == (tmp_path / "lab_glofas").resolve()
    assert b.source == "geoglows_v2" and b.river_id == 712345678
    assert b.glofas_version == "4.0"  # YAML float 4.0 accepted

    bad = write_project(
        tmp_path,
        f"""
        river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id,river_id
        A,sa,{CHAIN[0]},{CHAIN[-1]},712345678
    """,
    )
    with pytest.raises(ConfigError, match="river_id is a GEOGLOWS"):
        Project.load(bad)
