"""Step discharge with a fake GEOGLOWS source (no network needed)."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from rivmapper_v2 import Project, run_pipeline
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
                        "LINKNO": 700000000 + 10 * int(o) + k,
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


@pytest.fixture
def stage_project(synthetic_project):
    cfg = synthetic_project
    cfg.write_text(cfg.read_text() + "masks:\n  year_start: 2002\n  year_end: 2004\n")
    p = Project.load(cfg)
    run_pipeline(p, steps=["sword", "reaches"])
    river = p.river("Synth")
    # a scene manifest as extract_masks writes it: per reach, two scenes a
    # year near the seasonal low (early Jan) in 2002, near the high (Jul)
    # in 2003, mixed in 2004; one dropped scene that must be ignored
    rows = []
    lines = gpd.read_file(river.paths.reach_lines)
    for o in lines["ds_order"]:
        for y, dates in {
            2002: ["01-05", "01-10"],
            2003: ["07-01", "07-05"],
            2004: ["01-05", "07-05"],
        }.items():
            for d in dates:
                rows.append(
                    dict(
                        ds_order=o,
                        year=y,
                        scene_id=f"s{y}{d}",
                        date=f"{y}-{d}",
                        clear_frac=0.8,
                        kept=True,
                        reason="",
                    )
                )
            rows.append(
                dict(
                    ds_order=o,
                    year=y,
                    scene_id=f"bad{y}",
                    date=f"{y}-04-01",
                    clear_frac=0.9,
                    kept=False,
                    reason="false_water",
                )
            )
    river.paths.masks_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(river.paths.scene_manifest, index=False)
    return p


def test_weighted_median():
    assert weighted_median(np.array([1.0, 2, 3]), np.array([1.0, 1, 1])) == 2
    assert weighted_median(np.array([1.0, 2, 3]), np.array([0.1, 0.1, 5])) == 3
    assert np.isnan(weighted_median(np.array([np.nan]), np.array([1.0])))


def test_join_prefers_area_match_over_nearest(stage_project):
    river = stage_project.river("Synth")
    lines = gpd.read_file(river.paths.reach_lines)
    src = FakeGeoglows(lines)
    join = join_reaches(lines, src, search_km=5, tol=2)
    # the tributary is nearer, but the mainstem link matches on area
    assert (join.river_id % 10 == 0).all()
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
    src = FakeGeoglows(gpd.read_file(river.paths.reach_lines))
    monkeypatch.setattr(discharge, "Geoglows", lambda: src)
    table = run_pipeline(stage_project, steps=["discharge"])
    assert table.iloc[0]["status"] == "ok", table.iloc[0]["message"]
    p = river.paths
    join = pd.read_csv(p.discharge_join)
    stage = pd.read_csv(p.stage_table, keep_default_na=False, na_values=[""])
    daily = pd.read_csv(p.discharge_daily, index_col="date", parse_dates=True)
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

    monkeypatch.setattr(discharge, "Geoglows", Offline)
    table = run_pipeline(stage_project, steps=["discharge", "process_masks"])
    st = dict(zip(table.step, table.status, strict=True))
    assert st["discharge"] == "failed"
    assert st["process_masks"] != "blocked"  # fails on its own (no raw masks)


def test_manual_river_id(stage_project, monkeypatch):
    from rivmapper_v2 import discharge

    river = stage_project.river("Synth")
    river.discharge.river_id = 123456789
    src = FakeGeoglows(gpd.read_file(river.paths.reach_lines))
    res = discharge.discharge(river, src)
    assert res.status == "ok", res.message
    join = pd.read_csv(river.paths.discharge_join)
    assert (join.river_id == 123456789).all() and (join.method == "manual").all()
