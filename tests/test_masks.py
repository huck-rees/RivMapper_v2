"""extract_masks with a fake Earth Engine backend (no network needed)."""

import json

import numpy as np
import pytest
import rasterio

from rivmapper_v2 import Project, run_pipeline
from rivmapper_v2.crs import choose_analysis_crs, crs_to_string
from rivmapper_v2.gee import NODATA, Grid
from rivmapper_v2.masks import chunks, reach_grid, river_origin


# ── CRS ─────────────────────────────────────────────────────────────────
def test_crs_utm_for_compact_river():
    crs, why = choose_analysis_crs((141.1, -8.2, 142.0, -5.3))  # Ok Tedi-Fly
    assert crs_to_string(crs) == "EPSG:32754" and why.startswith("utm")


def test_crs_laea_for_long_river():
    crs, why = choose_analysis_crs((-165.0, 61.0, -141.0, 66.0))  # Yukon-like
    assert crs.to_epsg() is None and "laea" in crs.to_proj4() and why.startswith("laea")
    assert crs.is_projected


def test_crs_override():
    crs, why = choose_analysis_crs((0, 0, 1, 1), "EPSG:3857")
    assert crs.to_epsg() == 3857 and why == "override"


# ── grid ────────────────────────────────────────────────────────────────
def test_reach_grids_share_the_river_grid():
    origin = river_origin((1000.7, 0.0, 9000.0, 5012.3))
    assert origin == (990.0, 5040.0)
    g1 = reach_grid((1000.7, 2000.0, 3001.0, 5012.3), origin, "EPSG:32633")
    g2 = reach_grid((2999.0, 0.0, 9000.0, 3000.0), origin, "EPSG:32633")
    for g in (g1, g2):
        assert (g.x0 - origin[0]) % 30 == 0 and (origin[1] - g.y0) % 30 == 0
    assert g1.x0 <= 1000.7 and g1.x0 + g1.width * 30 >= 3001.0


def test_chunks_cover_grid_exactly():
    g = Grid("EPSG:32633", 0.0, 3000.0, 30.0, 100, 70)
    parts = list(chunks(g, 32))
    cover = np.zeros((70, 100), int)
    for r, c, sub in parts:
        cover[r : r + sub.height, c : c + sub.width] += 1
        assert sub.x0 == c * 30 and sub.y0 == 3000 - r * 30
    assert (cover == 1).all()


def test_pixel_grid_payload():
    pg = Grid("EPSG:32754", 10.0, 20.0, 30.0, 5, 4).pixel_grid()
    assert pg["crsCode"] == "EPSG:32754"
    assert pg["affineTransform"]["scaleY"] == -30.0
    wkt = Grid("PROJCRS[...]", 0, 0, 30, 1, 1).pixel_grid()
    assert "crsWkt" in wkt and "crsCode" not in wkt


# ── fake Earth Engine ───────────────────────────────────────────────────
class FakeEE:
    """10 clear scenes per year (none in 2003); water in a horizontal band.

    In the band: wet in 6 (class 1), 8 (1-2), 9 (1-3), 9 (1-4) of 10 scenes.
    Screening: scene 7 of each year is hazy (5% false water on reference-dry
    pixels); the others are clean. Counts ignore which scenes are kept.
    """

    def __init__(self):
        self.calls = []
        self.fetched_ids = []
        self.screened = []

    def counts(self, geom, year, le07_slc_off=False):
        self.calls.append(year)
        return {"year": year, "le07": le07_slc_off}

    def n_scenes(self, comp):
        return 0 if comp["year"] == 2003 else 10

    def scenes(self, comp, ref, scale_m=60.0):
        self.screened.append((comp["year"], ref.area > 0))
        y = comp["year"]
        return [
            {
                "scene_id": f"LC08_000000_{y}{k:04d}",
                "sensor": "LC08",
                "time_ms": 1e12 + k * 86400e3,
                "clear_frac": 0.9,
                "ref_px": 1000,
                "ref_wet2_px": 30 if k == 7 else 2,
                "ref_wet3_px": 50 if k == 7 else 5,
            }
            for k in range(10)
        ]

    def fetch(self, comp, grid, scene_ids=None):
        self.fetched_ids.append(None if scene_ids is None else len(scene_ids))
        ys = grid.y0 - (np.arange(grid.height) + 0.5) * grid.res
        band = np.abs(ys - ys.mean())[:, None] < 300
        band = np.broadcast_to(band, (grid.height, grid.width))
        z = np.zeros(band.shape, np.uint8)
        return {
            "n_obs": np.full(band.shape, 10, np.uint8),
            "n_wet_1": np.where(band, 6, z).astype(np.uint8),
            "n_wet_2": np.where(band, 8, z).astype(np.uint8),
            "n_wet_3": np.where(band, 9, z).astype(np.uint8),
            "n_wet_4": np.where(band, 9, 1).astype(np.uint8),
        }


@pytest.fixture
def mask_project(synthetic_project):
    cfg = synthetic_project
    cfg.write_text(
        cfg.read_text() + "masks:\n  year_start: 2002\n  year_end: 2004\n"
        "  chunk_px: 256\n  reach_end: 2\n"
    )
    p = Project.load(cfg)
    run_pipeline(p, steps=["sword", "reaches"])
    return p


def test_extract_counts_with_fake_backend(mask_project):
    import pandas as pd

    from rivmapper_v2 import masks

    p = mask_project
    river = p.river("Synth")
    fake = FakeEE()
    [res] = masks.run(p, [river], backend=fake)
    assert res.status == "ok", res.message
    paths = river.paths
    # every reach-year gets a file, including the no-imagery year
    for r in (1, 2):
        for y in (2002, 2003, 2004):
            assert paths.raw_mask(r, y).exists()
    assert sorted(fake.calls) == [
        2002,
        2002,
        2003,
        2003,
        2004,
        2004,
    ]  # one per reach-year
    with rasterio.open(paths.raw_mask(1, 2002)) as src:
        assert src.count == 6 and src.nodata == NODATA and src.dtypes[0] == "uint8"
        assert [d.split(":")[0] for d in src.descriptions] == [
            "n_obs",
            "n_wet_1",
            "n_wet_2",
            "n_wet_3",
            "n_wet_4",
            "n_obs_le07",
        ]
        assert src.crs.to_string().startswith("EPSG:327")
        assert src.transform.a == 30 and src.transform.c % 30 == 0
        assert src.tags()["n_scenes"] == "10"
        assert json.loads(src.tags()["rivmapper_params"])["product"] == "dswe_counts_v2"
        assert src.tags()["n_scenes_kept"] == "9"
        n_obs, w1, w2, w3, w4, _le07 = src.read()
        inside = n_obs != NODATA
        assert (~inside).any() and inside.any()
        for b in (w1, w2, w3, w4):
            assert (b[~inside] == NODATA).all()
        # counts are nested and never exceed n_obs
        assert (w1[inside] <= w2[inside]).all() and (w2[inside] <= w3[inside]).all()
        assert (w3[inside] <= w4[inside]).all() and (w4[inside] <= n_obs[inside]).all()
        assert set(np.unique(w2[inside])) == {0, 8}
    with rasterio.open(paths.raw_mask(1, 2003)) as src:
        n_obs = src.read(1)
        assert set(np.unique(n_obs)) <= {0, NODATA}
    with (
        rasterio.open(paths.raw_mask(1, 2004)) as a,
        rasterio.open(paths.raw_mask(2, 2004)) as b,
    ):
        assert (a.transform.c - b.transform.c) % 30 == 0
        assert (a.transform.f - b.transform.f) % 30 == 0
    log = pd.read_csv(paths.masks_log)
    assert len(log) == 6
    assert log.set_index(["ds_order", "year"]).loc[(1, 2003), "status"] == "no_imagery"
    assert (log.loc[log.status == "ok", "n_obs_median"] == 10).all()
    ok = log[log.status == "ok"]
    assert (ok.n_scenes_kept == 9).all() and (ok.n_scenes_dropped == 1).all()
    # only the kept scenes were requested
    assert set(fake.fetched_ids) == {9}
    # scene manifest: every scene of every imaged reach-year, hazy ones dropped
    man = pd.read_csv(paths.scene_manifest)
    assert len(man) == 2 * 2 * 10  # 2 reaches x 2 imaged years x 10 scenes
    assert (man.groupby(["ds_order", "year"]).kept.sum() == 9).all()
    hazy = man[~man.kept.astype(bool)]
    assert (hazy.reason == "false_water").all() and (hazy.false_rate3 == 0.05).all()
    assert man.date.str.match(r"\d{4}-\d{2}-\d{2}").all()

    # Second run: everything already there.
    fake2 = FakeEE()
    [res2] = masks.run(p, [river], backend=fake2)
    assert res2.status == "skipped" and fake2.calls == []


def test_dswe_level_change_does_not_invalidate_raw(mask_project, synthetic_project):
    """DSWE level / percentile are processing choices: raw stays valid."""
    from rivmapper_v2 import masks

    masks.run(mask_project, [mask_project.river("Synth")], backend=FakeEE())
    cfg = synthetic_project
    cfg.write_text(cfg.read_text() + "  dswe_level: 3\n  percentile: 75\n")
    p2 = Project.load(cfg)
    [res] = masks.run(p2, [p2.river("Synth")], backend=FakeEE())
    assert res.status == "skipped"


def test_missing_years_is_clear_error(synthetic_project):
    from rivmapper_v2 import masks

    p = Project.load(synthetic_project)
    run_pipeline(p, steps=["sword", "reaches"])
    [res] = masks.run(p, [p.river("Synth")], backend=FakeEE())
    assert res.status == "failed" and "year_start" in res.message


def test_missing_gee_project_message(mask_project):
    from rivmapper_v2 import masks

    [res] = masks.run(mask_project, [mask_project.river("Synth")])
    assert res.status == "failed" and "gee.project" in res.message


def test_retired_window_settings_are_ignored(synthetic_project, caplog):
    cfg = synthetic_project
    cfg.write_text(
        cfg.read_text() + "masks:\n  year_start: 2002\n  year_end: 2002\n"
        "  composite_windows: [0, 1, 2]\n  min_clear_fraction: 0.5\n"
    )
    Project.load(cfg)
    assert "composite_windows" in caplog.text


# ── adaptive tiling ─────────────────────────────────────────────────────
class MemoryLimitedEE(FakeEE):
    """Counts that depend only on absolute map position (so any tiling must
    reproduce them exactly), and any request over ``max_px`` pixels fails
    the way Earth Engine does when a tile has too many scenes for memory."""

    def __init__(self, max_px=10**12, message="User memory limit exceeded."):
        super().__init__()
        self.max_px = max_px
        self.message = message
        self.requests = []

    def fetch(self, comp, grid, scene_ids=None):
        self.requests.append((grid.width, grid.height))
        if grid.width * grid.height > self.max_px:
            raise RuntimeError(self.message)
        xs = grid.x0 + (np.arange(grid.width) + 0.5) * grid.res
        ys = grid.y0 - (np.arange(grid.height) + 0.5) * grid.res
        n_obs = ((xs[None, :] // 90 + ys[:, None] // 150) % 7 + 3).astype(np.uint8)
        wet = (((xs[None, :] // 600) + (ys[:, None] // 450)) % 2 == 0).astype(np.uint8)
        out = {"n_obs": n_obs}
        for k in range(1, 5):
            out[f"n_wet_{k}"] = (wet * np.minimum(n_obs, k + 1)).astype(np.uint8)
        return out


def test_split_quadrants_cover_grid_exactly():
    from rivmapper_v2.masks import split_quadrants

    for w, h in [(101, 70), (1, 9), (8, 1), (2, 2)]:
        g = Grid("EPSG:32633", 0.0, 3000.0, 30.0, w, h)
        cover = np.zeros((h, w), int)
        for r, c, sub in split_quadrants(g):
            cover[r : r + sub.height, c : c + sub.width] += 1
            assert sub.x0 == c * 30 and sub.y0 == 3000 - r * 30
        assert (cover == 1).all()


def test_adaptive_fetch_matches_single_download():
    from rivmapper_v2.masks import fetch_adaptive

    g = Grid("EPSG:32633", 0.0, 30000.0, 30.0, 300, 200)
    direct = MemoryLimitedEE().fetch({"year": 2000}, g)
    ee = MemoryLimitedEE(max_px=5000)
    stats = {}
    split = fetch_adaptive(ee, {"year": 2000}, g, min_px=16, stats=stats)
    for b in direct:
        np.testing.assert_array_equal(split[b], direct[b])
    assert stats["splits"] > 0 and stats["tiles"] > 4
    assert sum(w * h <= 5000 for w, h in ee.requests) == stats["tiles"]


def test_adaptive_fetch_gives_up_at_min_size():
    from rivmapper_v2.masks import fetch_adaptive

    g = Grid("EPSG:32633", 0.0, 3000.0, 30.0, 64, 64)
    with pytest.raises(RuntimeError, match="memory limit"):
        fetch_adaptive(MemoryLimitedEE(max_px=10), {"year": 2000}, g, min_px=32)


def test_adaptive_fetch_does_not_retry_other_errors():
    from rivmapper_v2.masks import fetch_adaptive

    ee = MemoryLimitedEE(max_px=0, message="Permission denied for project")
    g = Grid("EPSG:32633", 0.0, 3000.0, 30.0, 64, 64)
    with pytest.raises(RuntimeError, match="Permission"):
        fetch_adaptive(ee, {"year": 2000}, g, min_px=8)
    assert len(ee.requests) == 1


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("User memory limit exceeded.", True),
        ("Computation timed out.", True),
        (
            "Total request size (52428800 bytes) must be less than or equal "
            "to 50331648 bytes.",
            True,
        ),
        ("Request payload size exceeds the limit", True),
        ("Image.select: Pattern 'x' did not match any bands.", False),
    ],
)
def test_is_resource_error(msg, expected):
    from rivmapper_v2.masks import is_resource_error

    assert is_resource_error(RuntimeError(msg)) is expected


def test_large_tile_is_split_and_logged(mask_project):
    import pandas as pd

    from rivmapper_v2 import masks

    river = mask_project.river("Synth")
    river.masks.min_chunk_px = 32
    ee = MemoryLimitedEE(max_px=100 * 100)  # 256 px chunks must be split
    [res] = masks.run(mask_project, [river], backend=ee)
    assert res.status == "ok", res.message
    log = pd.read_csv(river.paths.masks_log)
    assert list(log.columns) == masks.LOG_COLUMNS
    ok = log[log.status == "ok"]
    assert (ok.n_tiles >= 4).all()
    # same content as an unconstrained run
    with rasterio.open(river.paths.raw_mask(1, 2002)) as src:
        split = src.read()
    import shutil

    shutil.rmtree(river.paths.raw_masks_dir)
    river.paths.masks_log.unlink()
    masks.run(mask_project, [river], backend=MemoryLimitedEE())
    with rasterio.open(river.paths.raw_mask(1, 2002)) as src:
        np.testing.assert_array_equal(src.read(), split)


def test_legacy_log_columns_are_dropped(mask_project):
    import pandas as pd

    from rivmapper_v2 import masks

    river = mask_project.river("Synth")
    river.paths.masks_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"ds_order": 9, "year": 1999, "window": 1, "water_km2": 2.0, "status": "ok"}]
    ).to_csv(river.paths.masks_log, index=False)
    masks.run(mask_project, [river], backend=FakeEE())
    log = pd.read_csv(river.paths.masks_log)
    assert list(log.columns) == masks.LOG_COLUMNS
    assert ((log.ds_order == 9) & (log.year == 1999)).any()  # old rows kept


def test_min_chunk_cannot_exceed_chunk(synthetic_project):
    from rivmapper_v2.config import ConfigError

    cfg = synthetic_project
    cfg.write_text(
        cfg.read_text() + "masks:\n  year_start: 2002\n  year_end: 2004\n"
        "  chunk_px: 512\n  min_chunk_px: 1024\n"
    )
    with pytest.raises(ConfigError, match="min_chunk_px"):
        Project.load(cfg)


# ── scene screening (raw product v2) ────────────────────────────────────
def test_screen_rule():
    from rivmapper_v2.config import MaskSettings
    from rivmapper_v2.masks import screen

    m = MaskSettings(max_false_water=0.02, min_ref_px=100)
    rows = screen(
        [
            {"scene_id": "a", "ref_px": 1000, "ref_wet2_px": 1, "ref_wet3_px": 10},
            {"scene_id": "b", "ref_px": 1000, "ref_wet2_px": 5, "ref_wet3_px": 30},
            {"scene_id": "c", "ref_px": 50, "ref_wet2_px": 50, "ref_wet3_px": 50},
            {"scene_id": "d", "ref_px": 0, "ref_wet2_px": 0, "ref_wet3_px": 0},
        ],
        m,
    )
    assert [r["kept"] for r in rows] == [True, False, True, True]
    assert rows[1]["reason"] == "false_water" and rows[1]["false_rate3"] == 0.03
    assert rows[2]["reason"] == rows[3]["reason"] == "too_few_reference_px"
    assert rows[3]["false_rate3"] is None


def test_reference_zone_falls_back_to_half_distance():
    from shapely.geometry import LineString, box

    from rivmapper_v2.masks import reference_zone

    line = LineString([(0, 500), (10000, 500)])
    wide = box(0, -2000, 10000, 3000)
    zone, k = reference_zone(wide, line, 100.0, 10, 1e6)
    assert k == 10 and zone.distance(line) >= 999
    narrow = box(0, -300, 10000, 1300)  # 800 m either side: only 5 W fits
    zone, k = reference_zone(narrow, line, 100.0, 10, 1e6)
    assert k == 5
    zone, k = reference_zone(box(0, 300, 10000, 700), line, 100.0, 10, 1e6)
    assert zone is None and k is None


def test_screen_only_writes_manifest_not_masks(mask_project):
    import pandas as pd

    from rivmapper_v2 import masks

    river = mask_project.river("Synth")
    river.masks.screen_only = True
    fake = FakeEE()
    [res] = masks.run(mask_project, [river], backend=fake)
    assert res.status == "ok" and "would be dropped" in res.message
    assert not river.paths.raw_masks_dir.exists() or not any(
        river.paths.raw_masks_dir.iterdir()
    )
    assert fake.fetched_ids == []
    man = pd.read_csv(river.paths.scene_manifest)
    assert len(man) == 40 and (~man.kept.astype(bool)).sum() == 4
    # running it again resumes: nothing asked again, no duplicate rows
    again = FakeEE()
    masks.run(mask_project, [river], backend=again)
    assert len(pd.read_csv(river.paths.scene_manifest)) == 40
    assert again.screened == []


def test_le07_gap_years_reach_the_backend(mask_project):
    from rivmapper_v2 import masks

    river = mask_project.river("Synth")
    river.masks.le07_gap_years = [2004]
    seen = []

    class Spy(FakeEE):
        def counts(self, geom, year, le07_slc_off=False):
            seen.append((year, le07_slc_off))
            return super().counts(geom, year, le07_slc_off)

    masks.run(mask_project, [river], backend=Spy())
    assert {y for y, le07 in seen if le07} == {2004}


def test_raw_v1_files_are_stale_for_v2(mask_project):
    """Existing v1 raw files must not be silently mixed with v2 ones."""
    import yaml

    from rivmapper_v2 import masks

    river = mask_project.river("Synth")
    [res] = masks.run(mask_project, [river], backend=FakeEE())
    rec_path = river.paths.run_record("extract_masks")
    rec = yaml.safe_load(rec_path.read_text())
    rec["parameters"]["product"] = "dswe_counts_v1"
    rec_path.write_text(yaml.safe_dump(rec))
    river.paths.raw_mask(1, 2002).unlink()
    [res] = masks.run(mask_project, [river], backend=FakeEE())
    assert res.status == "stale" and "--overwrite" in res.message


def test_rate_limit_errors_are_retried():
    from rivmapper_v2.masks import is_rate_error, with_rate_retry

    assert is_rate_error(RuntimeError("Too many concurrent aggregations."))
    assert not is_rate_error(RuntimeError("User memory limit exceeded."))
    calls, waited = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("Too many concurrent aggregations.")
        return "ok"

    assert with_rate_retry(flaky, waits=(1, 2, 4), sleep=waited.append) == "ok"
    assert len(calls) == 3 and len(waited) == 2
    with pytest.raises(RuntimeError, match="concurrent"):
        with_rate_retry(
            lambda: (_ for _ in ()).throw(RuntimeError("Too many concurrent x")),
            waits=(1,),
            sleep=waited.append,
        )
    with pytest.raises(ValueError):  # other errors are not retried
        with_rate_retry(lambda: (_ for _ in ()).throw(ValueError("bad")), sleep=print)
