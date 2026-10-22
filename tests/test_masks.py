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
    """Water in a horizontal band; window 0 is cloudy for even years."""

    def __init__(self):
        self.calls = []

    def composite(self, geom, year, window, level, pct):
        self.calls.append((year, window))
        return {"year": year, "window": window}

    def n_scenes(self, comp):
        return 0 if comp["year"] == 2003 else 5 + comp["window"]

    def stats(self, comp, geom, scale):
        cloudy = comp["year"] % 2 == 0 and comp["window"] == 0
        return {"all": 1000.0, "valid": 200.0 if cloudy else 950.0, "water": 300.0}

    def fetch(self, comp, grid):
        ys = grid.y0 - (np.arange(grid.height) + 0.5) * grid.res
        water = np.where(np.abs(ys - ys.mean())[:, None] < 300, 1, 0)
        water = np.broadcast_to(water, (grid.height, grid.width)).astype(np.uint8)
        return {"water": water.copy(), "n_obs": np.full_like(water, 7)}


@pytest.fixture
def mask_project(synthetic_project):
    cfg = synthetic_project
    cfg.write_text(
        cfg.read_text()
        + "masks:\n  year_start: 2002\n  year_end: 2004\n  dswe_level: 2\n"
        "  chunk_px: 256\n  reach_end: 2\n  composite_windows: [0, 1, 2]\n"
    )
    p = Project.load(cfg)
    run_pipeline(p, steps=["sword", "reaches"])
    return p


def test_extract_masks_with_fake_backend(mask_project):
    from rivmapper_v2 import masks

    p = mask_project
    river = p.river("Synth")
    fake = FakeEE()
    [res] = masks.run(p, [river], backend=fake)
    assert res.status == "ok", res.message
    paths = river.paths
    # 2 reaches x 3 years, 2003 has no imagery
    assert paths.raw_mask(1, 2002).exists() and not paths.raw_mask(1, 2003).exists()
    # even year: window 0 too cloudy -> widened to 1
    assert (2002, 0) in fake.calls and (2002, 1) in fake.calls
    with rasterio.open(paths.raw_mask(1, 2002)) as src:
        assert src.count == 2 and src.nodata == NODATA and src.dtypes[0] == "uint8"
        assert src.crs.to_string().startswith("EPSG:327")
        assert src.transform.a == 30 and src.transform.c % 30 == 0
        assert src.tags()["composite_window"] == "1"
        assert json.loads(src.tags()["rivmapper_params"])["qa_mask_bits"] == [
            0,
            1,
            2,
            3,
            4,
            5,
        ]
        w = src.read(1)
        assert set(np.unique(w)) <= {0, 1, NODATA}
        assert (w == NODATA).any() and (w == 1).any()  # outside polygon + water
    with (
        rasterio.open(paths.raw_mask(1, 2004)) as a,
        rasterio.open(paths.raw_mask(2, 2004)) as b,
    ):
        # both reaches on the same river grid
        assert (a.transform.c - b.transform.c) % 30 == 0
        assert (a.transform.f - b.transform.f) % 30 == 0
    import pandas as pd

    log = pd.read_csv(paths.masks_log)
    assert len(log) == 6
    assert set(log.status) == {"ok", "no_imagery"}

    # Second run: nothing new to download.
    fake2 = FakeEE()
    [res2] = masks.run(p, [river], backend=fake2)
    assert res2.status in ("ok", "skipped")
    assert all(y == 2003 for y, _ in fake2.calls)  # only the no-imagery year retried


def test_changed_settings_are_stale(mask_project, synthetic_project):
    from rivmapper_v2 import masks

    p = mask_project
    masks.run(p, [p.river("Synth")], backend=FakeEE())
    cfg = synthetic_project
    cfg.write_text(cfg.read_text().replace("dswe_level: 2", "dswe_level: 3"))
    p2 = Project.load(cfg)
    [res] = masks.run(p2, [p2.river("Synth")], backend=FakeEE())
    assert res.status == "stale"


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


def test_default_is_single_year(synthetic_project):
    from rivmapper_v2 import masks

    cfg = synthetic_project
    cfg.write_text(
        cfg.read_text()
        + "masks:\n  year_start: 2002\n  year_end: 2002\n  dswe_level: 2\n"
        "  reach_end: 1\n"
    )
    p = Project.load(cfg)
    run_pipeline(p, steps=["sword", "reaches"])
    fake = FakeEE()
    [res] = masks.run(p, [p.river("Synth")], backend=fake)
    assert fake.calls == [(2002, 0)]  # cloudy year is not widened
    import pandas as pd

    log = pd.read_csv(p.river("Synth").paths.masks_log)
    assert log.status.tolist() == ["flagged"]
    assert res.status == "ok"
