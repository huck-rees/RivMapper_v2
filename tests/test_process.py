"""Phase 0 of mask processing: settings, synthetic rivers, reach windows."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from rivmapper_v2 import Project, run_pipeline
from rivmapper_v2.config import ConfigError, ProcessSettings
from rivmapper_v2.gee import COUNT_BANDS, NODATA
from rivmapper_v2.process.synthetic import SCENARIOS, make_river
from rivmapper_v2.process.windows import (
    RiverLayout,
    offset,
    read_reach_stack,
    read_window_year,
    union_grid,
)

YEARS = list(range(1995, 2003))


@pytest.fixture(scope="module")
def small_river():
    return make_river(scenarios=SCENARIOS, shape=(120, 180), width_px=6, seed=3)


@pytest.fixture
def synth_project(tmp_path, small_river):
    cfg = small_river.write_project(tmp_path)
    return Project.load(cfg)


# ── settings ────────────────────────────────────────────────────────────
def test_process_settings_defaults_and_coercion():
    s = ProcessSettings()
    assert s.envelope_threshold == "calibrate" and s.states == 3
    assert (
        ProcessSettings.model_validate({"envelope_threshold": "0.6"}).envelope_threshold
        == 0.6
    )
    assert ProcessSettings.model_validate({"states": "2"}).states == 2
    with pytest.raises(ValueError):
        ProcessSettings(envelope_threshold=1.5)
    with pytest.raises(ValueError):
        ProcessSettings(states=4)


def test_process_override_per_river(tmp_path, small_river):
    cfg = small_river.write_project(tmp_path)
    (tmp_path / "rivers.csv").write_text(
        "river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id,"
        "envelope_threshold,dead_end_widths\n"
        "Synth,oc,51000000011,51000000021,0.4,5\n"
    )
    r = Project.load(cfg).river("Synth")
    assert r.process.envelope_threshold == 0.4 and r.process.dead_end_widths == 5
    (tmp_path / "rivers.csv").write_text(
        "river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id,"
        "envelope_threshold\nSynth,oc,51000000011,51000000021,2\n"
    )
    with pytest.raises(ConfigError, match="envelope_threshold"):
        Project.load(cfg)


def test_template_has_process_section():
    from rivmapper_v2.templates import render_config

    d = yaml.safe_load(render_config())
    assert ProcessSettings.model_validate(d["process"]) == ProcessSettings()


# ── synthetic rivers ────────────────────────────────────────────────────
def test_synthetic_is_deterministic():
    a = make_river(shape=(60, 90), years=YEARS, seed=7)
    b = make_river(shape=(60, 90), years=YEARS, seed=7)
    c = make_river(shape=(60, 90), years=YEARS, seed=8)
    assert all(np.array_equal(a.counts[k], b.counts[k]) for k in COUNT_BANDS)
    assert not np.array_equal(a.counts["n_obs"], c.counts["n_obs"])


def test_synthetic_truth_and_counts_are_consistent(small_river):
    r = small_river
    T = len(r.years)
    assert r.truth.shape == (T, 120, 180) and r.truth.any(axis=(1, 2)).all()
    assert not (r.truth_core & ~r.truth).any()  # core inside bankfull
    assert not (r.truth & ~r.water).any()  # every channel pixel is water
    n = r.counts["n_obs"]
    for k in (1, 2, 3):
        assert (r.counts[f"n_wet_{k}"] <= r.counts[f"n_wet_{k + 1}"]).all()
    assert (r.counts["n_wet_4"] <= n).all()
    # empty years stay empty; others hit the Ok Tedi medians roughly
    for y in (1984, 1985, 1986, 2012):
        assert n[r.years.index(y)].max() == 0
    for y, target in ((1995, 3), (2000, 6), (2022, 11)):
        assert abs(np.median(n[r.years.index(y)]) - target) <= 2
    # scenario events recorded and placed in the record
    assert set(r.events) == set(SCENARIOS)


def test_synthetic_distractors_are_not_channel(small_river):
    r = small_river
    t_f = r.years.index(r.events["flood"])
    flood_only = r.water[t_f] & ~r.truth[t_f]
    assert flood_only.sum() > 0
    # the cutoff leaves an oxbow: water that is not channel the year after
    t_c = r.years.index(r.events["cutoff"]) + 1
    assert (r.water[t_c] & ~r.truth[t_c]).sum() > 0
    # channel water is detected far more often than dry land
    i = r.years.index(2022)
    n = r.counts["n_obs"][i].astype(float)
    f = np.where(n > 0, r.counts["n_wet_2"][i] / np.maximum(n, 1), np.nan)
    assert np.nanmean(f[r.truth_core[i]]) > 0.6
    assert np.nanmean(f[~r.water[i]]) < 0.05


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError, match="unknown scenario"):
        make_river(scenarios=("volcano",))


# ── round trip through the raw-file format, and windows ─────────────────
def test_synthetic_round_trip_through_raw_files(synth_project, small_river):
    r = small_river
    river = synth_project.river("Synth")
    layout = RiverLayout.from_paths(river.paths)
    assert layout.ds_orders == [1, 2, 3]
    assert layout.width_px(2) == pytest.approx(6)
    years = [1999, 2000, 2022]
    idx = [r.years.index(y) for y in years]
    for o in layout.ds_orders:
        grid, stack = read_reach_stack(river.paths, o, years, COUNT_BANDS)
        rr, cc = offset(grid, r.grid)
        inside = layout.owner(grid, [o]) == o
        for b in COUNT_BANDS:
            want = r.counts[b][idx, rr : rr + grid.height, cc : cc + grid.width]
            assert np.array_equal(stack[b][:, inside], want[:, inside])
            assert (stack[b][:, ~inside] == NODATA).all()


def test_windows_tile_the_river_exactly(synth_project, small_river):
    r = small_river
    river = synth_project.river("Synth")
    layout = RiverLayout.from_paths(river.paths)
    full = union_grid(layout.grids.values())
    owner = layout.owner(full)
    centre_claims = np.zeros(owner.shape, int)
    for w in layout.windows():
        mosaic = read_window_year(river.paths, w, 2000, ("n_obs", "n_wet_2"))
        own = layout.owner(w.grid, w.members)
        assert np.array_equal(mosaic["n_obs"] != NODATA, own > 0)
        rr, cc = offset(w.grid, full)
        centre_claims[rr : rr + w.grid.height, cc : cc + w.grid.width] += (
            own == w.center
        )
        # mosaic values equal the synthetic counts
        r0, c0 = offset(w.grid, r.grid)
        want = r.counts["n_wet_2"][r.years.index(2000)][
            r0 : r0 + w.grid.height, c0 : c0 + w.grid.width
        ]
        assert np.array_equal(mosaic["n_wet_2"][own > 0], want[own > 0])
    assert np.array_equal(centre_claims, (owner > 0).astype(int))
    w1, w2, w3 = layout.windows()
    assert w1.upstream_cut is None and w3.downstream_cut is None
    assert w1.downstream_cut is not None and w1.members == (1, 2)
    assert w2.members == (1, 2, 3)
    cut = layout.rasterize_line(w1.downstream_cut, w1.grid)
    assert cut.any() and cut.shape == (w1.grid.height, w1.grid.width)


# ── step scaffold ───────────────────────────────────────────────────────
def test_process_step_scaffold(synth_project):
    table = run_pipeline(synth_project, steps=["process_masks"])
    row = table.iloc[0]
    assert row["status"] == "skipped" and "phase 0" in row["message"]
    rec = yaml.safe_load(
        synth_project.river("Synth").paths.run_record("process_masks").read_text()
    )
    assert rec["summary"]["n_windows"] == 3
    assert rec["parameters"]["envelope_threshold"] == "calibrate"


def test_process_step_reports_missing_raw(synth_project):
    river = synth_project.river("Synth")
    river.paths.raw_mask(2, 2001).unlink()
    table = run_pipeline(synth_project, steps=["process_masks"])
    row = table.iloc[0]
    assert row["status"] == "failed" and "extract_masks" in row["message"]


# ── real data (Ok Tedi raw counts) ──────────────────────────────────────
@pytest.fixture
def oktedi_root() -> Path:
    """Folder containing OkTedi/ outputs; set RIVMAPPER_TEST_OKTEDI_DIR."""
    d = os.environ.get("RIVMAPPER_TEST_OKTEDI_DIR")
    if not d or not (Path(d) / "OkTedi" / "masks" / "raw").exists():
        pytest.skip("set RIVMAPPER_TEST_OKTEDI_DIR to run tests on Ok Tedi data")
    return Path(d)


def test_oktedi_windows_tile_without_gaps_or_overlap(oktedi_root):
    from rivmapper_v2.paths import RiverPaths
    from rivmapper_v2.process.windows import read_raw

    paths = RiverPaths(oktedi_root, "OkTedi")
    layout = RiverLayout.from_paths(paths)
    full = union_grid(layout.grids.values())
    claims = np.zeros((full.height, full.width), np.uint8)
    for o in layout.ds_orders:
        g, a = read_raw(paths.raw_mask(o, 2000), ["n_obs"])
        inside = a["n_obs"] != NODATA
        assert np.array_equal(inside, layout.owner(g, [o]) == o)
        rr, cc = offset(g, full)
        claims[rr : rr + g.height, cc : cc + g.width] += inside
    assert claims.max() == 1
    assert np.array_equal(claims > 0, layout.owner(full) > 0)
