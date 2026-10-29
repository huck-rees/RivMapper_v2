"""Mask processing: settings, synthetic rivers, windows, HMM, channel, QA."""

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


# ── HMM ─────────────────────────────────────────────────────────────────
def _obs_from(states, theta, n_looks, rng):
    """(T, P, 3) multinomial looks for a (T, P) state array."""
    T, P = states.shape
    out = np.zeros((T, P, 3), np.int32)
    for s_ in range(theta.shape[0]):
        sel = states == s_
        out[sel] = rng.multinomial(n_looks, theta[s_], size=int(sel.sum()))
    return out


def test_hmm_recovers_states_and_fills_gaps():
    from rivmapper_v2.process import hmm

    rng = np.random.default_rng(0)
    T, P = 30, 4000
    truth = np.zeros((T, P), int)
    truth[:, : P // 4] = 1  # always channel
    truth[15:, P // 4 : P // 2] = 1  # channel from year 15 (avulsion)
    theta = np.array([[0.93, 0.03, 0.04], [0.12, 0.78, 0.10]])
    obs = _obs_from(truth, theta, 12, rng)
    obs[7] = 0  # a year with no clear looks at all
    params = hmm.fit(obs, n_states=2)
    assert params.names == ("dry", "channel")
    assert np.allclose(params.theta, theta, atol=0.03)
    p_ch, p_mg = hmm.posteriors(obs, params, chunk=1000)
    assert not p_mg.any()
    acc = ((p_ch >= 0.5) == truth).mean()
    assert acc > 0.99
    assert (p_ch[7, : P // 4] > 0.9).all()  # gap year filled from neighbours
    assert (p_ch[15:, P // 4 : P // 2] > 0.5).mean() > 0.97  # avulsion taken


def test_hmm_rejects_inconsistent_counts():
    from rivmapper_v2.process import hmm

    with pytest.raises(ValueError):
        hmm.observations([3], [2], [1])


def test_hmm_three_states_ordered_by_wetness():
    from rivmapper_v2.process import hmm

    rng = np.random.default_rng(1)
    truth = rng.choice(3, size=(1, 6000), p=[0.6, 0.2, 0.2]).repeat(20, axis=0)
    theta = np.array([[0.95, 0.02, 0.03], [0.5, 0.25, 0.25], [0.08, 0.85, 0.07]])
    params = hmm.fit(_obs_from(truth, theta, 10, rng), n_states=3)
    assert params.names == ("dry", "margin", "channel")
    assert np.all(np.diff(params.theta[:, 1]) > 0)


# ── channel extraction ──────────────────────────────────────────────────
def _channel_inputs(p_ch, n_obs=None, n_wet_3=None, W=4.0):
    from rivmapper_v2.process.channel import ChannelInputs

    shape = p_ch.shape
    n_obs = np.full(shape, 10) if n_obs is None else n_obs
    n_wet_3 = np.round(p_ch * 10).astype(int) if n_wet_3 is None else n_wet_3
    centerline = np.zeros(shape, bool)
    centerline[shape[0] // 2, :] = True
    inside = np.ones(shape, bool)
    inside[[0, -1], :] = False
    inside[:, [0, -1]] = False
    term = np.zeros(shape, np.int16)
    term[1:-1, 1:3] = 1
    term[1:-1, -3:-1] = 2
    return ChannelInputs(
        p_channel=p_ch,
        p_margin=np.zeros(shape),
        n_obs=n_obs,
        n_wet_3=n_wet_3,
        n_obs_prev=n_obs,
        inside=inside,
        centerline=centerline,
        prev_mask=None,
        width_px=W,
        terminals=term,
    )


def _straight(shape=(60, 200), half=2):
    p = np.zeros(shape)
    r = shape[0] // 2
    p[r - half : r + half, 1:-1] = 1.0
    return p


def test_channel_bridges_small_gap_but_not_observed_dry():
    from rivmapper_v2.process.channel import extract

    s = ProcessSettings(min_component_w2=1)
    p = _straight()
    p[:, 98:104] = 0.3  # weak water evidence in a 6 px gap (1.5 W)
    res = extract(_channel_inputs(p), s, 0.5)
    assert res.stats["through"] and res.channel[30, 100]
    # same gap, but observed dry in every clear look: never bridged
    n_wet = np.round(p * 10).astype(int)
    n_wet[:, 98:104] = 0
    res = extract(_channel_inputs(p, n_wet_3=n_wet), s, 0.5)
    assert not res.channel[30, 100]
    assert not res.stats["through"]


def test_channel_removes_disconnected_water_and_long_dead_ends():
    from rivmapper_v2.process.channel import extract

    s = ProcessSettings(min_component_w2=1)
    p = _straight()
    p[5:12, 40:80] = 1.0  # a pond off the channel
    p[30:55, 140:144] = 1.0  # a dead-end arm 25 px = 6 W long
    p[30:36, 60:64] = 1.0  # a short spur 1.5 W
    res = extract(_channel_inputs(p), s, 0.5)
    assert res.channel[30, 5] and res.channel[30, 195]
    assert not res.channel[8, 60]  # pond gone
    assert not res.channel[50, 142]  # long dead end gone
    assert res.channel[33, 62]  # short spur kept
    assert res.stats["dead_ends_removed"] == 1


def test_channel_keeps_anabranches():
    from rivmapper_v2.process.channel import extract

    s = ProcessSettings(min_component_w2=1)
    p = _straight()
    p[12:16, 60:140] = 1.0  # second thread
    p[12:32, 58:62] = 1.0
    p[12:32, 138:142] = 1.0
    res = extract(_channel_inputs(p), s, 0.5)
    assert res.channel[14, 100]


# ── QA ──────────────────────────────────────────────────────────────────
def test_year_status():
    from rivmapper_v2.process import qa

    years = list(range(2000, 2008))
    cov = [0, 0, 0.8, 0.3, 0.0, 0.9, 0.05, 0.0]
    st = qa.year_status(years, cov, max_infer_gap=2)
    assert st == {
        2000: "no_data",
        2001: "no_data",
        2002: "observed",
        2003: "sparse",
        2004: "inferred",
        2005: "observed",
        2006: "no_data",
        2007: "no_data",
    }


def test_outlier_scores_flag_big_jumps_only():
    from rivmapper_v2.process import qa

    area = [100, 103, 98, 101, 99, 160, 102, 97, 100, 60, 101]
    z = qa.outlier_scores(area, signed=True)
    assert z[5] > 3 and z[9] < -3
    assert (np.abs(np.delete(z, [5, 9])) < 3).all()
    assert np.isnan(qa.outlier_scores([1.0, np.nan])).all()


def test_memory_reworking_metrics():
    from rivmapper_v2.process import metrics

    a = np.zeros((10, 10), bool)
    a[:, :2] = True
    b = np.roll(a, 1, axis=1)
    out = metrics.memory_reworking([a, b, None], lags=(1, 2))
    assert out[1] == (0.5, 0.5)
    assert np.isnan(out[2][0])


# ── step, end to end ────────────────────────────────────────────────────
def test_process_step_end_to_end(tmp_path):
    from rivmapper_v2.process import metrics

    river = make_river(["cutoff"], shape=(120, 200), width_px=6, seed=2)
    project = Project.load(river.write_project(tmp_path))
    table = run_pipeline(project, steps=["process_masks"])
    assert table.iloc[0]["status"] == "ok", table.iloc[0]["message"]
    r = project.river("Synth")
    rec = yaml.safe_load(r.paths.run_record("process_masks").read_text())
    assert rec["summary"]["hmm"]["states"][-1] == "channel"
    assert "calibration" in rec["summary"]["envelope_threshold_note"]
    import pandas as pd
    import rasterio

    qa_table = pd.read_csv(r.paths.process_qa_table)
    assert len(qa_table) == 3 * len(river.years)
    assert r.paths.mask_qa_figure(2).exists()
    layout = RiverLayout.from_paths(r.paths)
    ious = []
    for t, y in enumerate(river.years):
        if set(qa_table[qa_table.year == y].status) != {"observed"}:
            continue
        pred = np.zeros(river.truth[t].shape, bool)
        for o in layout.ds_orders:
            with rasterio.open(r.paths.processed_mask(o, y)) as src:
                assert src.count == 4
                band = src.read(1)
                g = layout.grids[o]
            r0, c0 = offset(g, river.grid)
            sub = pred[r0 : r0 + g.height, c0 : c0 + g.width]
            sub |= band == 1
        ious.append(metrics.iou(pred, river.truth[t]))
    assert np.mean(ious) > 0.85
    # diagnostics: decision codes agree with the channel band
    from rivmapper_v2.process.channel import DECISIONS

    y = river.years[20]
    with rasterio.open(r.paths.processed_mask(2, y)) as src:
        ch = src.read(1)
    with rasterio.open(r.paths.diagnostic(2, y)) as src:
        d = src.read(1)
        assert "decision_codes" in src.tags()
    assert ((ch == 1) == np.isin(d, [1, 2, 3, 4])).all()
    assert ((ch == 255) == (d == 255)).all()
    assert set(np.unique(d)) <= set(DECISIONS) | {255}
    assert (d == 10).any()  # the cutoff's oxbow / noise was dropped somewhere
    with rasterio.open(r.paths.window_context(2)) as src:
        assert src.count == 3 and src.read(2).max() >= 2  # two terminal zones
    for col in ("seed_px", "not_connected_px", "envelope_threshold", "width_over_W"):
        assert col in qa_table.columns
    assert qa_table["width_over_W"].between(0.5, 3).mean() > 0.8
    # no discharge step run: stage columns empty, flagged
    assert qa_table["flags"].fillna("").str.contains("no_discharge").all()
    assert qa_table["q_scene_pctile"].isna().all()
    # second run: nothing to do
    again = run_pipeline(project, steps=["process_masks"])
    assert again.iloc[0]["status"] == "skipped"


def test_process_step_resumes_after_interruption(tmp_path, caplog):
    import logging

    import pandas as pd

    river = make_river([], shape=(100, 160), width_px=5, seed=4)
    project = Project.load(river.write_project(tmp_path))
    r = project.river("Synth")
    assert run_pipeline(project, steps=["process_masks"]).iloc[0]["status"] == "ok"
    full = pd.read_csv(r.paths.process_qa_table)
    # simulate a run killed while reach 3 was being written: reaches 1-2
    # finished (row checkpoints present), no run record or QA table yet
    from rivmapper_v2.process import step

    key = step._run_key(
        step._parameters(r, step._raw_fingerprint(r.paths.raw_mask(1, river.years[0])))
    )
    for o in (1, 2):
        part = full[full.ds_order == o]
        part.to_csv(
            step._ckpt(r.paths, f"r{o:03d}_rows", key).with_suffix(".csv"), index=False
        )
    r.paths.run_record("process_masks").unlink()
    r.paths.process_qa_table.unlink()
    r.paths.processed_mask(3, river.years[-1]).unlink()
    with caplog.at_level(logging.INFO, logger="rivmapper_v2.process.step"):
        res = run_pipeline(project, steps=["process_masks"]).iloc[0]
    assert res["status"] == "ok", res["message"]
    assert "resuming" in caplog.text
    assert caplog.text.count("already done") == 2
    again = pd.read_csv(r.paths.process_qa_table)
    pd.testing.assert_frame_equal(
        again[["ds_order", "year", "status", "channel_km2"]],
        full[["ds_order", "year", "status", "channel_km2"]],
    )
    assert not list(r.paths.model_dir.glob("*_rows_*.csv"))


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


def test_channel_crossing_a_wide_wet_patch_is_not_cut():
    """A wide wet patch lying on the channel (flooded bend, bar flats) is
    trimmed, but the channel through it is kept, so the river stays
    connected end to end and nothing upstream is pruned as a dead end."""
    from rivmapper_v2.process.channel import extract

    s = ProcessSettings(min_component_w2=1)
    p = _straight()
    p[12:52, 150:190] = 1.0  # wet patch 10 W wide, straddling the channel
    inp = _channel_inputs(p)
    inp.centerline[:] = False
    inp.centerline[5, :] = True  # SWORD line far off (migrated channel)
    inp.prev_mask = _straight() > 0.5  # last year's channel seeds instead
    res = extract(inp, s, 0.5)
    assert res.channel[30, 20] and res.channel[30, 170] and res.channel[30, 195]
    assert not res.channel[48, 170]  # far side of the patch removed
    assert res.stats["through"]


def test_channel_drops_floodplain_lattice_but_keeps_side_channel():
    """Water attached to the channel at two points (a loop) is dropped when
    it is wide and ragged (floodplain), kept when it looks like a channel."""
    from rivmapper_v2.process.channel import extract

    s = ProcessSettings(min_component_w2=1)
    # floodplain: a ragged wet patch 6-12 W wide, joined to the channel by
    # two narrow arms (so it is on a cycle, not a dead end)
    p = _straight(shape=(90, 200))
    r = 45
    # a band ~2 W wide (narrow enough to pass the wide-blob test), with
    # bulges: wider than a channel thread and uneven
    p[r - 24 : r - 16, 70:130] = 1.0
    p[r - 30 : r - 16, 85:95] = 1.0
    p[r - 30 : r - 16, 105:112] = 1.0
    p[r - 16 : r - 2, 72:76] = 1.0
    p[r - 16 : r - 2, 124:128] = 1.0
    inp = _channel_inputs(p, W=4.0)
    inp.centerline[:] = False
    inp.centerline[r, :] = True
    res = extract(inp, s, 0.5)
    assert res.channel[r, 5] and res.channel[r, 195] and res.channel[r, 100]
    assert not res.channel[r - 20, 100]  # floodplain patch dropped
    assert res.stats["links_main"] >= 1 and res.stats["floodplain_px"] > 0
    assert (res.decision == 15).any()


def test_network_selection_can_be_switched_off():
    from rivmapper_v2.process.channel import extract

    p = _straight()
    p[12:16, 60:140] = 1.0
    p[12:32, 58:62] = 1.0
    p[12:32, 138:142] = 1.0
    on = extract(_channel_inputs(p), ProcessSettings(min_component_w2=1), 0.5)
    off = extract(
        _channel_inputs(p),
        ProcessSettings(min_component_w2=1, network_selection=False),
        0.5,
    )
    assert on.channel[14, 100] and off.channel[14, 100]  # anabranch kept by both
    assert "links" in on.stats and "links" not in off.stats


def test_side_channel_on_last_years_channel_is_kept_with_looser_limits():
    """A loop a bit too wide for a new thread stays channel if it was channel
    last year (persistence), and is dropped without that history."""
    from rivmapper_v2.process.channel import extract

    s = ProcessSettings(min_component_w2=1)
    p = _straight(shape=(80, 200))
    r = 40
    p[r - 22 : r - 15, 60:140] = 1.0  # loop 7 px = 1.75 W wide
    p[r - 22 : r - 2, 58:62] = 1.0
    p[r - 22 : r - 2, 138:142] = 1.0
    inp = _channel_inputs(p, W=4.0)
    inp.centerline[:] = False
    inp.centerline[r, :] = True
    new = extract(inp, s, 0.5)
    assert not new.channel[r - 18, 100]  # new and too wide: dropped
    inp.prev_mask = p > 0.5  # it was channel last year
    kept = extract(inp, s, 0.5)
    assert kept.channel[r - 18, 100] and kept.channel[r, 100]
    off = extract(inp, ProcessSettings(min_component_w2=1, thread_persistence=False), 0.5)
    assert not off.channel[r - 18, 100]
