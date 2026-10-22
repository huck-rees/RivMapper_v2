import numpy as np
import pytest
from shapely.geometry import LineString

from rivmapper_v2 import Project, run_pipeline
from rivmapper_v2.config import ReachSettings
from rivmapper_v2.reaches import (
    compute_split_distances,
    sample_facc,
    split_at_distances,
)


def test_sample_facc_picks_containing_segment():
    cum_end, facc = np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 3.0])
    assert [sample_facc(d, cum_end, facc) for d in (0, 10, 10.1, 30, 99)] == [
        1,
        1,
        2,
        3,
        3,
    ]


def test_split_distances_follow_regression():
    s = ReachSettings(len_slope=1.0, len_intercept=0.0, length_modulator=2.0)
    # Constant facc 50 → reach length 100 m along a 1000 m line.
    starts = compute_split_distances(np.array([1000.0]), np.array([50.0]), s)
    assert starts == [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]


def test_split_at_distances_preserves_length():
    line = LineString([(0, 0), (500, 0), (500, 500)])
    pieces = split_at_distances(line, [100.0, 600.0])
    assert [round(p.length, 6) for p in pieces] == [100.0, 500.0, 400.0]
    assert pieces[1].coords[0] == (100.0, 0.0) and pieces[1].coords[-1] == (
        500.0,
        100.0,
    )


def test_reaches_step_on_synthetic_river(synthetic_project):
    p = Project.load(synthetic_project)
    t = run_pipeline(p)
    assert t["status"].tolist() == ["ok", "ok"]
    import geopandas as gpd

    r = p.river("Synth")
    lines = gpd.read_file(r.paths.reach_lines)
    polys = gpd.read_file(r.paths.reach_polygons)
    assert lines["ds_dist"].iloc[0] == 0
    assert np.allclose(lines["ds_dist"].iloc[1:], lines["reach_len"].cumsum().iloc[:-1])
    assert list(lines.columns) == [
        "ds_order",
        "reach_len",
        "ds_dist",
        "reach_ar",
        "us_drain_ar",
        "valley_len",
        "valley_dist",
        "sinuosity",
        "width_ref",
        "geometry",
    ]
    # constant width 100 m → every reach but the last is 80 widths of valley
    assert np.allclose(lines["valley_len"].iloc[:-1], 8000.0)
    g = list(polys.geometry)
    overlap = sum(g[i].intersection(g[i + 1]).area for i in range(len(g) - 1))
    assert overlap < 1.0  # m²; cut lines separate neighbours


def test_missing_buffer_width_is_clear(synthetic_project):
    cfg = synthetic_project
    cfg.write_text(cfg.read_text().replace("buffer_width: 0.5", "buffer_width: null"))
    t = run_pipeline(Project.load(cfg))
    assert t["status"].tolist() == ["ok", "failed"]
    assert "buffer_width" in t["message"].iloc[1]


def test_unknown_step(synthetic_project):
    with pytest.raises(KeyError, match="masks"):
        run_pipeline(Project.load(synthetic_project), steps=["masks"])


def _hairpin_sword():
    """SWORD-like reaches for a channel that runs east, then loops back west."""
    import geopandas as gpd

    coords = [(x, 0.0) for x in range(0, 40_001, 1000)]
    coords += [
        (40_000 + 3000 * np.sin(t), 3000 - 3000 * np.cos(t))
        for t in np.linspace(0.1, np.pi, 20)
    ]
    coords += [(x, 6000.0) for x in range(40_000, -1, -1000)]
    line = LineString(coords)
    pieces = split_at_distances(line, list(np.arange(10_000, line.length, 10_000)))
    gdf = gpd.GeoDataFrame(
        {"reach_order": range(1, len(pieces) + 1), "facc": 100.0},
        geometry=pieces,
        crs="EPSG:32633",
    )
    return gdf


@pytest.mark.parametrize("method", ["cut_lines", "nearest_centerline"])
def test_boundary_methods_on_hairpin(method):
    from rivmapper_v2.reaches import build_reaches

    s = ReachSettings(
        buffer_width=0.5,
        len_slope=1.0,
        len_intercept=0.0,
        length_modulator=100.0,
        length_method="drainage_regression",
        boundary_method=method,
    )
    built = build_reaches(_hairpin_sword(), s)
    lines, polys, diag = built.lines, built.polygons, built.diagnostics
    assert diag["boundary_method"] == method
    if method == "nearest_centerline":
        assert diag["overlap_km2"] < 1e-6
        # every reach polygon contains (most of) its own centreline
        for line, poly in zip(lines.geometry, polys.geometry):
            assert line.intersection(poly).length > 0.95 * line.length
    else:
        assert diag["overlap_km2"] > 1  # buffers spill across the hairpin


# ── valley-length reaches ───────────────────────────────────────────────
def _meandering_river(n_bends=40, wavelength=1500.0, amp=1200.0, width=120.0):
    """Sine-wave channel along a straight east-west valley, with SWORD-like
    nodes every ~200 m along the channel and drainage area growing 10x."""
    import geopandas as gpd

    xv = np.linspace(0, n_bends * wavelength, n_bends * 400)
    yv = amp * np.sin(2 * np.pi * xv / wavelength)
    chan = LineString(np.column_stack([xv, yv]))
    d = np.arange(0, chan.length, 200.0)
    pts = [chan.interpolate(v) for v in d]
    area = np.geomspace(1_000, 10_000, len(d))
    nodes = gpd.GeoDataFrame(
        {
            "node_order": np.arange(1, len(d) + 1),
            "width": width * (area / 1_000) ** 0.5,
            "facc": area,
        },
        geometry=pts,
        crs="EPSG:32633",
    )
    pieces = split_at_distances(chan, list(np.arange(10_000, chan.length, 10_000)))
    sword = gpd.GeoDataFrame(
        {"reach_order": range(1, len(pieces) + 1), "facc": 1000.0},
        geometry=pieces,
        crs="EPSG:32633",
    )
    return sword, nodes, xv[-1]


def test_width_model_recovers_power_law():
    from rivmapper_v2.valley import fit_width_model

    rng = np.random.default_rng(0)
    a = np.geomspace(100, 100_000, 5000)
    w = 5.0 * a**0.45 * np.exp(rng.normal(0, 0.2, a.size))
    m = fit_width_model(w, a)
    assert m.method == "power_law"
    assert abs(m.exponent - 0.45) < 0.03
    assert abs(m(10_000) / (5.0 * 10_000**0.45) - 1) < 0.1
    flat = fit_width_model(np.full(50, 80.0), np.full(50, 2000.0))
    assert flat.method == "constant" and flat(5) == pytest.approx(80.0)


def test_valley_line_removes_meanders():
    from rivmapper_v2.valley import valley_line

    _, nodes, valley_true = _meandering_river()
    x, y = nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()
    vl = valley_line(x, y, nodes["width"].to_numpy())
    assert np.all(np.diff(vl.valley_dist) >= 0)
    assert vl.channel_dist[-1] / valley_true > 2.0  # strongly sinuous channel
    assert abs(vl.valley_dist[-1] / valley_true - 1) < 0.05
    # bends smoothed out away from the ends (ends are reflection-padded and
    # less reliable, as documented)
    inner = (vl.valley_dist > 0.1 * vl.valley_dist[-1]) & (
        vl.valley_dist < 0.9 * vl.valley_dist[-1]
    )
    assert np.abs(vl.y[inner]).max() < 0.1 * 1200


def test_reach_breaks_spacing_and_merge():
    from rivmapper_v2.valley import reach_breaks

    v = np.arange(0, 10_050, 50.0)
    w = np.full(v.size, 10.0)  # k*W = 1000 m
    starts = reach_breaks(v, w, 100.0, final_merge_fraction=0.5)
    assert np.allclose(np.diff(starts), 1000.0)
    assert len(starts) == 10
    # 10 000 m total → a 300 m remainder would be merged
    v2 = np.arange(0, 10_350, 50.0)
    assert len(reach_breaks(v2, np.full(v2.size, 10.0), 100.0)) == 10


def test_valley_reaches_on_meandering_river():
    from rivmapper_v2.reaches import build_reaches

    sword, nodes, valley_true = _meandering_river()
    s = ReachSettings(buffer_width=0.15, widths_per_reach=80)
    b = build_reaches(sword, s, nodes)
    d = b.diagnostics
    assert d["length_method"] == "valley_widths"
    assert d["overlap_km2"] < 1e-6
    lines = b.lines
    # each reach is ~80 modelled widths of valley, and grows downstream
    nw = lines["valley_len"] / lines["width_ref"]
    assert nw.iloc[:-1].between(70, 90).all()
    assert lines["valley_len"].is_monotonic_increasing
    assert (lines["sinuosity"] > 1.5).all()
    assert abs(lines["valley_len"].sum() / valley_true - 1) < 0.05
    assert b.valley is not None and len(b.valley) == 1


def test_valley_method_needs_nodes():
    from rivmapper_v2.reaches import build_reaches

    sword, _, _ = _meandering_river(n_bends=10)
    with pytest.raises(RuntimeError, match="nodes"):
        build_reaches(sword, ReachSettings(buffer_width=0.15))
