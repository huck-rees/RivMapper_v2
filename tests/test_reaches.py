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
        "geometry",
    ]
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
        boundary_method=method,
    )
    lines, polys, diag = build_reaches(_hairpin_sword(), s)
    assert diag["boundary_method"] == method
    if method == "nearest_centerline":
        assert diag["overlap_km2"] < 1e-6
        # every reach polygon contains (most of) its own centreline
        for line, poly in zip(lines.geometry, polys.geometry):
            assert line.intersection(poly).length > 0.95 * line.length
    else:
        assert diag["overlap_km2"] > 1  # buffers spill across the hairpin
