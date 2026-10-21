import geopandas as gpd
import pytest

from rivmapper_v2 import Project, run_pipeline
from rivmapper_v2.sword import trace_reaches

from .conftest import CHAIN, SIDE


@pytest.fixture
def reaches(synthetic_project):
    d = synthetic_project.parent / "SWORD"
    g = gpd.read_file(d / "sa_sword_reaches_v17b.gpkg")
    return g[g["type"] == 1]


def test_trace_full_chain(reaches):
    assert trace_reaches(CHAIN[0], CHAIN[-1], reaches) == CHAIN


def test_trace_from_tributary(reaches):
    assert trace_reaches(SIDE, CHAIN[-1], reaches) == [SIDE, *CHAIN[2:]]


def test_swapped_endpoints(reaches):
    with pytest.raises(RuntimeError, match="swapped"):
        trace_reaches(CHAIN[-1], CHAIN[0], reaches)


def test_missing_endpoint(reaches):
    with pytest.raises(RuntimeError, match="not a river reach"):
        trace_reaches(62200000993, CHAIN[-1], reaches)  # lake reach, filtered out


def test_ds_not_on_path(reaches):
    with pytest.raises(RuntimeError, match="outlet"):
        trace_reaches(
            CHAIN[3],
            SIDE,
            reaches.assign(
                dist_out=reaches["dist_out"].where(reaches["reach_id"] != SIDE, 0.0)
            ),
        )


def test_sword_step_outputs_and_skip(synthetic_project):
    p = Project.load(synthetic_project)
    t = run_pipeline(p, steps=["sword"])
    assert t["status"].tolist() == ["ok"]
    r = p.river("Synth")
    nodes = gpd.read_file(r.paths.sword_nodes)
    assert nodes["node_order"].tolist() == list(range(1, len(nodes) + 1))
    assert nodes["reach_order"].is_monotonic_increasing
    assert r.paths.run_record("sword").exists()
    # Second run: nothing to do.
    assert run_pipeline(p, steps=["sword"])["status"].tolist() == ["skipped"]


def test_changed_settings_reported_stale(synthetic_project):
    run_pipeline(Project.load(synthetic_project), steps=["sword"])
    csv = synthetic_project.parent / "rivers.csv"
    csv.write_text(csv.read_text().replace(str(CHAIN[-1]), str(CHAIN[-2])))
    t = run_pipeline(Project.load(synthetic_project), steps=["sword", "reaches"])
    assert t["status"].tolist() == ["stale", "blocked"]
