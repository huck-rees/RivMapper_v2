import pytest

from rivmapper_v2.config import ConfigError, Project

from .conftest import CHAIN, write_project

ROW = f"Synth,sa,{CHAIN[0]},{CHAIN[-1]}"


def test_defaults_and_relative_paths(tmp_path):
    cfg = write_project(
        tmp_path,
        f"river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id\n{ROW}\n",
    )
    project = Project.load(cfg)
    r = project.river("Synth")
    assert r.data_root == (tmp_path / "data").resolve()
    assert r.sword_dir == (tmp_path / "SWORD").resolve()
    assert r.reaches.buffer_width == 0.5
    assert r.reaches.len_slope == 0.0668
    assert r.paths.reach_polygons.name == "Synth_reach_polygons.gpkg"


def test_row_overrides_and_blank_cells(tmp_path):
    cfg = write_project(
        tmp_path,
        f"""
        river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id,buffer_width,year_start,year_end
        A,SA,{CHAIN[0]},{CHAIN[-1]},0.8,2000.0,2010
        B,sa,{CHAIN[0]},{CHAIN[-1]},,,
    """,
        "masks:\n  year_start: 1990\n  year_end: 2020\n",
    )
    p = Project.load(cfg)
    a, b = p.river("A"), p.river("B")
    assert a.continent == "sa"
    assert (a.reaches.buffer_width, a.masks.year_start, a.masks.year_end) == (
        0.8,
        2000,
        2010,
    )
    assert (b.reaches.buffer_width, b.masks.year_start, b.masks.year_end) == (
        0.5,
        1990,
        2020,
    )


def test_collects_all_row_errors(tmp_path):
    cfg = write_project(
        tmp_path,
        f"""
        river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id,year_start,year_end
        A,xx,{CHAIN[0]},{CHAIN[-1]},,
        B,sa,6.22E+10,{CHAIN[-1]},,
        C,sa,{CHAIN[0]},{CHAIN[-1]},2010,2000
        C,sa,{CHAIN[0]},{CHAIN[-1]},,
    """,
    )
    with pytest.raises(ConfigError) as e:
        Project.load(cfg)
    text = str(e.value)
    assert "duplicate river_name" in text
    assert "row 2 (A): sword_cont_abb" in text
    assert "scientific notation" in text
    assert "year_start (2010) is after year_end (2000)" in text


def test_unknown_column_rejected_but_notes_allowed(tmp_path):
    cfg = write_project(
        tmp_path,
        f"""
        river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id,notes,_mine,bufer_width
        A,sa,{CHAIN[0]},{CHAIN[-1]},hi,x,0.3
    """,
    )
    with pytest.raises(ConfigError, match="bufer_width"):
        Project.load(cfg)


def test_unknown_yaml_key_rejected(tmp_path):
    cfg = write_project(
        tmp_path,
        f"river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id\n{ROW}\n",
        "  smoothing_window: 2\n",
    )
    with pytest.raises(ConfigError, match="reaches.smoothing_window: unknown setting"):
        Project.load(cfg)


def test_legacy_column_names_accepted(tmp_path, caplog):
    cfg = write_project(
        tmp_path,
        f"""
        river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id,reach_length_modulator,DSWE_threshold_level
        A,sa,{CHAIN[0]},{CHAIN[-1]},2,3
    """,
    )
    r = Project.load(cfg).river("A")
    assert r.reaches.length_modulator == 2 and r.masks.dswe_level == 3
    assert "reach_length_modulator → length_modulator" in caplog.text


def test_select_unknown_river(synthetic_project):
    p = Project.load(synthetic_project)
    with pytest.raises(KeyError, match="Nope"):
        p.select(["Nope"])
