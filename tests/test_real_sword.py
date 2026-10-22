"""End-to-end on the real Beni (needs RIVMAPPER_TEST_SWORD_DIR)."""

import geopandas as gpd

from rivmapper_v2 import Project, run_pipeline

from .conftest import write_project


def test_beni_end_to_end(tmp_path, real_sword_dir):
    cfg = write_project(
        tmp_path,
        """
        river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id
        Beni,sa,62266500601,62266100011
    """,
    )
    cfg.write_text(
        cfg.read_text().replace("dir: ./SWORD", f"dir: {real_sword_dir.as_posix()}")
    )
    t = run_pipeline(Project.load(cfg), steps=["sword", "reaches"])
    assert t["status"].tolist() == ["ok", "ok"]
    r = Project.load(cfg).river("Beni")
    assert len(gpd.read_file(r.paths.sword_reaches)) == 73
    assert len(gpd.read_file(r.paths.sword_nodes)) == 4890
    lines = gpd.read_file(r.paths.reach_lines)
    # valley_widths (default): 100 widths of valley per reach, growing downstream
    nw = lines["valley_len"] / lines["width_ref"]
    assert nw.iloc[:-1].between(99, 101).all()
    assert r.paths.valley_line.exists()

    # drainage_regression reproduces the v2 notebook result (45 reaches)
    cfg.write_text(
        cfg.read_text().replace(
            "reaches:\n", "reaches:\n  length_method: drainage_regression\n"
        )
    )
    t = run_pipeline(Project.load(cfg), steps=["reaches"])
    assert t["status"].tolist() == ["stale"]
    p = Project.load(cfg)
    p.rivers = [x.model_copy(update={"overwrite": True}) for x in p.rivers]
    run_pipeline(p, steps=["reaches"])
    assert len(gpd.read_file(r.paths.reach_lines)) == 45
