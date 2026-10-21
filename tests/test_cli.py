from rivmapper_v2.cli import main

from .conftest import CHAIN


def test_init_then_validate(tmp_path, capsys):
    assert main(["init", str(tmp_path)]) == 0
    assert main(["validate", str(tmp_path / "rivmapper.yaml")]) == 0
    assert "buffer_width is unset" in capsys.readouterr().out


def test_validate_reports_errors(tmp_path, capsys):
    main(["init", str(tmp_path)])
    (tmp_path / "rivers.csv").write_text("river_name,sword_cont_abb\nX,sa\n")
    assert main(["validate", str(tmp_path / "rivmapper.yaml")]) == 2
    assert "missing column" in capsys.readouterr().err


def test_migrate_old_master_csv(tmp_path, capsys):
    master = tmp_path / "old_master.csv"
    master.write_text(
        "river_name,working_directory,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id,"
        "reach_length_modulator,buffer_width,DSWE_threshold_level,year_start,year_end,"
        "traced_reach_ids\n"
        f'A,E:\\Data,sa,{CHAIN[0]},{CHAIN[-1]},1,0.5,2,1985,2024,"1,2"\n'
        f'B,E:\\Data,sa,{CHAIN[1]},{CHAIN[-1]},2,0.5,2,2000,2024,"3"\n'
    )
    out = tmp_path / "proj"
    assert main(["migrate", str(master), "--out", str(out)]) == 0
    report = capsys.readouterr().out
    assert "traced_reach_ids" in report
    yaml_text = (out / "rivmapper.yaml").read_text()
    assert "data_root: E:/Data" in yaml_text
    assert "buffer_width: 0.5" in yaml_text and "dswe_level: 2" in yaml_text
    header = (out / "rivers.csv").read_text().splitlines()[0].split(",")
    assert header == [
        "river_name",
        "sword_cont_abb",
        "us_sword_reach_id",
        "ds_sword_reach_id",
        "length_modulator",
        "year_start",
    ]
