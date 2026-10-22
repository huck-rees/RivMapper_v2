import subprocess
import sys
import zipfile

import pytest

from rivmapper_v2 import data
from rivmapper_v2.cli import main


def test_registry_has_sword():
    ds = data.get_dataset("sword")
    assert ds.version == "v17b"
    assert ds.hash.startswith("md5:")
    assert "oc_sword_reaches_v17b.gpkg" in ds.expected_files
    assert len(ds.expected_files) == 12
    with pytest.raises(KeyError, match="v99"):
        data.get_dataset("sword", "v99")


def test_datasets_dir_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv(data.ENV_VAR, str(tmp_path / "shared"))
    assert data.datasets_dir() == (tmp_path / "shared").resolve()
    monkeypatch.delenv(data.ENV_VAR)
    root = data.repo_root()
    if root is not None:  # running from a git checkout
        assert data.datasets_dir() == root / "data"


@pytest.fixture
def fake_download(tmp_path, monkeypatch):
    """Make pooch.retrieve 'download' a local zip shaped like the SWORD one."""
    ds = data.get_dataset("sword")
    src = tmp_path / "src.zip"
    with zipfile.ZipFile(src, "w") as zf:
        for f in ds.expected_files:
            zf.writestr(f"SWORD_v17b_gpkg/gpkg/{f}", f"fake {f}")
        zf.writestr("__MACOSX/._af_sword_nodes_v17b.gpkg", "junk")
        zf.writestr("SWORD_v17b_gpkg/README.txt", "readme")
    calls = []

    def retrieve(url, known_hash, fname, path, progressbar):
        calls.append(url)
        dest = path / fname
        dest.write_bytes(src.read_bytes())
        return str(dest)

    import pooch

    monkeypatch.setattr(pooch, "retrieve", retrieve)
    return calls


def test_fetch_unpacks_flat_and_verifies(tmp_path, fake_download):
    dest = data.fetch("sword", folder=tmp_path / "SWORD")
    names = sorted(p.name for p in dest.iterdir())
    assert names == sorted([*data.get_dataset("sword").expected_files, data.MANIFEST])
    assert data.verify("sword", folder=dest) == []
    # Already installed: no second download.
    data.fetch("sword", folder=dest)
    assert len(fake_download) == 1
    # Tampering is detected.
    (dest / "oc_sword_nodes_v17b.gpkg").write_text("edited")
    assert data.verify("sword", folder=dest) == [
        "changed since download: oc_sword_nodes_v17b.gpkg"
    ]


def test_sword_step_fetches_when_missing(synthetic_project, fake_download):
    import shutil

    from rivmapper_v2 import Project, run_pipeline

    shutil.rmtree(synthetic_project.parent / "SWORD")
    # The fake archive is not real SWORD, so the step fails *after* fetching.
    t = run_pipeline(Project.load(synthetic_project), steps=["sword"])
    assert len(fake_download) == 1
    assert t["status"].tolist() == ["failed"]


def test_data_cli(tmp_path, capsys, fake_download):
    assert main(["data", "list", "--dir", str(tmp_path / "S")]) == 0
    assert "not downloaded" in capsys.readouterr().out
    assert main(["data", "fetch", "sword", "--dir", str(tmp_path / "S")]) == 0
    assert main(["data", "verify", "sword", "--dir", str(tmp_path / "S")]) == 0
    assert main(["data", "cite", "sword"]) == 0
    assert "Altenau" in capsys.readouterr().out
    assert main(["data", "fetch"]) == 2


def test_version_does_not_import_heavy_libraries():
    code = (
        "import sys\n"
        "from rivmapper_v2.cli import build_parser\n"
        "try:\n    build_parser().parse_args(['--version'])\n"
        "except SystemExit:\n    pass\n"
        "heavy = [m for m in ('geopandas', 'pandas', 'shapely', 'pydantic') "
        "if m in sys.modules]\n"
        "print('HEAVY=' + ','.join(heavy))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "HEAVY=\n" in out.stdout, f"--version imported: {out.stdout}"
