"""Shared fixtures: tiny synthetic SWORD data and project files."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString

# A chain of 6 reaches flowing east along the equator-ish at lon -60,
# each ~11 km long, plus a side branch (reach 7) joining reach 3.
CHAIN = [62200000011 + 10 * i for i in range(6)]  # US → DS
SIDE = 62200000901


def make_sword(tmp: Path, continent: str = "sa", version: str = "v17b") -> Path:
    """Write minimal continental SWORD reaches + nodes geopackages."""
    sword_dir = tmp / "SWORD"
    sword_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, rid in enumerate(CHAIN):
        x0 = -60.0 + 0.1 * i
        dn = str(CHAIN[i + 1]) if i + 1 < len(CHAIN) else "0"
        rows.append(
            dict(
                reach_id=rid,
                rch_id_dn=dn,
                dist_out=float((len(CHAIN) - i) * 11_000),
                facc=1000.0 * (i + 1),
                type=1,
                geometry=LineString(
                    [(x0, -10.0), (x0 + 0.05, -10.01), (x0 + 0.1, -10.0)]
                ),
            )
        )
    # Side tributary flowing into reach index 2; also a lake reach (type 3).
    rows.append(
        dict(
            reach_id=SIDE,
            rch_id_dn=str(CHAIN[2]),
            dist_out=50_000.0,
            facc=10.0,
            type=1,
            geometry=LineString([(-59.8, -9.8), (-59.8, -10.0)]),
        )
    )
    rows.append(
        dict(
            reach_id=62200000993,
            rch_id_dn=str(CHAIN[0]),
            dist_out=90_000.0,
            facc=5.0,
            type=3,
            geometry=LineString([(-60.1, -10.0), (-60.0, -10.0)]),
        )
    )
    reaches = gpd.GeoDataFrame(rows, crs=4326)
    reaches.to_file(
        sword_dir / f"{continent}_sword_reaches_{version}.gpkg", driver="GPKG"
    )

    nodes = []
    for r in rows:
        line = r["geometry"]
        for k in range(5):
            nodes.append(
                dict(
                    node_id=r["reach_id"] * 1000 + k,
                    reach_id=r["reach_id"],
                    dist_out=r["dist_out"] - k * 2000.0,
                    width=100.0,
                    facc=r["facc"],
                    geometry=line.interpolate(k / 4, normalized=True),
                )
            )
    gpd.GeoDataFrame(nodes, crs=4326).to_file(
        sword_dir / f"{continent}_sword_nodes_{version}.gpkg", driver="GPKG"
    )
    return sword_dir


def write_project(tmp: Path, rivers_csv: str, yaml_extra: str = "") -> Path:
    """Write rivmapper.yaml + rivers.csv into ``tmp``; return the yaml path."""
    (tmp / "rivers.csv").write_text(
        textwrap.dedent(rivers_csv).lstrip(), encoding="utf-8"
    )
    cfg = tmp / "rivmapper.yaml"
    cfg.write_text(
        "data_root: ./data\nrivers_table: rivers.csv\n"
        "sword:\n  dir: ./SWORD\n"
        "reaches:\n  buffer_width: 0.5\n" + yaml_extra,
        encoding="utf-8",
    )
    return cfg


@pytest.fixture
def synthetic_project(tmp_path: Path) -> Path:
    make_sword(tmp_path)
    return write_project(
        tmp_path,
        f"""
        river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id
        Synth,sa,{CHAIN[0]},{CHAIN[-1]}
    """,
    )


@pytest.fixture
def real_sword_dir() -> Path:
    """Folder with real SWORD gpkgs; set RIVMAPPER_TEST_SWORD_DIR to enable."""
    d = os.environ.get("RIVMAPPER_TEST_SWORD_DIR")
    if not d or not Path(d).exists():
        pytest.skip("set RIVMAPPER_TEST_SWORD_DIR to run tests on real SWORD data")
    return Path(d)
