"""Convert an old master CSV into ``rivmapper.yaml`` + ``rivers.csv``.

Columns whose value is identical for every river become YAML defaults;
columns that vary stay in the rivers table as per-river overrides.
Pipeline-derived columns (e.g. ``traced_reach_ids``) are dropped, since
outputs and run records now hold them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import (
    LEGACY_COLUMN_ALIASES,
    OVERRIDE_COLUMNS,
    RIVER_ID_COLUMNS,
    _clean_cell,
)
from .templates import render_config


@dataclass
class MigrationReport:
    config_path: Path
    rivers_path: Path
    n_rivers: int
    data_root: str
    to_yaml: dict[str, object] = field(default_factory=dict)
    per_river: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Wrote {self.config_path} and {self.rivers_path} ({self.n_rivers} rivers)",
            f"  data_root: {self.data_root}",
        ]
        if self.to_yaml:
            lines.append("  Same for every river → YAML defaults:")
            lines += [f"    {k}: {v}" for k, v in self.to_yaml.items()]
        if self.per_river:
            lines.append(f"  Kept as per-river columns: {', '.join(self.per_river)}")
        if self.dropped:
            lines.append(f"  Dropped (not pipeline inputs): {', '.join(self.dropped)}")
        return "\n".join(lines)


def _as_number(text: str):
    try:
        f = float(text)
    except ValueError:
        return text
    return int(f) if f.is_integer() else f


def migrate_master_csv(
    master_csv: str | Path, out_dir: str | Path, force: bool = False
) -> MigrationReport:
    """Split ``master_csv`` into ``rivmapper.yaml`` + ``rivers.csv`` in ``out_dir``."""
    master_csv, out_dir = Path(master_csv), Path(out_dir)
    config_path, rivers_path = out_dir / "rivmapper.yaml", out_dir / "rivers.csv"
    for p in (config_path, rivers_path):
        if p.exists() and not force:
            raise FileExistsError(f"{p} already exists (use --force to replace)")

    df = pd.read_csv(master_csv, dtype=str).dropna(how="all")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(
        columns={k: v for k, v in LEGACY_COLUMN_ALIASES.items() if k in df.columns}
    )
    df = df.map(_clean_cell)
    df = df.dropna(subset=["river_name"]).reset_index(drop=True)

    missing = [c for c in RIVER_ID_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{master_csv} is missing required column(s): {missing}")

    # data_root: one value → YAML; several → per-river column.
    data_root = "./data"
    keep = list(RIVER_ID_COLUMNS)
    if "data_root" in df.columns:
        roots = df["data_root"].dropna().unique()
        if len(roots) == 1 and df["data_root"].notna().all():
            data_root = str(roots[0]).replace("\\", "/")
        else:
            keep.append("data_root")
            data_root = str(roots[0]).replace("\\", "/") if len(roots) else data_root

    sections: dict[str, dict] = {"reaches": {}, "masks": {}}
    to_yaml, per_river = {}, []
    for col in df.columns:
        if col not in OVERRIDE_COLUMNS:
            continue
        values = df[col]
        uniq = values.dropna().unique()
        if len(uniq) == 1 and values.notna().all():
            val = _as_number(uniq[0])
            sections[OVERRIDE_COLUMNS[col]][col] = val
            to_yaml[col] = val
        elif len(uniq) == 0:
            continue
        else:
            keep.append(col)
            per_river.append(col)

    dropped = [
        c
        for c in df.columns
        if c not in keep and c not in OVERRIDE_COLUMNS and c != "data_root"
    ]
    if "notes" in df.columns:
        keep.append("notes")
        dropped.remove("notes")

    # Old layout kept SWORD in {working_directory}/SWORD; keep using it.
    sword_dir = None
    old_sword = Path(data_root) / "SWORD"
    if old_sword.is_dir() and any(old_sword.glob("*_sword_reaches_*.gpkg")):
        sword_dir = old_sword.as_posix()

    out_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        render_config(
            data_root=data_root,
            sword_dir=sword_dir,
            rivers_table=rivers_path.name,
            reaches=sections["reaches"],
            masks=sections["masks"],
        ),
        encoding="utf-8",
    )
    df[keep].to_csv(rivers_path, index=False)
    return MigrationReport(
        config_path, rivers_path, len(df), data_root, to_yaml, per_river, dropped
    )
