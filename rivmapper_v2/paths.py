"""Standard file layout under a project's ``data_root``.

Every step reads and writes through ``RiverPaths`` so the folder
convention is defined in exactly one place. The layout matches the
v2 notebooks, so existing outputs are picked up as-is::

    {data_root}/
    ├── SWORD/                                   continental SWORD gpkgs
    │   └── Extracted_Rivers/{river}/            step: sword
    │       ├── {river}_SWORD_reaches.gpkg
    │       ├── {river}_SWORD_reaches.csv
    │       ├── {river}_SWORD_nodes.gpkg
    │       └── {river}_sword_run.yaml           provenance record
    └── RiverMapping/
        └── Reaches/{river}/                     step: reaches
            ├── {river}_reach_lines.gpkg
            ├── {river}_reach_polygons.gpkg
            └── {river}_reaches_run.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def sword_continent_files(
    sword_dir: Path, continent: str, version: str
) -> tuple[Path, Path]:
    """Return (reaches_gpkg, nodes_gpkg) for one SWORD continent."""
    return (
        sword_dir / f"{continent}_sword_reaches_{version}.gpkg",
        sword_dir / f"{continent}_sword_nodes_{version}.gpkg",
    )


@dataclass(frozen=True)
class RiverPaths:
    """Input/output paths for one river."""

    data_root: Path
    river: str

    # ── step: sword ─────────────────────────────────────────────────
    @property
    def sword_dir(self) -> Path:
        return self.data_root / "SWORD" / "Extracted_Rivers" / self.river

    @property
    def sword_reaches(self) -> Path:
        return self.sword_dir / f"{self.river}_SWORD_reaches.gpkg"

    @property
    def sword_reaches_csv(self) -> Path:
        return self.sword_dir / f"{self.river}_SWORD_reaches.csv"

    @property
    def sword_nodes(self) -> Path:
        return self.sword_dir / f"{self.river}_SWORD_nodes.gpkg"

    # ── step: reaches ───────────────────────────────────────────────
    @property
    def reaches_dir(self) -> Path:
        return self.data_root / "RiverMapping" / "Reaches" / self.river

    @property
    def reach_lines(self) -> Path:
        return self.reaches_dir / f"{self.river}_reach_lines.gpkg"

    @property
    def reach_polygons(self) -> Path:
        return self.reaches_dir / f"{self.river}_reach_polygons.gpkg"

    # ── provenance ──────────────────────────────────────────────────
    def run_record(self, step: str) -> Path:
        """Provenance YAML written next to a step's outputs."""
        folder = {"sword": self.sword_dir, "reaches": self.reaches_dir}[step]
        return folder / f"{self.river}_{step}_run.yaml"
