"""Standard file layout under a project's ``data_root``.

Every step reads and writes through ``RiverPaths`` so the folder
convention is defined in exactly one place. One folder per river, one
subfolder per step::

    {data_root}/
    └── {river}/
        ├── sword/                               step: sword
        │   ├── {river}_sword_reaches.gpkg
        │   ├── {river}_sword_reaches.csv
        │   ├── {river}_sword_nodes.gpkg
        │   └── {river}_sword_run.yaml           provenance record
        ├── reaches/                             step: reaches
        │   ├── {river}_reach_lines.gpkg
        │   ├── {river}_reach_polygons.gpkg
        │   ├── {river}_valley_line.gpkg
        │   └── {river}_reaches_run.yaml
        ├── masks/                               steps: extract_masks, process_masks
        │   ├── raw/{river}_r{NNN}_{year}.tif    DSWE counts, river grid
        │   ├── {river}_extract_masks_log.csv    one row per reach-year
        │   ├── {river}_extract_masks_run.yaml
        │   ├── {river}_scene_manifest.csv       one row per scene (raw v2)
        │   ├── model/{river}_r{NNN}_posterior.tif   cached HMM probabilities
        │   ├── model/{river}_r{NNN}_window.tif      window context (owner, terminals, reach line)
        │   ├── diagnostics/{river}_r{NNN}_{year}.tif  per-pixel decision codes
        │   ├── processed/{river}_r{NNN}_{year}.tif  channel masks
        │   ├── qa/{river}_r{NNN}_qa.png         QA figures
        │   ├── {river}_process_masks_qa.csv     one row per reach-year
        │   └── {river}_process_masks_run.yaml
        └── discharge/                           step: discharge
            ├── {river}_geoglows_join.csv        reach → GEOGLOWS river id
            ├── {river}_geoglows_daily.csv       daily discharge, one column per river id
            ├── {river}_stage.csv                discharge on scene dates, per reach-year
            └── {river}_discharge_run.yaml

Shared external datasets (the continental SWORD files) live outside
``data_root``; see ``rivmapper_v2.data``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Pipeline steps in run order (kept here, dependency-free, so the CLI can
# list them without importing the heavy step modules).
STEP_NAMES = ("sword", "reaches", "extract_masks", "discharge", "process_masks")


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
    def river_dir(self) -> Path:
        """All outputs for this river."""
        return self.data_root / self.river

    @property
    def sword_dir(self) -> Path:
        return self.river_dir / "sword"

    @property
    def sword_reaches(self) -> Path:
        return self.sword_dir / f"{self.river}_sword_reaches.gpkg"

    @property
    def sword_reaches_csv(self) -> Path:
        return self.sword_dir / f"{self.river}_sword_reaches.csv"

    @property
    def sword_nodes(self) -> Path:
        return self.sword_dir / f"{self.river}_sword_nodes.gpkg"

    # ── step: reaches ───────────────────────────────────────────────
    @property
    def reaches_dir(self) -> Path:
        return self.river_dir / "reaches"

    @property
    def reach_lines(self) -> Path:
        return self.reaches_dir / f"{self.river}_reach_lines.gpkg"

    @property
    def reach_polygons(self) -> Path:
        return self.reaches_dir / f"{self.river}_reach_polygons.gpkg"

    @property
    def valley_line(self) -> Path:
        return self.reaches_dir / f"{self.river}_valley_line.gpkg"

    # ── step: extract_masks ─────────────────────────────────────────
    @property
    def masks_dir(self) -> Path:
        return self.river_dir / "masks"

    @property
    def raw_masks_dir(self) -> Path:
        return self.masks_dir / "raw"

    def raw_mask(self, ds_order: int, year: int) -> Path:
        return self.raw_masks_dir / f"{self.river}_r{ds_order:03d}_{year}.tif"

    @property
    def masks_log(self) -> Path:
        return self.masks_dir / f"{self.river}_extract_masks_log.csv"

    @property
    def scene_manifest(self) -> Path:
        return self.masks_dir / f"{self.river}_scene_manifest.csv"

    # ── step: process_masks ─────────────────────────────────────────
    @property
    def processed_masks_dir(self) -> Path:
        return self.masks_dir / "processed"

    def processed_mask(self, ds_order: int, year: int) -> Path:
        return self.processed_masks_dir / f"{self.river}_r{ds_order:03d}_{year}.tif"

    @property
    def diagnostics_dir(self) -> Path:
        return self.masks_dir / "diagnostics"

    def diagnostic(self, ds_order: int, year: int) -> Path:
        return self.diagnostics_dir / f"{self.river}_r{ds_order:03d}_{year}.tif"

    @property
    def model_dir(self) -> Path:
        return self.masks_dir / "model"

    def posterior(self, ds_order: int) -> Path:
        return self.model_dir / f"{self.river}_r{ds_order:03d}_posterior.tif"

    def window_context(self, ds_order: int) -> Path:
        return self.model_dir / f"{self.river}_r{ds_order:03d}_window.tif"

    @property
    def mask_qa_dir(self) -> Path:
        return self.masks_dir / "qa"

    def mask_qa_figure(self, ds_order: int) -> Path:
        return self.mask_qa_dir / f"{self.river}_r{ds_order:03d}_qa.png"

    @property
    def process_qa_table(self) -> Path:
        return self.masks_dir / f"{self.river}_process_masks_qa.csv"

    # ── discharge ───────────────────────────────────────────────────
    @property
    def discharge_dir(self) -> Path:
        return self.river_dir / "discharge"

    @property
    def discharge_join(self) -> Path:
        return self.discharge_dir / f"{self.river}_geoglows_join.csv"

    @property
    def discharge_daily(self) -> Path:
        return self.discharge_dir / f"{self.river}_geoglows_daily.csv"

    @property
    def stage_table(self) -> Path:
        return self.discharge_dir / f"{self.river}_stage.csv"

    # ── provenance ──────────────────────────────────────────────────
    def run_record(self, step: str) -> Path:
        """Provenance YAML written next to a step's outputs."""
        folder = {
            "sword": self.sword_dir,
            "reaches": self.reaches_dir,
            "extract_masks": self.masks_dir,
            "process_masks": self.masks_dir,
            "discharge": self.discharge_dir,
        }[step]
        return folder / f"{self.river}_{step}_run.yaml"
