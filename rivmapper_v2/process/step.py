"""Step ``process_masks``: raw DSWE counts → annual channel masks.

Phase 0 scaffold. The step already validates its inputs (reach polygons,
widths, every raw reach-year file, the reach windows) and writes a run
record, but does not yet write masks; it reports ``skipped`` so a full
pipeline run is not mistaken for a finished one. The processing itself
(HMM, channel extraction, year check) lands in phase 1 — see
docs/methods/processing.md and the implementation plan.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import Project, River
from ..provenance import write_run_record
from ..results import FAILED, SKIPPED, StepResult
from .windows import RiverLayout

log = logging.getLogger(__name__)

STEP = "process_masks"


def check_inputs(river: River) -> tuple[RiverLayout, list[int]]:
    """Load the reach layout and confirm every raw file exists.

    Returns:
        (layout, years).

    Raises:
        FileNotFoundError / ValueError: with a message saying what to run.
    """
    river.require("masks", "year_start", "year_end")
    p = river.paths
    layout = RiverLayout.from_paths(p)
    for o in layout.ds_orders:
        layout.width_m(o)  # raises if width_ref is missing
    years = list(range(river.masks.year_start, river.masks.year_end + 1))
    missing = [
        p.raw_mask(o, y).name
        for o in layout.ds_orders
        for y in years
        if not p.raw_mask(o, y).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} raw mask file(s) missing (e.g. {missing[:3]}) — run "
            f"extract_masks first"
        )
    return layout, years


def process_masks(river: River, config_path: Path | None = None) -> StepResult:
    """Run step ``process_masks`` for one river."""
    try:
        layout, years = check_inputs(river)
    except (FileNotFoundError, ValueError) as e:
        return StepResult(river.name, STEP, FAILED, str(e))

    windows = layout.windows()
    narrow = [
        o for o in layout.ds_orders if layout.width_px(o) < river.process.resolvable_px
    ]
    summary = {
        "phase": "0 (scaffold: inputs checked, no masks written)",
        "analysis_crs": layout.crs,
        "grid_origin": list(layout.origin),
        "n_reaches": len(layout.ds_orders),
        "n_windows": len(windows),
        "years": [years[0], years[-1]],
        "reaches_below_resolvable_px": narrow,
    }
    write_run_record(
        river.paths.run_record(STEP),
        step=STEP,
        river=river.name,
        config_path=config_path,
        parameters=river.process.model_dump(mode="json"),
        inputs={
            "reach_polygons": river.paths.reach_polygons,
            "raw_masks": river.paths.raw_masks_dir,
        },
        outputs={},
        summary=summary,
    )
    return StepResult(
        river.name,
        STEP,
        SKIPPED,
        f"not implemented yet (phase 0 scaffold): inputs OK for "
        f"{len(layout.ds_orders)} reaches × {len(years)} years, "
        f"{len(windows)} windows",
        summary,
    )


def run(project: Project, rivers: list[River]) -> list[StepResult]:
    """Run step ``process_masks`` for ``rivers``."""
    results = []
    for r in rivers:
        log.info("%s", r.name)
        try:
            res = process_masks(r, project.config_path)
        except Exception as e:  # report per river, keep going
            log.debug("%s: error", r.name, exc_info=True)
            res = StepResult(r.name, STEP, FAILED, f"{type(e).__name__}: {e}")
        results.append(res)
    return results
