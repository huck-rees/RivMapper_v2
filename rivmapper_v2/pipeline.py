"""Run pipeline steps in order across a project's rivers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import pandas as pd

from . import discharge, imagery, masks, reaches, sword
from .config import Project, River
from .process import clean as clean_step
from .process import step as process_step
from .results import BLOCKED, StepResult

log = logging.getLogger(__name__)

StepFn = Callable[[Project, list[River]], list[StepResult]]

# Pipeline order.
STEPS: dict[str, StepFn] = {
    sword.STEP: sword.run,
    reaches.STEP: reaches.run,
    masks.STEP: masks.run,
    imagery.STEP: imagery.run,
    discharge.STEP: discharge.run,
    process_step.STEP: process_step.run,
    clean_step.STEP: clean_step.run,
}
# Steps whose failure does not block later steps (their outputs are extras).
OPTIONAL_STEPS = {discharge.STEP, imagery.STEP, clean_step.STEP}
# Steps that run only when named (--steps), not in a default run.
EXPLICIT_STEPS = {imagery.STEP}


def run_pipeline(
    project: Project,
    steps: Iterable[str] | None = None,
    rivers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Run ``steps`` (default: all) for ``rivers`` (default: all).

    Steps always run in pipeline order, whatever order they are given in.
    A river that fails a step is marked ``blocked`` for later steps.

    Returns:
        One row per (river, step) with status and message.
    """
    wanted = list(steps) if steps else [s for s in STEPS if s not in EXPLICIT_STEPS]
    unknown = [s for s in wanted if s not in STEPS]
    if unknown:
        raise KeyError(f"Unknown step(s) {unknown}. Available: {list(STEPS)}")
    ordered = [s for s in STEPS if s in wanted]
    selected = project.select(rivers)

    all_results: list[StepResult] = []
    failed: dict[str, str] = {}
    for step in ordered:
        runnable = [r for r in selected if r.name not in failed]
        log.info(
            "── %s (%d river%s) ──",
            step,
            len(runnable),
            "" if len(runnable) == 1 else "s",
        )
        for r in selected:
            if r.name in failed:
                all_results.append(
                    StepResult(
                        r.name, step, BLOCKED, f"{failed[r.name]} step did not succeed"
                    )
                )
        if not runnable:
            continue
        for res in STEPS[step](project, runnable):
            log.info("  %-24s %s", res.river, res)
            all_results.append(res)
            if not res.succeeded and step not in OPTIONAL_STEPS:
                failed[res.river] = step

    table = pd.DataFrame(
        [
            {"river": r.river, "step": r.step, "status": r.status, "message": r.message}
            for r in all_results
        ]
    )
    if not table.empty:
        order = {n: i for i, n in enumerate(project.river_names)}
        table = table.sort_values(
            ["river", "step"],
            key=lambda c: (
                c.map(order)
                if c.name == "river"
                else c.map({s: i for i, s in enumerate(STEPS)})
            ),
        ).reset_index(drop=True)
    return table
