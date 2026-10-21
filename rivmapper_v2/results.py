"""Shared result type and skip/stale logic for pipeline steps."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .provenance import recorded_parameters

log = logging.getLogger(__name__)

OK, SKIPPED, STALE, FAILED, BLOCKED = "ok", "skipped", "stale", "failed", "blocked"


@dataclass
class StepResult:
    """Outcome of one step for one river.

    ``status`` is one of: ok, skipped (outputs current), stale (outputs
    exist but were made with different parameters; not re-run without
    ``overwrite``), failed, or blocked (an earlier step failed).
    """

    river: str
    step: str
    status: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in (OK, SKIPPED)

    def __str__(self) -> str:
        return f"{self.status}" + (f": {self.message}" if self.message else "")


def check_existing(
    river: str,
    step: str,
    outputs: Iterable[Path],
    record_path: Path,
    parameters: dict[str, Any],
    overwrite: bool,
) -> StepResult | None:
    """Decide whether a step can be skipped.

    Returns a SKIPPED or STALE result if all outputs already exist and
    ``overwrite`` is False; returns None if the step should run.
    """
    if overwrite or not all(p.exists() for p in outputs):
        return None

    previous = recorded_parameters(record_path)
    if previous is None:
        # Outputs pre-date run records (e.g. made by the old notebooks).
        return StepResult(
            river, step, SKIPPED, "outputs exist (no run record to compare)"
        )
    changed = sorted(
        k
        for k in set(previous) | set(parameters)
        if previous.get(k) != parameters.get(k)
    )
    if changed:
        diffs = ", ".join(
            f"{k}: {previous.get(k)!r} → {parameters.get(k)!r}" for k in changed
        )
        log.warning(
            "%s/%s: outputs exist but settings changed (%s). "
            "Re-run with --overwrite to rebuild.",
            river,
            step,
            diffs,
        )
        return StepResult(
            river,
            step,
            STALE,
            f"settings changed since last run ({', '.join(changed)}); "
            f"use --overwrite to rebuild",
            {"changed": changed},
        )
    return StepResult(river, step, SKIPPED, "outputs up to date")
