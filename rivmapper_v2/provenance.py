"""Run records: what settings and code produced a step's outputs.

Each step writes ``{river}_{step}_run.yaml`` beside its outputs. On the
next run, if outputs exist but the recorded *parameters* differ from the
current config, the step reports the outputs as stale instead of
silently skipping them.
"""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from . import __version__

_PACKAGES = ("geopandas", "shapely", "pyogrio", "pandas", "numpy", "pydantic")


def _git_commit() -> str | None:
    """Commit hash of the installed package source, if it is a git checkout."""
    try:
        here = Path(__file__).resolve().parent
        out = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "-C", str(here), "status", "--porcelain", "--", "."],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def write_run_record(
    path: Path,
    *,
    step: str,
    river: str,
    config_path: Path | None,
    parameters: dict[str, Any],
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> None:
    """Write a provenance YAML for one step of one river."""
    record = {
        "step": step,
        "river": river,
        "finished_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "rivmapper_v2": {"version": __version__, "git_commit": _git_commit()},
        "config_file": str(config_path) if config_path else None,
        "parameters": parameters,
        "inputs": {k: str(v) for k, v in (inputs or {}).items()},
        "outputs": {k: str(v) for k, v in (outputs or {}).items()},
        "summary": summary or {},
        "environment": _package_versions(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(record, f, sort_keys=False, allow_unicode=True)


def recorded_parameters(path: Path) -> dict[str, Any] | None:
    """Return the ``parameters`` block of an existing run record, if any."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("parameters")
    except (OSError, yaml.YAMLError):
        return None
