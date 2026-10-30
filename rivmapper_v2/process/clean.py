"""Step ``clean_masks``: learned removal of non-channel pieces.

Runs after ``process_masks``. For every reach:

1. Splits each processed reach-year mask into the main corridor and
   candidate pieces (``units.py``) and measures them, using the neighbouring
   years (±1, ±2) and the reach's whole record for persistence.
2. Writes the piece table ``masks/pieces/{river}_r{NNN}_pieces.csv`` (one row
   per piece and year: features, context, and — with a model — P(channel)).
3. With ``process.clean_model`` set: removes pieces whose P(channel) is
   below ``process.clean_threshold`` and writes the result to
   ``masks/cleaned/`` (same four bands as ``processed/``; ``core`` is
   restricted to the cleaned channel). The QA table gains
   ``channel_km2_cleaned`` and ``pieces_removed``. Without a model the step
   only writes the piece tables (what the labelling bundle needs).

The main corridor is never removed, so cleaning cannot break the river.
Processed masks are left untouched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Project, River
from ..crs import crs_to_string
from ..gee import Grid
from ..results import FAILED, OK, StepResult
from . import units

log = logging.getLogger(__name__)

STEP = "clean_masks"
NODATA = 255
CONTEXT = ("river", "ds_order", "year", "status", "q_scene_pctile", "braided", "w_px")


def _read(path: Path):
    import rasterio

    if not path.exists():
        return None, None
    with rasterio.open(path) as s:
        return s.read(), s.profile


def reach_stack(river: River, o: int, years: list[int]) -> dict[int, np.ndarray | None]:
    """{year: 4-band processed mask, or None (missing / no data)}."""
    out = {}
    for y in years:
        a, _ = _read(river.paths.processed_mask(o, y))
        out[y] = None if a is None or (a[0] == NODATA).all() else a
    return out


def reach_pieces(
    river: River, o: int, years: list[int], w_px: float, qa: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Pieces and features for every year of reach ``o``.

    Returns (table with one row per piece-year, {year: piece-id array}).
    """
    stack = reach_stack(river, o, years)
    obs = [a for a in stack.values() if a is not None]
    if not obs:
        return pd.DataFrame(), {}
    inside = np.zeros(obs[0].shape[1:], bool)
    for a in obs:
        inside |= a[0] != NODATA
    freq = np.mean([a[0] == 1 for a in obs], axis=0)
    ctx = qa.set_index(["ds_order", "year"]) if qa is not None and len(qa) else None
    rows, labs = [], {}
    for y in years:
        a = stack[y]
        if a is None:
            continue
        ch = a[0] == 1
        others = {}
        for k in (-2, -1, 1, 2):
            b = stack.get(y + k)
            others[k] = (b[0] == 1) if b is not None else None
        lab, _, tab = units.pieces(
            ch,
            inside,
            w_px,
            core=a[1] == 1,
            p_env=a[2] / 100.0,
            p_core=a[3] / 100.0,
            others=others,
            freq_all=freq,
        )
        labs[y] = lab
        if tab.empty:
            continue
        tab = tab.reset_index()
        tab["river"], tab["ds_order"], tab["year"], tab["w_px"] = river.name, o, y, w_px
        if ctx is not None and (o, y) in ctx.index:
            r = ctx.loc[(o, y)]
            tab["status"] = r.get("status", "")
            tab["q_scene_pctile"] = r.get("q_scene_pctile", np.nan)
            tab["braided"] = bool(r.get("braided", False)) if pd.notna(r.get("braided", np.nan)) else False
        else:
            tab["status"], tab["q_scene_pctile"], tab["braided"] = "", np.nan, False
        rows.append(tab)
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return table, labs


def clean_masks(river: River) -> StepResult:
    from . import ml
    from .step import _write_mask
    from .windows import RiverLayout

    p = river.paths
    if not p.process_qa_table.exists():
        return StepResult(river.name, STEP, FAILED, "no processed masks — run process_masks")
    qa = pd.read_csv(p.process_qa_table, keep_default_na=False, na_values=[""])
    layout = RiverLayout.from_paths(p)
    years = sorted(qa["year"].unique().tolist())
    s = river.process
    model = ml.load(s.clean_model) if s.clean_model is not None else None
    removed_rows = []
    n_pieces = 0
    for o in layout.ds_orders:
        table, labs = reach_pieces(river, o, years, layout.width_px(o), qa[qa.ds_order == o])
        n_pieces += len(table)
        if model is not None and len(table):
            table["p_channel"] = ml.predict(model, table)
            table["removed"] = table["p_channel"] < s.clean_threshold
        p.pieces_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(p.pieces_table(o), index=False)
        if model is None:
            log.info("  reach %d: %d pieces", o, len(table))
            continue
        drop = table[table["removed"]] if len(table) else table
        for y in years:
            a, prof = _read(p.processed_mask(o, y))
            if a is None:
                continue
            out = a.copy()
            ids = drop.loc[drop["year"] == y, "piece"].to_numpy() if len(drop) else []
            km2 = np.nan
            if y in labs and len(ids):
                gone = np.isin(labs[y], ids)
                out[0][gone] = 0
                out[1][gone & (a[1] == 1)] = 0
            if (a[0] != NODATA).any():
                km2 = float((out[0] == 1).sum()) * 9e-4
            t = prof["transform"]
            grid = Grid(crs_to_string(prof["crs"]), t.c, t.f, t.a, prof["width"], prof["height"])
            _write_mask(
                p.cleaned_mask(o, y),
                grid,
                {b: out[i] for i, b in enumerate(("channel", "core", "p_envelope", "p_core"))},
                {"river": river.name, "ds_order": str(o), "year": str(y),
                 "clean_model": str(s.clean_model), "clean_threshold": str(s.clean_threshold),
                 "pieces_removed": json.dumps([int(i) for i in ids])},
            )
            removed_rows.append({"ds_order": o, "year": y, "channel_km2_cleaned": km2,
                                 "pieces_removed": len(ids)})
        log.info("  reach %d: %d pieces, %d removed", o, len(table),
                 int(table["removed"].sum()) if len(table) else 0)
    if model is None:
        return StepResult(
            river.name, STEP, OK,
            f"{n_pieces} candidate pieces written (no clean_model set: labelling mode)",
        )
    rem = pd.DataFrame(removed_rows)
    qa = qa.drop(columns=[c for c in ("channel_km2_cleaned", "pieces_removed") if c in qa])
    qa = qa.merge(rem, on=["ds_order", "year"], how="left")
    qa.to_csv(p.process_qa_table, index=False)
    return StepResult(
        river.name, STEP, OK,
        f"{n_pieces} pieces; {int(rem['pieces_removed'].sum())} removed; cleaned masks written",
    )


def run(project: Project, rivers: list[River]) -> list[StepResult]:
    out = []
    for r in rivers:
        log.info("%s", r.name)
        try:
            out.append(clean_masks(r))
        except Exception as e:  # noqa: BLE001
            log.debug("%s: error", r.name, exc_info=True)
            out.append(StepResult(r.name, STEP, FAILED, f"{type(e).__name__}: {e}"))
    return out
