"""Step ``process_masks``: raw DSWE counts → annual channel masks.

Per river, in five passes:

1. **Fit** the HMM (``hmm.py``) on a sample of pixels from every reach.
2. **Posteriors** for every pixel and year, per reach, cached in
   ``masks/model/`` (P(channel), P(margin), × 100).
3. **Extract** the channel window by window (``windows.py``,
   ``channel.py``): each reach with its two neighbours, years in order so
   each year's channel seeds the next; only the centre reach is kept.
4. **Check** each reach's channel area against its 5-year running median;
   outliers are rerun once with a stricter envelope threshold, then
   flagged if still out (``qa.py``).
5. **Write** masks, the QA table, QA figures and the run record.

Outputs per reach-year, ``masks/processed/{river}_r{NNN}_{year}.tif``
(uint8, nodata 255 outside the reach polygon and in ``no_data`` years):

* band 1 ``channel``: near-bankfull channel (0/1);
* band 2 ``core``: low-flow core channel (0/1);
* band 3 ``p_envelope``: P(channel) + P(margin) × 100;
* band 4 ``p_core``: P(channel) × 100.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Project, River
from ..gee import NODATA, Grid
from ..provenance import recorded_parameters, write_run_record
from ..results import FAILED, OK, STALE, StepResult, check_existing
from . import hmm, qa
from .channel import ChannelInputs, extract
from .windows import RiverLayout, Window, offset, read_raw, read_reach_stack

log = logging.getLogger(__name__)

STEP = "process_masks"
PRODUCT = "channel_masks_v1"
OUT_BANDS = ("channel", "core", "p_envelope", "p_core")
RAW_BANDS = ("n_obs", "n_wet_2", "n_wet_3")
STRICT_STEP = 0.2  # envelope threshold increase for outlier reruns


# ─────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────
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


def _raw_fingerprint(path: Path) -> dict:
    import rasterio

    with rasterio.open(path) as src:
        tag = src.tags().get("rivmapper_params")
    return json.loads(tag) if tag else {}


def _envelope_threshold(river: River) -> tuple[float, str]:
    t = river.process.envelope_threshold
    if t == "calibrate":
        return 0.5, "calibration not yet implemented (phase 3); used 0.5"
    return float(t), "fixed"


def _parameters(river: River, raw_fp: dict) -> dict:
    return {
        "product": PRODUCT,
        "raw_product": raw_fp.get("product"),
        "years": [river.masks.year_start, river.masks.year_end],
        # keep_diagnostics changes what is written, not the masks
        **river.process.model_dump(mode="json", exclude={"keep_diagnostics"}),
    }


# ─────────────────────────────────────────────────────────────────────
# Checkpoints: an interrupted run resumes where it stopped
# ─────────────────────────────────────────────────────────────────────
def _run_key(params: dict) -> str:
    """Short fingerprint of the settings; checkpoints only match the same key."""
    import hashlib

    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def _ckpt(paths, name: str, key: str) -> Path:
    return paths.model_dir / f"{paths.river}_{name}_{key}"


def _save_hmm(path: Path, params: hmm.HMMParams) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "theta": params.theta.tolist(),
                "A": params.A.tolist(),
                "pi": params.pi.tolist(),
                "loglik": params.loglik,
                "iterations": params.iterations,
                "notes": params.notes,
            }
        )
    )


def _load_hmm(path: Path) -> hmm.HMMParams:
    d = json.loads(path.read_text())
    return hmm.HMMParams(
        theta=np.array(d["theta"]),
        A=np.array(d["A"]),
        pi=np.array(d["pi"]),
        loglik=d["loglik"],
        iterations=d["iterations"],
        notes=d["notes"],
    )


def _thumbs_from_disk(paths, o, thumb_years) -> dict:
    import rasterio

    out = {}
    for y in thumb_years:
        with rasterio.open(paths.processed_mask(o, y)) as src:
            ch, co = src.read(1), src.read(2)
        inside = ch != NODATA
        out[y] = (ch == 1, co == 1, inside)
    return out


def _thumb_years(years, status_o) -> list[int]:
    obs = [y for y in years if status_o[y] == qa.OBSERVED]
    if len(obs) < 6:
        return obs
    return [obs[i] for i in np.linspace(0, len(obs) - 1, 6).round().astype(int)]


# ─────────────────────────────────────────────────────────────────────
# Passes 1–2: HMM
# ─────────────────────────────────────────────────────────────────────
def _sample(layout, paths, years, n_target, rng) -> np.ndarray:
    """Observations (T, P, 3) for a sample: half near the reach lines."""
    from scipy.ndimage import distance_transform_edt

    lines = _reach_lines(paths)
    total = 0
    sizes = {}
    for o in layout.ds_orders:
        g = layout.grids[o]
        sizes[o] = int((layout.owner(g, [o]) == o).sum())
        total += sizes[o]
    parts = []
    for o in layout.ds_orders:
        g, st = read_reach_stack(paths, o, years, RAW_BANDS)
        inside = st["n_obs"][0] != NODATA
        k = max(50, round(n_target * sizes[o] / max(total, 1)))
        line = layout.rasterize_line(_lines_for(lines, [o]), g, width_px=0)
        near = inside & (distance_transform_edt(~line) <= 2 * layout.width_px(o))
        idx_near = np.flatnonzero(near.ravel())
        idx_all = np.flatnonzero(inside.ravel())
        pick = np.concatenate(
            [
                rng.choice(idx_near, min(k // 2, idx_near.size), replace=False)
                if idx_near.size
                else np.array([], int),
                rng.choice(idx_all, min(k - k // 2, idx_all.size), replace=False),
            ]
        )
        flat = {b: st[b].reshape(len(years), -1)[:, pick] for b in RAW_BANDS}
        parts.append(hmm.observations(flat["n_obs"], flat["n_wet_2"], flat["n_wet_3"]))
    return np.concatenate(parts, axis=1)


def _write_posterior(path: Path, grid: Grid, years, p_ch, p_mg, inside) -> None:
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    T = len(years)
    data = np.full((2 * T, grid.height, grid.width), NODATA, np.uint8)
    data[:T][:, inside] = np.round(100 * p_ch).astype(np.uint8)
    data[T:][:, inside] = np.round(100 * p_mg).astype(np.uint8)
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 2 * T,
        "dtype": "uint8",
        "crs": grid.crs,
        "transform": from_origin(grid.x0, grid.y0, grid.res, grid.res),
        "nodata": NODATA,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "interleave": "band",
    }
    tmp = path.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data)
        for i, y in enumerate(years):
            dst.set_band_description(i + 1, f"p_channel_{y}")
            dst.set_band_description(T + i + 1, f"p_margin_{y}")
    tmp.replace(path)


def _posteriors(layout, paths, years, params, key: str) -> dict[int, dict]:
    """Pass 2: posteriors + coverage per reach. Returns {ds_order: info}.

    Reaches whose posteriors were already written under the same ``key``
    are not recomputed.
    """
    info = {}
    for o in layout.ds_orders:
        done = _ckpt(paths, f"r{o:03d}_coverage", key).with_suffix(".json")
        if done.exists() and paths.posterior(o).exists():
            info[o] = json.loads(done.read_text())
            continue
        g, st = read_reach_stack(paths, o, years, RAW_BANDS)
        inside = st["n_obs"][0] != NODATA
        obs = hmm.observations(*(st[b][:, inside] for b in RAW_BANDS))
        p_ch, p_mg = hmm.posteriors(obs, params)
        _write_posterior(paths.posterior(o), g, years, p_ch, p_mg, inside)
        n_in = max(int(inside.sum()), 1)
        coverage = [
            (st["n_obs"][t][inside] > 0).sum() / n_in for t in range(len(years))
        ]
        info[o] = {"coverage": [float(c) for c in coverage], "inside_px": n_in}
        done.write_text(json.dumps(info[o]))
    return info


# ─────────────────────────────────────────────────────────────────────
# Pass 3: windows
# ─────────────────────────────────────────────────────────────────────
def _reach_lines(paths):
    import geopandas as gpd

    if not paths.reach_lines.exists():
        raise FileNotFoundError(f"reach lines not found at {paths.reach_lines}")
    return gpd.read_file(paths.reach_lines)


def _lines_for(lines, members):
    sel = lines[lines["ds_order"].isin(members)]
    return sel.geometry.union_all() if len(sel) else None


def _mosaic(window: Window, layout, reader, n_out: int, fill) -> list[np.ndarray]:
    """Mosaic per-reach arrays onto the window grid.

    ``reader(ds_order)`` returns (grid, [arrays], inside).
    """
    shape = (window.grid.height, window.grid.width)
    outs = [np.full(shape, fill, dtype=np.float32) for _ in range(n_out)]
    for o in window.members:
        g, arrs, inside = reader(o)
        r, c = offset(g, window.grid)
        sub = (slice(r, r + g.height), slice(c, c + g.width))
        for out, a in zip(outs, arrs, strict=True):
            out[sub][inside] = a[inside]
    return outs


def _read_post_band(paths, o, band: int):
    import rasterio

    with rasterio.open(paths.posterior(o)) as src:
        return src.read(band)


def _window_year(paths, layout, window, years, t):
    """Raw counts (this year and last) and posteriors on the window grid."""
    T = len(years)

    def raw(o, year):
        g, a = read_raw(paths.raw_mask(o, year), ("n_obs", "n_wet_3"))
        inside = a["n_obs"] != NODATA
        return g, [a["n_obs"], a["n_wet_3"]], inside

    def post(o):
        g = layout.grids[o]
        pc = _read_post_band(paths, o, t + 1)
        pm = _read_post_band(paths, o, T + t + 1)
        inside = pc != NODATA
        return g, [pc / 100.0, pm / 100.0], inside

    def raw_prev(o):
        g, arrs, inside = raw(o, years[t - 1])
        return g, arrs[:1], inside

    n_obs, n_wet_3 = _mosaic(window, layout, lambda o: raw(o, years[t]), 2, 0)
    if t > 0:
        (n_prev,) = _mosaic(window, layout, raw_prev, 1, 0)
    else:
        n_prev = np.zeros_like(n_obs)
    p_ch, p_mg = _mosaic(window, layout, post, 2, 0)
    return n_obs, n_wet_3, n_prev, p_ch, p_mg


def _process_window(river, layout, paths, window, years, status, env_thr, lines):
    """Extract every year of one window.

    Returns {year: None (no data) or dict(channel, core, stats, score,
    rerun)}, with ``channel`` and ``core`` cropped to the centre reach's
    grid. Only last year's full-window mask is held in memory, so memory
    does not grow with the record length.
    """
    s = river.process
    o = window.center
    W = layout.width_px(o)
    owner = layout.owner(window.grid, window.members)
    inside = owner > 0
    g = layout.grids[o]
    r0, c0 = offset(g, window.grid)
    sub = (slice(r0, r0 + g.height), slice(c0, c0 + g.width))
    centre = (owner == o)[sub]
    centerline = layout.rasterize_line(
        _lines_for(lines, window.members), window.grid, width_px=0
    )
    by_reach = {
        int(k): geom for k, geom in zip(lines["ds_order"], lines.geometry, strict=True)
    }
    term = layout.terminals(window, by_reach, inside, radius_px=max(10 * W, 20))

    def inputs(t, prev):
        n_obs, n_wet_3, n_prev, p_ch, p_mg = _window_year(
            paths, layout, window, years, t
        )
        return ChannelInputs(
            p_channel=p_ch,
            p_margin=p_mg,
            n_obs=n_obs,
            n_wet_3=n_wet_3,
            n_obs_prev=n_prev,
            inside=inside,
            centerline=centerline,
            prev_mask=prev,
            width_px=W,
            terminals=term,
        )

    def keep(res, thr):
        return {
            "channel": res.channel[sub] & centre,
            "core": res.core[sub] & centre,
            "decision": res.decision[sub] if res.decision is not None else None,
            "stats": res.stats,
            "threshold": thr,
        }

    prev = None
    prev_used = {}  # the seed mask each year used, for reruns (centre crop only)
    results = {}
    for t, y in enumerate(years):
        if status[o][y] == qa.NO_DATA:
            results[y] = None
            continue
        res = extract(inputs(t, prev), s, env_thr)
        prev_used[y] = prev
        results[y] = keep(res, env_thr)
        prev = res.channel
    # Pass 4: year-to-year check on the centre reach. Too much water is
    # rerun once with a stricter envelope threshold (it is usually false
    # water from bad scenes or a flood); too little is only flagged.
    obs_years = [y for y in years if status[o][y] in (qa.OBSERVED, qa.SPARSE)]
    area = [float(results[y]["channel"].sum()) for y in obs_years]
    scores = qa.outlier_scores(area, signed=True)
    for i, y in enumerate(obs_years):
        results[y]["score"] = float(scores[i])
        if scores[i] > s.outlier_mad:
            strict = min(env_thr + STRICT_STEP, 0.95)
            t = years.index(y)
            res2 = keep(extract(inputs(t, prev_used[y]), s, strict), strict)
            a2 = list(area)
            a2[i] = float(res2["channel"].sum())
            sc2 = float(qa.outlier_scores(a2, signed=True)[i])
            results[y]["rerun"] = True
            results[y]["area_before_rerun_px"] = area[i]
            if abs(sc2) < abs(scores[i]):
                first = results[y]["channel"]
                results[y].update(res2, score=sc2)
                d = results[y]["decision"]
                if d is not None:
                    d[first & ~res2["channel"]] = 14
    if river.process.keep_diagnostics:
        _write_window_context(
            paths.window_context(o), window.grid, owner, term, centerline
        )
    return centre, results


def _write_window_context(path, grid, owner, term, centerline) -> None:
    """Window grid: reach owner (ds_order), terminal zones, rasterised reach line."""
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    bands = [
        ("owner", "ds_order of the reach owning the pixel (0 = none)", owner),
        ("terminal", "terminal zone id: where water may enter/leave (0 = none)", term),
        ("reach_line", "SWORD reach line, rasterised (0/1)", centerline),
    ]
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": len(bands),
        "dtype": "uint16",
        "crs": grid.crs,
        "transform": from_origin(grid.x0, grid.y0, grid.res, grid.res),
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    tmp = path.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        for i, (name, desc, arr) in enumerate(bands, start=1):
            dst.write(arr.astype(np.uint16), i)
            dst.set_band_description(i, f"{name}: {desc}")
    tmp.replace(path)


def _write_diagnostic(path, grid, decision, inside, tags) -> None:
    """One band of decision codes (``channel.DECISIONS``), 255 outside."""
    import rasterio
    from rasterio.transform import from_origin

    from .channel import DECISIONS

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((grid.height, grid.width), NODATA, np.uint8)
    if decision is not None:
        data[inside] = decision[inside]
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": "uint8",
        "crs": grid.crs,
        "transform": from_origin(grid.x0, grid.y0, grid.res, grid.res),
        "nodata": NODATA,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    tmp = path.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "decision: why each pixel is or is not channel")
        dst.update_tags(
            decision_codes=json.dumps({str(k): v for k, v in DECISIONS.items()}),
            **{
                k: (v if isinstance(v, str) else json.dumps(v)) for k, v in tags.items()
            },
        )
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────
# Pass 5: outputs
# ─────────────────────────────────────────────────────────────────────
def _write_mask(path, grid, bands: dict, tags: dict) -> None:
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": len(OUT_BANDS),
        "dtype": "uint8",
        "crs": grid.crs,
        "transform": from_origin(grid.x0, grid.y0, grid.res, grid.res),
        "nodata": NODATA,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    desc = {
        "channel": "near-bankfull channel (0/1)",
        "core": "low-flow core channel (0/1)",
        "p_envelope": "P(channel) + P(margin) x 100",
        "p_core": "P(channel) x 100",
    }
    tmp = path.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        for i, b in enumerate(OUT_BANDS, start=1):
            dst.write(bands[b], i)
            dst.set_band_description(i, f"{b}: {desc[b]}")
        dst.update_tags(
            **{k: (v if isinstance(v, str) else json.dumps(v)) for k, v in tags.items()}
        )
    tmp.replace(path)


STAGE_COLUMNS = ("q_scene_median", "q_scene_pctile", "stage_flag")


def _add_stage(table: pd.DataFrame, paths) -> pd.DataFrame:
    """Join the discharge step's stage statistics, if it has run.

    Adds ``q_scene_median`` (m³/s), ``q_scene_pctile`` and ``stage_flag``,
    and appends ``low_stage`` / ``high_stage`` to ``flags``. Without them
    the columns are empty and ``flags`` gains ``no_discharge``.
    """
    table["flags"] = table["flags"].fillna("").astype(str)
    if paths.stage_table.exists():
        stage = pd.read_csv(paths.stage_table)[["ds_order", "year", *STAGE_COLUMNS]]
        stage["stage_flag"] = stage["stage_flag"].fillna("")
        table = table.drop(columns=[c for c in STAGE_COLUMNS if c in table.columns])
        table = table.merge(stage, on=["ds_order", "year"], how="left")
        extra = table["stage_flag"].fillna("")
    else:
        for c in STAGE_COLUMNS:
            table[c] = np.nan if c != "stage_flag" else ""
        extra = pd.Series("no_discharge", index=table.index)
    table["flags"] = [
        ";".join(x for x in (f, e) if x)
        for f, e in zip(table["flags"], extra, strict=True)
    ]
    return table


def _flags(row, river: River, width_px: float) -> str:
    f = []
    if row["status"] in (qa.SPARSE, qa.INFERRED):
        f.append(row["status"])
    if width_px < 2:
        f.append("unresolved")
    elif width_px < river.process.resolvable_px:
        f.append("marginal")
    sc = row.get("score")
    if sc is not None and np.isfinite(sc) and abs(sc) > river.process.outlier_mad:
        f.append("area_high" if sc > 0 else "area_low")
    return ";".join(f)


def process_masks(river: River, config_path: Path | None = None) -> StepResult:
    """Run step ``process_masks`` for one river."""
    t0 = time.time()
    try:
        layout, years = check_inputs(river)
    except (FileNotFoundError, ValueError) as e:
        return StepResult(river.name, STEP, FAILED, str(e))
    p = river.paths
    raw_fp = _raw_fingerprint(p.raw_mask(layout.ds_orders[0], years[0]))
    params = _parameters(river, raw_fp)
    outputs = [p.processed_mask(o, y) for o in layout.ds_orders for y in years]
    existing = check_existing(
        river.name, STEP, outputs, p.run_record(STEP), params, river.overwrite
    )
    if existing:
        return existing
    previous = recorded_parameters(p.run_record(STEP))
    if (
        not river.overwrite
        and previous is not None
        and previous != params
        and any(o.exists() for o in outputs)
    ):
        return StepResult(
            river.name,
            STEP,
            STALE,
            "existing masks were made with different settings; use --overwrite",
        )

    s = river.process
    env_thr, env_note = _envelope_threshold(river)
    rng = np.random.default_rng(0)

    key = _run_key(params)
    hmm_file = _ckpt(p, "hmm", key).with_suffix(".json")
    if hmm_file.exists():
        log.info("  resuming: HMM and finished reaches from an interrupted run")
        params_hmm = _load_hmm(hmm_file)
    else:
        log.info(
            "  fitting %d-state HMM on %d sampled pixels", s.states, s.fit_sample_px
        )
        sample = _sample(layout, p, years, s.fit_sample_px, rng)
        params_hmm = hmm.fit(sample, s.states)
        _save_hmm(hmm_file, params_hmm)
    log.info("  posteriors for %d reaches", len(layout.ds_orders))
    info = _posteriors(layout, p, years, params_hmm, key)
    status = {
        o: qa.year_status(years, info[o]["coverage"], s.max_infer_gap)
        for o in layout.ds_orders
    }

    lines = _reach_lines(p)
    reach_len = (
        dict(zip(lines["ds_order"].astype(int), lines["reach_len"], strict=True))
        if "reach_len" in lines.columns
        else {}
    )
    rows = []
    thumbs_all = {}
    for window in layout.windows():
        o = window.center
        rows_file = _ckpt(p, f"r{o:03d}_rows", key).with_suffix(".csv")
        if (
            not river.overwrite
            and rows_file.exists()
            and all(
                p.processed_mask(o, y).exists()
                and (not s.keep_diagnostics or p.diagnostic(o, y).exists())
                for y in years
            )
        ):
            done = pd.read_csv(rows_file, keep_default_na=False, na_values=[""])
            done["flags"] = done["flags"].fillna("").astype(str)
            rows.extend(done.to_dict("records"))
            thumbs_all[o] = _thumbs_from_disk(p, o, _thumb_years(years, status[o]))
            log.info("  reach %d: already done", o)
            continue
        centre, results = _process_window(
            river, layout, p, window, years, status, env_thr, lines
        )
        g = layout.grids[o]
        inside = centre
        W = layout.width_px(o)
        T = len(years)
        thumb_years = _thumb_years(years, status[o])
        thumbs = {}
        for t, y in enumerate(years):
            res = results[y]
            bands = {
                b: np.full((g.height, g.width), NODATA, np.uint8) for b in OUT_BANDS
            }
            row = {
                "ds_order": o,
                "year": y,
                "status": status[o][y],
                "coverage": round(float(info[o]["coverage"][t]), 4),
            }
            if res is not None:
                ch = res["channel"]
                co = res["core"]
                pc = _read_post_band(p, o, t + 1)
                pm = _read_post_band(p, o, T + t + 1)
                bands["channel"][inside] = ch[inside]
                bands["core"][inside] = co[inside]
                pe = np.minimum(pc[inside].astype(np.int16) + pm[inside], 100)
                bands["p_envelope"][inside] = pe.astype(np.uint8)
                bands["p_core"][inside] = pc[inside]
                px_km2 = layout.res**2 / 1e6
                row.update(
                    channel_km2=round(float((ch & inside).sum() * px_km2), 4),
                    core_km2=round(float((co & inside).sum() * px_km2), 4),
                    score=res.get("score"),
                    rerun=bool(res.get("rerun", False)),
                    channel_km2_before_rerun=(
                        round(res["area_before_rerun_px"] * px_km2, 4)
                        if "area_before_rerun_px" in res
                        else np.nan
                    ),
                    envelope_threshold=res["threshold"],
                    **res["stats"],
                )
                if y in thumb_years:
                    thumbs[y] = (ch & inside, co & inside, inside)
            else:
                row.update(channel_km2=np.nan, core_km2=np.nan, score=None, rerun=False)
            # mean width = area / reach length, and relative to width_ref
            length = reach_len.get(o)
            if length and np.isfinite(row["channel_km2"]):
                row["mean_width_m"] = round(row["channel_km2"] * 1e6 / length, 1)
                row["width_over_W"] = round(row["mean_width_m"] / layout.width_m(o), 3)
            else:
                row["mean_width_m"] = row["width_over_W"] = np.nan
            row["flags"] = _flags(row, river, W)
            _write_mask(
                p.processed_mask(o, y),
                g,
                bands,
                {
                    "rivmapper_params": params,
                    "river": river.name,
                    "ds_order": str(o),
                    "year": str(y),
                    "status": row["status"],
                    "flags": row["flags"],
                },
            )
            if s.keep_diagnostics:
                _write_diagnostic(
                    p.diagnostic(o, y),
                    g,
                    res["decision"] if res is not None else None,
                    inside,
                    {"river": river.name, "ds_order": str(o), "year": str(y)},
                )
            rows.append(row)
        thumbs_all[o] = thumbs
        pd.DataFrame([r for r in rows if r["ds_order"] == o]).to_csv(
            rows_file, index=False
        )
        log.info(
            "  reach %d: %d years, %d rerun, %d flagged outliers",
            o,
            sum(1 for y in years if results[y] is not None),
            sum(1 for r in rows if r["ds_order"] == o and r.get("rerun")),
            sum(1 for r in rows if r["ds_order"] == o and "area_" in r["flags"]),
        )

    table = pd.DataFrame(rows)
    table["score"] = pd.to_numeric(table["score"], errors="coerce").round(2)
    table = _add_stage(table, p)
    table.to_csv(p.process_qa_table, index=False)
    for o in layout.ds_orders:
        sub_rows = table[table.ds_order == o].to_dict("records")
        qa.reach_figure(
            sub_rows,
            thumbs_all[o],
            p.mask_qa_figure(o),
            f"{river.name} reach {o} (W = {layout.width_m(o):.0f} m, "
            f"{layout.width_px(o):.1f} px)",
        )
    qa.overview_figure(
        table,
        p.mask_qa_dir / f"{river.name}_overview.png",
        f"{river.name}: channel area by reach and year, relative to each reach's median",
    )

    counts = table["status"].value_counts().to_dict()
    n_out = int(table["flags"].fillna("").str.contains("area_").sum())
    summary = {
        "hmm": params_hmm.to_dict(),
        "envelope_threshold": env_thr,
        "envelope_threshold_note": env_note,
        "status_counts": counts,
        "area_outliers": n_out,
        "reruns": int(table["rerun"].sum()),
        "seconds": round(time.time() - t0, 1),
    }
    write_run_record(
        p.run_record(STEP),
        step=STEP,
        river=river.name,
        config_path=config_path,
        parameters=params,
        inputs={"raw_masks": p.raw_masks_dir, "reach_polygons": p.reach_polygons},
        outputs={"processed": p.processed_masks_dir, "qa_table": p.process_qa_table},
        summary=summary,
    )
    for f in p.model_dir.glob(f"{p.river}_r*_rows_{key}.csv"):
        f.unlink()  # finished: per-reach checkpoints are in the QA table now
    return StepResult(
        river.name,
        STEP,
        OK,
        f"{len(outputs)} masks ({counts.get('observed', 0)} observed, "
        f"{counts.get('sparse', 0)} sparse, {counts.get('inferred', 0)} inferred, "
        f"{counts.get('no_data', 0)} no data; {n_out} area outliers)",
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
