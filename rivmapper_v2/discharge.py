"""Step ``discharge`` — modelled discharge on scene dates (stage flag).

Annual masks mix scenes from different water levels. This step puts a
number on that: for every reach and year, the modelled discharge on the
dates of the scenes that went into its counts, compared with the reach's
own discharge record. Years mapped mostly at low or high water are
flagged, so their channel areas can be read (or excluded) accordingly.

Source: GEOGLOWS v2 retrospective simulation (daily, 1940–present, on the
TDX-Hydro river network; CC BY 4.0). Read anonymously from the AWS Open
Data bucket ``s3://geoglows-v2``.

Per river:

1. **Join.** Each reach is matched to one GEOGLOWS river (``LINKNO``):
   among the links whose outlet lies within ``search_km`` of the reach's
   downstream end, the nearest one whose drainage area is within a factor
   ``area_tolerance`` of the reach's (SWORD ``facc``). If none matches on
   area, the best area match within twice the radius is used and marked.
2. **Fetch** daily discharge for the joined links over the mask years.
3. **Stage statistics** per reach-year, from the kept scenes in the scene
   manifest (``extract_masks``), weighted by each scene's clear fraction
   over the reach: the weighted median scene-date discharge, its
   percentile in the reach's daily record over the mask years, and
   ``low_stage`` / ``high_stage`` flags beyond the set percentiles.

Outputs (``{river}/discharge/``): ``{river}_geoglows_join.csv``,
``{river}_geoglows_daily.csv``, ``{river}_stage.csv`` and the run record.

The step is optional: if it fails (no network, no manifest), later steps
still run and the QA table records no stage information.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Protocol

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import Project, River
from .provenance import write_run_record
from .results import FAILED, OK, StepResult, check_existing

log = logging.getLogger(__name__)

STEP = "discharge"
DAILY_URI = "s3://geoglows-v2/retrospective/daily.zarr"
JOIN_COLUMNS = [
    "ds_order",
    "river_id",
    "outlet_lon",
    "outlet_lat",
    "dist_km",
    "link_area_km2",
    "reach_area_km2",
    "area_ratio",
    "method",
]
STAGE_COLUMNS = [
    "ds_order",
    "year",
    "river_id",
    "n_scene_dates",
    "q_scene_median",
    "q_scene_min",
    "q_scene_max",
    "q_year_median",
    "q_record_median",
    "q_scene_pctile",
    "stage_flag",
]


# ─────────────────────────────────────────────────────────────────────
# Data source (real one below; tests pass a fake)
# ─────────────────────────────────────────────────────────────────────
class DischargeSource(Protocol):
    def links(self, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
        """Links with outlets in ``bbox`` (lon/lat): LINKNO, lat, lon and,
        if available, area_km2 (upstream drainage area)."""

    def daily(self, river_ids: list[int], start: str, end: str) -> pd.DataFrame:
        """Daily discharge (m³/s): index = date, one column per river id."""


class Geoglows:
    """GEOGLOWS v2 from the AWS Open Data bucket (anonymous access).

    The model table (``v2-model-table.parquet``) is registered as the
    dataset ``geoglows_table`` and downloaded once into the shared datasets
    folder.
    """

    _AREA_COLUMNS = ("DSContArea", "USContArea")

    def __init__(self):
        self._table: pd.DataFrame | None = None

    def _load_table(self) -> pd.DataFrame:
        if self._table is None:
            import pyarrow.parquet as pq

            from .data import fetch, get_dataset

            ds = get_dataset("geoglows_table")
            path = fetch("geoglows_table") / ds.expected_files[0]
            names = pq.ParquetFile(path).schema.names
            cols = ["LINKNO", "lat", "lon"] + [
                c for c in self._AREA_COLUMNS if c in names
            ]
            t = pd.read_parquet(path, columns=cols)
            area = next((c for c in self._AREA_COLUMNS if c in t.columns), None)
            if area:
                a = t[area].astype(float)
                # TDX-Hydro areas are in m²; guard against a km² table
                t["area_km2"] = a / 1e6 if a.median() > 1e4 else a
            self._table = t
        return self._table

    def links(self, bbox):
        t = self._load_table()
        x0, y0, x1, y1 = bbox
        sel = t["lon"].between(x0, x1) & t["lat"].between(y0, y1)
        return t.loc[sel].reset_index(drop=True)

    def daily(self, river_ids, start, end):
        try:
            import xarray as xr
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "the discharge step needs xarray, zarr, s3fs and pyarrow: "
                "conda install -c conda-forge xarray zarr s3fs pyarrow"
            ) from e
        opts = {"storage_options": {"anon": True}}
        try:  # zarr 3 needs to be told the store is zarr format 2
            ds = xr.open_zarr(DAILY_URI, zarr_format=2, **opts)
        except TypeError:  # older xarray / zarr 2
            ds = xr.open_zarr(DAILY_URI, **opts)
        with ds:
            q = ds["Q"].sel(river_id=list(river_ids), time=slice(start, end)).load()
        df = (
            q.to_dataframe()
            .reset_index()
            .pivot(index="time", columns="river_id", values="Q")
        )
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = "date"
        df.columns = [int(c) for c in df.columns]
        return df


# ─────────────────────────────────────────────────────────────────────
# Join
# ─────────────────────────────────────────────────────────────────────
def downstream_end(lines: gpd.GeoDataFrame, o: int):
    """Downstream end point of reach ``o``'s line.

    The end nearer the next reach downstream, or (last reach) the end
    farther from the previous one.
    """
    from shapely.geometry import Point
    from shapely.ops import linemerge

    g = lines.loc[o, "geometry"]
    if g.geom_type == "MultiLineString":
        g = linemerge(g)
    parts = list(g.geoms) if hasattr(g, "geoms") else [g]
    ends = [Point(c) for p in parts for c in (p.coords[0], p.coords[-1])]
    orders = list(lines.index)
    i = orders.index(o)
    if i + 1 < len(orders):
        nxt = lines.loc[orders[i + 1], "geometry"]
        return min(ends, key=nxt.distance)
    if i > 0:
        prv = lines.loc[orders[i - 1], "geometry"]
        return max(ends, key=prv.distance)
    return ends[-1]


def join_reaches(
    lines: gpd.GeoDataFrame, source: DischargeSource, search_km: float, tol: float
) -> pd.DataFrame:
    """One GEOGLOWS link per reach (see module docstring)."""
    lines = lines.sort_values("ds_order").set_index("ds_order")
    ends = {o: downstream_end(lines, o) for o in lines.index}
    ends_ll = gpd.GeoSeries(
        list(ends.values()), index=list(ends), crs=lines.crs
    ).to_crs(4326)
    lon0, lat0, lon1, lat1 = ends_ll.total_bounds
    pad = 2 * search_km / 111.0 / max(math.cos(math.radians((lat0 + lat1) / 2)), 0.2)
    links = source.links((lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad))
    if links.empty:
        raise RuntimeError("no GEOGLOWS links found near the river")
    pts = gpd.GeoSeries(
        gpd.points_from_xy(links["lon"], links["lat"]), crs=4326
    ).to_crs(lines.crs)
    has_area = "area_km2" in links.columns and "us_drain_ar" in lines.columns
    rows = []
    for o, end in ends.items():
        d_km = pts.distance(end).to_numpy() / 1000.0
        reach_area = float(lines.loc[o, "us_drain_ar"]) if has_area else np.nan
        ratio = (
            links["area_km2"].to_numpy() / reach_area
            if has_area and reach_area > 0
            else np.full(len(links), np.nan)
        )
        log_r = np.abs(np.log(np.where(ratio > 0, ratio, np.nan)))
        near = d_km <= search_km
        ok = near & (log_r <= math.log(tol)) if has_area else near
        if ok.any():
            i = int(np.flatnonzero(ok)[np.argmin(d_km[ok])])
            method = "area_matched" if has_area else "nearest"
        else:
            wider = d_km <= 2 * search_km
            if has_area and np.isfinite(log_r[wider]).any():
                cand = np.flatnonzero(wider & np.isfinite(log_r))
                i = int(cand[np.argmin(log_r[cand])])
                method = "best_area_2x_radius"
            else:
                i = int(np.argmin(d_km))
                method = "nearest_unmatched"
        rows.append(
            {
                "ds_order": int(o),
                "river_id": int(links.loc[i, "LINKNO"]),
                "outlet_lon": round(float(links.loc[i, "lon"]), 5),
                "outlet_lat": round(float(links.loc[i, "lat"]), 5),
                "dist_km": round(float(d_km[i]), 3),
                "link_area_km2": round(float(links.loc[i, "area_km2"]), 1)
                if has_area
                else np.nan,
                "reach_area_km2": round(reach_area, 1) if has_area else np.nan,
                "area_ratio": round(float(ratio[i]), 3) if has_area else np.nan,
                "method": method,
            }
        )
    return pd.DataFrame(rows, columns=JOIN_COLUMNS)


# ─────────────────────────────────────────────────────────────────────
# Stage statistics
# ─────────────────────────────────────────────────────────────────────
def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return np.nan
    v, w = values[ok], weights[ok]
    order = np.argsort(v)
    v, w = v[order], w[order]
    c = np.cumsum(w)
    half = 0.5 * c[-1]
    k = int(np.searchsorted(c, half))
    if k + 1 < len(v) and np.isclose(c[k], half):
        return float(0.5 * (v[k] + v[k + 1]))  # exactly half the weight below
    return float(v[k])


def stage_table(
    manifest: pd.DataFrame,
    join: pd.DataFrame,
    daily: pd.DataFrame,
    years: list[int],
    low: float,
    high: float,
) -> pd.DataFrame:
    """Scene-date discharge per reach-year (see module docstring)."""
    kept = manifest[manifest["kept"].astype(bool)].copy()
    kept["date"] = pd.to_datetime(kept["date"])
    rid = dict(zip(join["ds_order"], join["river_id"], strict=True))
    rows = []
    for o in sorted(rid):
        q = daily[rid[o]] if rid[o] in daily.columns else pd.Series(dtype=float)
        rec = q[q.index.year.isin(years)].dropna().to_numpy()
        rec_sorted = np.sort(rec)
        sub = kept[kept["ds_order"] == o]
        for y in years:
            s = sub[sub["year"] == y]
            row = {"ds_order": o, "year": y, "river_id": rid[o], "n_scene_dates": 0}
            qy = q[q.index.year == y]
            row["q_year_median"] = float(qy.median()) if len(qy) else np.nan
            row["q_record_median"] = float(np.median(rec)) if rec.size else np.nan
            if len(s) and len(q):
                qs = q.reindex(s["date"]).to_numpy(dtype=float)
                w = s["clear_frac"].to_numpy(dtype=float)
                wm = weighted_median(qs, w)
                row.update(
                    n_scene_dates=int(np.isfinite(qs).sum()),
                    q_scene_median=wm,
                    q_scene_min=float(np.nanmin(qs))
                    if np.isfinite(qs).any()
                    else np.nan,
                    q_scene_max=float(np.nanmax(qs))
                    if np.isfinite(qs).any()
                    else np.nan,
                )
                if np.isfinite(wm) and rec_sorted.size:
                    pct = (
                        100.0
                        * np.searchsorted(rec_sorted, wm, side="right")
                        / rec_sorted.size
                    )
                    row["q_scene_pctile"] = round(pct, 1)
                    row["stage_flag"] = (
                        "low_stage" if pct < low else "high_stage" if pct > high else ""
                    )
            rows.append(row)
    out = pd.DataFrame(rows, columns=STAGE_COLUMNS)
    for c in (
        "q_scene_median",
        "q_scene_min",
        "q_scene_max",
        "q_year_median",
        "q_record_median",
    ):
        out[c] = out[c].astype(float).round(2)
    out["stage_flag"] = out["stage_flag"].fillna("")
    return out


# ─────────────────────────────────────────────────────────────────────
# Step
# ─────────────────────────────────────────────────────────────────────
def _parameters(river: River) -> dict:
    return {
        "source": river.discharge.source,
        "daily_uri": DAILY_URI,
        "years": [river.masks.year_start, river.masks.year_end],
        **river.discharge.model_dump(mode="json"),
    }


def discharge(
    river: River, source: DischargeSource, config_path: Path | None = None
) -> StepResult:
    """Run step ``discharge`` for one river."""
    river.require("masks", "year_start", "year_end")
    p = river.paths
    for need, what in (
        (p.reach_lines, "the reaches step"),
        (p.scene_manifest, "extract_masks (raw product v2 writes the scene manifest)"),
    ):
        if not need.exists():
            return StepResult(
                river.name, STEP, FAILED, f"{need.name} not found — run {what}"
            )
    outputs = [p.discharge_join, p.discharge_daily, p.stage_table]
    params = _parameters(river)
    existing = check_existing(
        river.name, STEP, outputs, p.run_record(STEP), params, river.overwrite
    )
    if existing:
        return existing

    d = river.discharge
    years = list(range(river.masks.year_start, river.masks.year_end + 1))
    lines = gpd.read_file(p.reach_lines)
    if d.river_id is not None:
        join = pd.DataFrame(
            [
                {"ds_order": int(o), "river_id": d.river_id, "method": "manual"}
                for o in sorted(lines["ds_order"])
            ],
            columns=JOIN_COLUMNS,
        )
    else:
        join = join_reaches(lines, source, d.search_km, d.area_tolerance)
    ids = sorted(join["river_id"].unique().tolist())
    log.info("  %d reaches joined to %d GEOGLOWS links", len(join), len(ids))
    daily = source.daily(ids, f"{years[0]}-01-01", f"{years[-1]}-12-31")
    manifest = pd.read_csv(p.scene_manifest)
    stage = stage_table(
        manifest, join, daily, years, d.low_stage_pctile, d.high_stage_pctile
    )

    p.discharge_dir.mkdir(parents=True, exist_ok=True)
    join.to_csv(p.discharge_join, index=False)
    daily.round(3).to_csv(p.discharge_daily, index_label="date")
    stage.to_csv(p.stage_table, index=False)
    flags = stage["stage_flag"].value_counts().to_dict()
    summary = {
        "river_ids": [int(i) for i in ids],
        "join_methods": join["method"].value_counts().to_dict(),
        "reach_years_with_stage": int(stage["q_scene_pctile"].notna().sum()),
        "low_stage": int(flags.get("low_stage", 0)),
        "high_stage": int(flags.get("high_stage", 0)),
    }
    write_run_record(
        p.run_record(STEP),
        step=STEP,
        river=river.name,
        config_path=config_path,
        parameters=params,
        inputs={"reach_lines": p.reach_lines, "scene_manifest": p.scene_manifest},
        outputs={
            "join": p.discharge_join,
            "daily": p.discharge_daily,
            "stage": p.stage_table,
        },
        summary=summary,
    )
    unmatched = int((~join["method"].isin(["area_matched", "manual"])).sum())
    return StepResult(
        river.name,
        STEP,
        OK,
        f"{len(ids)} GEOGLOWS link(s); stage for {summary['reach_years_with_stage']} "
        f"reach-years ({summary['low_stage']} low, {summary['high_stage']} high)"
        + (
            f"; {unmatched} reach(es) joined without an area match" if unmatched else ""
        ),
        summary,
    )


def run(
    project: Project, rivers: list[River], source: DischargeSource | None = None
) -> list[StepResult]:
    """Run step ``discharge`` for ``rivers``."""
    source = source or Geoglows()
    results = []
    for r in rivers:
        log.info("%s", r.name)
        try:
            res = discharge(r, source, project.config_path)
        except Exception as e:  # report per river, keep going
            log.debug("%s: error", r.name, exc_info=True)
            res = StepResult(r.name, STEP, FAILED, f"{type(e).__name__}: {e}")
        results.append(res)
    return results
