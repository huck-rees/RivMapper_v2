"""Step 02 — divide each river into reaches and build reach polygons.

Default (``length_method: valley_widths``): reaches are ``widths_per_reach``
(k = 80) channel widths long measured along a smoothed *valley line*, with
width from a per-river power law W(A) fitted to SWORD node widths, so
reaches hold a consistent number of meander wavelengths and lengthen
downstream (see ``valley.py`` and docs/methods/reaches.md). The v1 rule,
channel length = ``length_modulator × len_slope × facc + len_intercept``,
is available as ``length_method: drainage_regression``.

Each reach centreline is buffered by ``buffer_width`` × its valley length
(channel length for drainage_regression). Neighbouring buffers are
separated by ``boundary_method``:

``nearest_centerline`` (default)
    Every point goes to the reach whose centreline is closest (a Voronoi
    partition of points sampled along the centrelines), so polygons never
    overlap or leave wedge gaps, even where the channel loops back.
``cut_lines``
    Perpendicular cuts at every boundary (original method); can overlap
    where the channel doubles back.

The river head and mouth are squared off with perpendicular cuts.

Outputs (``{data_root}/{river}/reaches/``):
    ``{river}_reach_lines.gpkg``, ``{river}_reach_polygons.gpkg`` (fields
    ``ds_order``, ``reach_len``, ``ds_dist``, ``reach_ar``, ``us_drain_ar``,
    ``valley_len``, ``valley_dist``, ``sinuosity``, ``width_ref``) and
    ``{river}_valley_line.gpkg``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import split, substring

from .config import Project, ReachSettings, River
from .provenance import write_run_record
from .results import FAILED, OK, StepResult, check_existing
from .valley import fit_width_model, reach_breaks, valley_line

log = logging.getLogger(__name__)

STEP = "reaches"
FIELDS = [
    "ds_order",
    "reach_len",
    "ds_dist",
    "reach_ar",
    "us_drain_ar",
    "valley_len",
    "valley_dist",
    "sinuosity",
    "width_ref",
]


# ─────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────
def utm_crs_for(gdf: gpd.GeoDataFrame) -> str:
    """EPSG code of the UTM zone containing the centre of ``gdf``'s bounds."""
    geo = gdf if (gdf.crs is not None and gdf.crs.is_geographic) else gdf.to_crs(4326)
    minx, miny, maxx, maxy = geo.total_bounds
    lon, lat = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    zone = int((lon + 180.0) / 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def merge_to_line(
    sword_sorted: gpd.GeoDataFrame, gap_tolerance_m: float = 10.0
) -> LineString:
    """Join SWORD reaches (sorted upstream → downstream) into one LineString.

    Reach 1 is oriented by comparing its endpoints with reach 2's; each
    later reach is reversed if its end is nearer the running line's end
    than its start is. Gaps above ``gap_tolerance_m`` are bridged with a
    straight segment and logged.
    """
    if len(sword_sorted) == 0:
        return LineString()

    coord_lists = []
    for geom in sword_sorted.geometry:
        if isinstance(geom, MultiLineString):
            geom = max(geom.geoms, key=lambda g: g.length)
        coord_lists.append(list(geom.coords))

    if len(coord_lists) >= 2:
        s1, e1 = coord_lists[0][0], coord_lists[0][-1]
        s2, e2 = coord_lists[1][0], coord_lists[1][-1]
        if min(math.dist(s1, s2), math.dist(s1, e2)) < min(
            math.dist(e1, s2), math.dist(e1, e2)
        ):
            coord_lists[0] = coord_lists[0][::-1]

    coords = list(coord_lists[0])
    for i, pts in enumerate(coord_lists[1:], start=1):
        prev_end = coords[-1]
        if math.dist(pts[-1], prev_end) < math.dist(pts[0], prev_end):
            pts = pts[::-1]
        gap = math.dist(pts[0], prev_end)
        if gap > gap_tolerance_m:
            row = sword_sorted.iloc[i]
            log.warning(
                "  %.1f m gap before reach_id %s (reach_order %s)",
                gap,
                row.get("reach_id"),
                row.get("reach_order"),
            )
            coords.extend(pts)
        else:
            coords.extend(pts[1:])
    return LineString(coords)


def split_at_distances(line: LineString, distances: list[float]) -> list[LineString]:
    """Split ``line`` at increasing cumulative ``distances`` along it.

    ``distances`` are the interior split points (exclusive of 0 and the
    line length). Returns ``len(distances) + 1`` pieces.
    """
    bounds = [0.0, *distances, line.length]
    return [substring(line, a, b) for a, b in zip(bounds[:-1], bounds[1:])]


def sample_facc(distance_m: float, cum_end_m: np.ndarray, facc: np.ndarray) -> float:
    """``facc`` of the SWORD reach containing cumulative ``distance_m``."""
    i = int(np.searchsorted(cum_end_m, distance_m, side="left"))
    return float(facc[min(i, len(facc) - 1)])


def compute_split_distances(
    cum_end_m: np.ndarray,
    facc: np.ndarray,
    settings: ReachSettings,
) -> list[float]:
    """Cumulative distances (m) of the upstream end of each new reach.

    Walks downstream from 0, setting each reach's length from ``facc``
    sampled at its upstream end. First entry is 0; the last is the start
    of the final (possibly short) reach.
    """
    total = float(cum_end_m[-1])
    starts = [0.0]
    while True:
        f = sample_facc(starts[-1], cum_end_m, facc)
        length = (
            settings.length_modulator * settings.len_slope * f + settings.len_intercept
        )
        nxt = starts[-1] + length
        if nxt >= total:
            return starts
        starts.append(nxt)


def smoothed_tangent(
    line: LineString, position_m: float, window_m: float, n_samples: int = 21
) -> np.ndarray:
    """Unit valley-direction vector at ``position_m``, pointing downstream.

    Fits a principal axis (SVD) through ``n_samples`` points sampled along
    ``line`` within ``window_m`` centred on ``position_m`` (one-sided at
    the ends), which averages out meander-scale curvature.
    """
    total = line.length
    start = max(0.0, position_m - window_m / 2.0)
    end = min(total, position_m + window_m / 2.0)
    if end - start < 1.0:
        end = min(total, start + 1.0)

    pts = [line.interpolate(d) for d in np.linspace(start, end, n_samples)]
    xy = np.array([(p.x, p.y) for p in pts])
    centred = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    tangent = vt[0]
    if np.dot(tangent, xy[-1] - xy[0]) < 0:
        tangent = -tangent
    return tangent


def make_cut_line(point: Point, tangent: np.ndarray, half_length: float) -> LineString:
    """Line through ``point`` perpendicular to ``tangent``, ``2 * half_length`` long."""
    perp = np.array([-tangent[1], tangent[0]])
    return LineString(
        [
            (point.x - perp[0] * half_length, point.y - perp[1] * half_length),
            (point.x + perp[0] * half_length, point.y + perp[1] * half_length),
        ]
    )


def clip_by_cut(polygon, cut: LineString, anchor: Point):
    """Split ``polygon`` with ``cut`` and keep the piece containing ``anchor``.

    Falls back to the unsplit polygon if the cut does not fully cross it.
    Returns (polygon, clipped: bool).
    """
    if polygon.is_empty:
        return polygon, False
    try:
        result = split(polygon, cut)
    except Exception:  # noqa: BLE001 — shapely raises several types here
        return polygon, False
    pieces = [
        g
        for g in getattr(result, "geoms", [result])
        if g.geom_type in ("Polygon", "MultiPolygon")
    ]
    if len(pieces) < 2:
        return polygon, False
    for piece in pieces:
        if piece.contains(anchor):
            return piece, True
    return min(pieces, key=lambda g: g.distance(anchor)), True


def compute_cut_lines(
    line: LineString,
    reach_lens: list[float],
    ds_dists: list[float],
    settings: ReachSettings,
    max_buffer_m: float,
) -> list[LineString]:
    """One cut line per reach boundary, head and mouth included.

    Cut ``i`` is the upstream boundary of reach ``i``; cut ``i + 1`` its
    downstream boundary. The smoothing window at a boundary scales with
    the longer of the two adjacent reaches.
    """
    positions = [*ds_dists, line.length]
    half_length = settings.cut_extent_factor * max_buffer_m
    n = len(reach_lens)
    cuts = []
    for i, pos in enumerate(positions):
        if i == 0:
            local = reach_lens[0]
        elif i == n:
            local = reach_lens[-1]
        else:
            local = max(reach_lens[i - 1], reach_lens[i])
        tangent = smoothed_tangent(
            line,
            pos,
            settings.smoothing_window_multiplier * local,
            settings.n_tangent_samples,
        )
        cuts.append(make_cut_line(line.interpolate(pos), tangent, half_length))
    return cuts


def nearest_centerline_cells(
    lines: list[LineString], extent, spacing_m: float = 100.0
) -> list:
    """Partition ``extent`` so each part is nearest to one reach line.

    Points are sampled every ``spacing_m`` along each reach centreline and
    a Voronoi diagram of all points is dissolved by reach. Cell ``i`` is the
    area closer to reach ``i``'s centreline than to any other reach's (to
    within the sampling spacing), so cells never overlap and their
    boundaries follow the mid-line between neighbouring channels, including
    where a channel loops back on itself.
    """
    import shapely

    pts, labels = [], []
    n = len(lines)
    for i, ln in enumerate(lines):
        k = max(2, int(np.ceil(ln.length / spacing_m)) + 1)
        # Shared junction points belong to the upstream reach only.
        ds = np.linspace(0.0, ln.length, k, endpoint=(i == n - 1))
        pts.extend(shapely.line_interpolate_point(ln, ds))
        labels.extend([i] * len(ds))
    pts = np.asarray(pts, dtype=object)
    labels = np.asarray(labels)

    cells = shapely.get_parts(
        shapely.voronoi_polygons(shapely.multipoints(pts), extend_to=extent)
    )
    point_idx, cell_idx = shapely.STRtree(cells).query(pts, predicate="within")
    owner = np.full(len(cells), -1)
    owner[cell_idx] = labels[point_idx]

    out = []
    for i in range(n):
        mine = cells[owner == i]
        try:
            merged = shapely.coverage_union_all(mine)
        except Exception:  # noqa: BLE001 — fall back if cells are not a clean coverage
            merged = shapely.union_all(mine)
        out.append(merged)
    return out


def _total_overlap_km2(geoms: list) -> float:
    """Sum of pairwise intersection areas between all polygons."""
    import shapely

    tree = shapely.STRtree(geoms)
    a, b = tree.query(geoms, predicate="intersects")
    keep = a < b
    return (
        float(
            sum(geoms[i].intersection(geoms[j]).area for i, j in zip(a[keep], b[keep]))
        )
        / 1e6
    )


# ─────────────────────────────────────────────────────────────────────
# Per-river processing
# ─────────────────────────────────────────────────────────────────────
@dataclass
class ReachBuild:
    """Result of ``build_reaches``."""

    lines: gpd.GeoDataFrame
    polygons: gpd.GeoDataFrame
    diagnostics: dict
    valley: gpd.GeoDataFrame | None = None  # valley line (valley_widths only)


def _order_nodes(nodes: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    col = "node_order" if "node_order" in nodes.columns else None
    if col is None:
        nodes = nodes.sort_values(["reach_order", "dist_out"], ascending=[True, False])
    else:
        nodes = nodes.sort_values(col)
    return nodes.to_crs(crs).reset_index(drop=True)


def _valley_splits(line, sword_utm, nodes_utm, settings, diag):
    """Split distances along ``line`` from valley-length reach breaks."""
    import shapely

    x = nodes_utm.geometry.x.to_numpy()
    y = nodes_utm.geometry.y.to_numpy()
    width = nodes_utm["width"].to_numpy(dtype=float)
    area = nodes_utm["facc"].to_numpy(dtype=float).copy()
    area[~(area > 0)] = np.nan
    area = pd.Series(area).interpolate(limit_direction="both").to_numpy()

    model = fit_width_model(width, area)
    vl = valley_line(
        x, y, width, settings.valley_sigma_widths, settings.valley_sinuosity_factor
    )
    w_ref = model(area)
    starts = reach_breaks(
        vl.valley_dist, w_ref, settings.widths_per_reach, settings.final_merge_fraction
    )

    # Position of every node along the merged channel line, forced monotonic,
    # then valley-distance breaks interpolated to channel distance.
    s_line = shapely.line_locate_point(line, shapely.points(x, y))
    s_line = np.maximum.accumulate(s_line)
    s_line[0] = 0.0
    v = vl.valley_dist
    breaks = [float(np.interp(d, v, s_line)) for d in starts[1:]]
    ok = [0.0 < b < line.length for b in breaks]
    starts = [0.0] + [d for d, k in zip(starts[1:], ok) if k]
    breaks = [b for b, k in zip(breaks, ok) if k]

    ends = [*starts[1:], float(v[-1])]
    per_reach = {
        "valley_len": [e - d for d, e in zip(starts, ends)],
        "valley_dist": list(starts),
        "width_ref": [float(np.interp(d, v, w_ref)) for d in starts],
        "us_drain_ar": [float(np.interp(d, v, area)) for d in starts],
    }
    valley_gdf = gpd.GeoDataFrame(
        {"valley_len": [float(v[-1])]},
        geometry=[LineString(np.column_stack([vl.x, vl.y]))],
        crs=nodes_utm.crs,
    )
    diag.update(
        {
            "length_method": "valley_widths",
            "widths_per_reach": settings.widths_per_reach,
            "width_model": {
                "form": "W = coef * A^exponent (W m, A km2)",
                "coef": round(model.coef, 4),
                "exponent": round(model.exponent, 4),
                "method": model.method,
                "n_nodes": model.n_nodes,
            },
            "valley_length_km": round(float(v[-1]) / 1000, 3),
            "channel_length_km_nodes": round(float(vl.channel_dist[-1]) / 1000, 3),
            "overall_sinuosity": round(float(vl.channel_dist[-1] / max(v[-1], 1)), 3),
        }
    )
    return breaks, per_reach, vl, valley_gdf


def build_reaches(
    sword: gpd.GeoDataFrame,
    settings: ReachSettings,
    nodes: gpd.GeoDataFrame | None = None,
) -> ReachBuild:
    """Build reach lines and polygons from one river's SWORD reaches.

    Args:
        sword: Extracted SWORD reaches with ``reach_order`` and ``facc``.
        settings: Resolved reach settings (``buffer_width`` must be set).
        nodes: Extracted SWORD nodes (``width``, ``facc``, ``node_order``);
            required for ``length_method: valley_widths``.

    Returns:
        ``ReachBuild`` with lines and polygons in local UTM, diagnostics and,
        for valley_widths, the valley line.
    """
    sword = sword.sort_values("reach_order").reset_index(drop=True)
    utm = utm_crs_for(sword)
    sword = sword.to_crs(utm)

    seg_len = sword.geometry.length.to_numpy()
    cum_end = np.cumsum(seg_len)
    facc = sword["facc"].to_numpy(dtype=float)

    line = merge_to_line(sword, settings.gap_tolerance_m)
    diag: dict = {}
    valley_gdf = None
    vl = None
    if settings.length_method == "valley_widths":
        if nodes is None or nodes.empty:
            raise RuntimeError(
                "length_method valley_widths needs the SWORD nodes file "
                "(re-run the sword step)"
            )
        nodes_utm = _order_nodes(nodes, utm)
        breaks, per_reach, vl, valley_gdf = _valley_splits(
            line, sword, nodes_utm, settings, diag
        )
        pieces = split_at_distances(line, breaks)
    else:
        starts = compute_split_distances(cum_end, facc, settings)
        pieces = split_at_distances(line, starts[1:])
        per_reach = None
        diag["length_method"] = "drainage_regression"

    lines = gpd.GeoDataFrame(geometry=pieces, crs=utm)
    lines["reach_len"] = lines.geometry.length
    if per_reach is not None:
        for k, vals in per_reach.items():
            lines[k] = vals
    lines = lines[lines["reach_len"] > 0.0].reset_index(drop=True)
    if lines.empty:
        raise RuntimeError("no non-empty reaches after splitting")

    trimmed_m = 0.0
    if per_reach is None and len(lines) >= 2:
        last, prev = lines["reach_len"].iloc[-1], lines["reach_len"].iloc[-2]
        if last < settings.ds_trim_fraction * prev:
            trimmed_m = float(last)
            lines = lines.iloc[:-1].reset_index(drop=True)

    lines["ds_order"] = np.arange(1, len(lines) + 1)
    lines["ds_dist"] = lines["reach_len"].cumsum().shift(1, fill_value=0.0)
    if per_reach is None:
        lines["us_drain_ar"] = [sample_facc(d, cum_end, facc) for d in lines["ds_dist"]]
        for k in ("valley_len", "valley_dist", "width_ref"):
            lines[k] = np.nan
    lines["sinuosity"] = lines["reach_len"] / lines["valley_len"]

    # Buffer radius: fraction of valley length (valley_widths) or of channel
    # length (drainage_regression).
    basis = lines["valley_len"] if per_reach is not None else lines["reach_len"]
    radii = (basis * settings.buffer_width).to_numpy()

    kept = substring(line, 0.0, float(lines["reach_len"].sum())) if trimmed_m else line
    max_buffer = float(radii.max())
    cuts = compute_cut_lines(
        kept,
        lines["reach_len"].tolist(),
        lines["ds_dist"].tolist(),
        settings,
        max_buffer,
    )
    if vl is not None:
        # Square off head and mouth perpendicular to the valley line.
        cuts[0] = _valley_end_cut(vl, lines, head=True, half_length=3 * max_buffer)
        cuts[-1] = _valley_end_cut(vl, lines, head=False, half_length=3 * max_buffer)

    reach_geoms = list(lines.geometry)
    buffers = [g.buffer(r) for g, r in zip(reach_geoms, radii)]
    mids = [g.interpolate(g.length / 2.0) for g in reach_geoms]
    n = len(reach_geoms)
    polys, unclipped, multipart = [], [], []

    if settings.boundary_method == "nearest_centerline":
        # Neighbours are separated by the nearest-centreline partition; cut
        # lines are only used to square off the river head and mouth.
        from shapely.geometry import box

        extent = box(*gpd.GeoSeries(buffers).total_bounds).buffer(1.0)
        cells = nearest_centerline_cells(
            reach_geoms, extent, settings.partition_spacing_m
        )
        for i in range(n):
            poly = buffers[i].intersection(cells[i])
            if i == 0:
                poly, ok = clip_by_cut(poly, cuts[0], mids[0])
                if not ok:
                    unclipped.append(i + 1)
            if i == n - 1:
                poly, ok = clip_by_cut(poly, cuts[-1], mids[-1])
                if not ok and (i + 1) not in unclipped:
                    unclipped.append(i + 1)
            if poly.geom_type == "MultiPolygon":
                multipart.append(i + 1)
            polys.append(poly)
    else:  # "cut_lines": perpendicular cuts at every boundary (original method)
        for i in range(n):
            poly, ok_us = clip_by_cut(buffers[i], cuts[i], mids[i])
            poly, ok_ds = clip_by_cut(poly, cuts[i + 1], mids[i])
            if not (ok_us and ok_ds):
                unclipped.append(i + 1)
            polys.append(poly)

    polygons = lines.copy()
    polygons["geometry"] = polys
    polygons = polygons.set_geometry("geometry", crs=utm)
    polygons["reach_ar"] = polygons.geometry.area / 1e6
    lines["reach_ar"] = polygons["reach_ar"].to_numpy()

    if unclipped:
        log.warning(
            "  cut line did not fully cross the buffer for reach(es) %s; "
            "those polygons keep an unclipped end — inspect in QGIS or raise "
            "cut_extent_factor",
            unclipped,
        )
    overlap_km2 = _total_overlap_km2(list(polygons.geometry))
    if overlap_km2 > 0.01:
        log.warning(
            "  reach polygons overlap by %.2f km² in total — likely crossing "
            "cut lines on a sharp bend; use boundary_method: nearest_centerline",
            overlap_km2,
        )
    if multipart:
        log.info(
            "  reach(es) %s have detached pieces (floodplain nearest to their "
            "centreline but cut off by a neighbour's loop)",
            multipart,
        )

    diagnostics = {
        **diag,
        "n_reaches": int(len(lines)),
        "river_length_km": round(float(cum_end[-1]) / 1000, 3),
        "trimmed_tail_km": round(trimmed_m / 1000, 3),
        "boundary_method": settings.boundary_method,
        "unclipped_reaches": unclipped,
        "multipart_reaches": multipart,
        "overlap_km2": round(overlap_km2, 4),
        "utm_crs": utm,
    }
    if per_reach is not None:
        nw = (lines["valley_len"] / lines["width_ref"]).to_numpy()
        diagnostics["valley_len_km_range"] = [
            round(float(lines["valley_len"].min()) / 1000, 2),
            round(float(lines["valley_len"].max()) / 1000, 2),
        ]
        diagnostics["widths_per_reach_range"] = [
            round(float(nw.min()), 1),
            round(float(nw.max()), 1),
        ]
    return ReachBuild(
        lines[FIELDS + ["geometry"]],
        polygons[FIELDS + ["geometry"]],
        diagnostics,
        valley_gdf,
    )


def _valley_end_cut(vl, lines, head: bool, half_length: float) -> LineString:
    """Cut line at the river head or mouth, perpendicular to the valley line."""
    v = vl.valley_dist
    span = float(lines["valley_len"].iloc[0 if head else -1]) * 0.5
    if head:
        i0, i1 = 0, int(np.searchsorted(v, v[0] + span))
        anchor = 0
    else:
        i0, i1 = int(np.searchsorted(v, v[-1] - span)), len(v) - 1
        anchor = len(v) - 1
    i1 = max(i1, i0 + 1)
    i1 = min(i1, len(v) - 1)
    t = np.array([vl.x[i1] - vl.x[i0], vl.y[i1] - vl.y[i0]])
    t = t / max(np.hypot(*t), 1e-9)
    end = lines.geometry.iloc[0 if head else -1]
    pt = Point(end.coords[0] if head else end.coords[-1])
    del anchor
    return make_cut_line(pt, t, half_length)


def _parameters(river: River) -> dict:
    return river.reaches.model_dump(mode="json")


def create_reaches(
    river: River, config_path: Path | None = None, check_outputs: bool = True
) -> StepResult:
    """Run step 02 for one river."""
    p = river.paths
    params = _parameters(river)
    if check_outputs:
        existing = check_existing(
            river.name,
            STEP,
            [p.reach_lines, p.reach_polygons],
            p.run_record(STEP),
            params,
            river.overwrite,
        )
        if existing:
            return existing

    river.require("reaches", "buffer_width")
    if not p.sword_reaches.exists():
        return StepResult(
            river.name,
            STEP,
            FAILED,
            f"SWORD reaches not found at {p.sword_reaches} — run the sword step",
        )
    sword = gpd.read_file(p.sword_reaches, engine="pyogrio")
    if sword.empty or "reach_order" not in sword.columns:
        return StepResult(
            river.name,
            STEP,
            FAILED,
            f"{p.sword_reaches.name} is empty or lacks `reach_order`",
        )

    nodes = None
    if river.reaches.length_method == "valley_widths":
        if not p.sword_nodes.exists():
            return StepResult(
                river.name,
                STEP,
                FAILED,
                f"SWORD nodes not found at {p.sword_nodes} — run the sword step",
            )
        nodes = gpd.read_file(p.sword_nodes, engine="pyogrio")
    built = build_reaches(sword, river.reaches, nodes)
    lines, polygons, diag = built.lines, built.polygons, built.diagnostics

    p.reaches_dir.mkdir(parents=True, exist_ok=True)
    lines.to_file(p.reach_lines, driver="GPKG", engine="pyogrio")
    polygons.to_file(p.reach_polygons, driver="GPKG", engine="pyogrio")
    outputs = {"reach_lines": p.reach_lines, "reach_polygons": p.reach_polygons}
    if built.valley is not None:
        built.valley.to_file(p.valley_line, driver="GPKG", engine="pyogrio")
        outputs["valley_line"] = p.valley_line
    write_run_record(
        p.run_record(STEP),
        step=STEP,
        river=river.name,
        config_path=config_path,
        parameters=params,
        inputs={"sword_reaches": p.sword_reaches, "sword_nodes": p.sword_nodes},
        outputs=outputs,
        summary=diag,
    )
    return StepResult(river.name, STEP, OK, f"{diag['n_reaches']} reaches", diag)


def run(project: Project, rivers: list[River]) -> list[StepResult]:
    """Run step 02 for ``rivers``."""
    results = []
    for r in rivers:
        log.info("%s", r.name)
        try:
            res = create_reaches(r, project.config_path)
        except Exception as e:  # noqa: BLE001 — report per river, keep going
            log.debug("%s: error", r.name, exc_info=True)
            res = StepResult(r.name, STEP, FAILED, f"{type(e).__name__}: {e}")
        results.append(res)
    return results
