"""Step 01 — extract each river's SWORD reaches and nodes.

For every river, walks SWORD's ``rch_id_dn`` topology from the upstream
reach ID to the downstream reach ID and writes the traced reaches and
their nodes to ``{data_root}/SWORD/Extracted_Rivers/{river}/``.

Rivers are grouped by (SWORD folder, continent) so each continental
reaches geopackage is read once per run. Nodes are read with an SQL
filter on the traced reach IDs rather than loading the whole continent.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pyogrio

from .config import Project, River, SwordSettings
from .paths import sword_continent_files
from .provenance import write_run_record
from .results import FAILED, OK, StepResult, check_existing

log = logging.getLogger(__name__)

STEP = "sword"


# ─────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────
def ensure_sword(
    sword_dir: Path, continent: str, settings: SwordSettings
) -> tuple[Path, Path]:
    """Return (reaches_gpkg, nodes_gpkg) for ``continent``, fetching SWORD if needed.

    If this continent's two geopackages are already present they are used
    as-is (even if other continents are missing). Otherwise the full SWORD
    archive for ``settings.version`` is fetched via the dataset registry
    (``rivmapper data fetch sword``), which checks its checksum.
    """
    reaches_path, nodes_path = sword_continent_files(
        sword_dir, continent, settings.version
    )
    if not (reaches_path.exists() and nodes_path.exists()):
        from .data import fetch

        fetch("sword", settings.version, folder=sword_dir)
    return reaches_path, nodes_path


# ─────────────────────────────────────────────────────────────────────
# Tracing
# ─────────────────────────────────────────────────────────────────────
SWORD_TYPES = {
    1: "river",
    3: "lake on river",
    4: "dam or waterfall",
    5: "unreliable topology",
    6: "ghost reach",
}


def _describe(rid: int, row) -> str:
    t = int(row["type"])
    name = SWORD_TYPES.get(t, "unknown")
    extra = []
    if "reach_len" in row:
        extra.append(f"{row['reach_len'] / 1000:.1f} km long")
    if "dist_out" in row:
        extra.append(f"{row['dist_out'] / 1000:.0f} km from outlet")
    return f"{rid} (type {t} = {name}" + (f", {', '.join(extra)})" if extra else ")")


def trace_reaches(
    us_reach_id: int,
    ds_reach_id: int,
    reaches: gpd.GeoDataFrame,
    allowed_types: list[int] | None = None,
    max_iter: int = 10_000,
) -> list[int]:
    """Walk SWORD topology from ``us_reach_id`` down to ``ds_reach_id``.

    At each reach the first non-zero entry of ``rch_id_dn`` is followed
    (a single-channel walk). ``rch_id_dn`` is a space-separated string
    when a reach has several downstream neighbours.

    The walk uses the full network, then checks the path against
    ``allowed_types``, so a disallowed reach (e.g. a short lake-flagged
    reach on the main channel) is reported by ID and type rather than
    looking like a break in the network.

    Args:
        us_reach_id: Upstream end of the river (inclusive).
        ds_reach_id: Downstream end of the river (inclusive).
        reaches: SWORD reaches with ``reach_id``, ``rch_id_dn``, ``dist_out``
            (and ``type`` if ``allowed_types`` is given). Pass the whole
            continent, not a type-filtered subset.
        allowed_types: SWORD ``type`` codes the path may contain; None = any.
        max_iter: Safety limit on the number of steps.

    Returns:
        Reach IDs ordered upstream → downstream.

    Raises:
        RuntimeError: If an endpoint is missing, the endpoints are swapped,
            the walk leaves the network or never reaches ``ds_reach_id``,
            or the path includes a reach type not in ``allowed_types``.
    """
    by_id = reaches.set_index("reach_id")
    dn_lookup = by_id["rch_id_dn"].to_dict()
    dist_lookup = by_id["dist_out"].to_dict()

    for label, rid in (
        ("us_sword_reach_id", us_reach_id),
        ("ds_sword_reach_id", ds_reach_id),
    ):
        if rid not in dist_lookup:
            raise RuntimeError(
                f"{label} {rid} is not in this SWORD continent. Check the ID "
                f"(SWORD reach IDs have 11 digits) and sword_cont_abb."
            )
    us_dist, ds_dist = dist_lookup[us_reach_id], dist_lookup[ds_reach_id]
    if us_dist <= ds_dist:
        raise RuntimeError(
            f"us_sword_reach_id {us_reach_id} (dist_out={us_dist:.0f} m) is not "
            f"upstream of ds_sword_reach_id {ds_reach_id} "
            f"(dist_out={ds_dist:.0f} m). The endpoints may be swapped."
        )

    path = [us_reach_id]
    current = us_reach_id
    for _ in range(max_iter):
        if current == ds_reach_id:
            break
        dn_ids = [int(x) for x in str(dn_lookup[current]).split() if int(x) != 0]
        if not dn_ids:
            raise RuntimeError(
                f"Reach {current} has no downstream neighbour; the walk reached "
                f"an outlet without passing {ds_reach_id}. Check the endpoint "
                f"pair (the downstream reach may be on a different branch)."
            )
        nxt = dn_ids[0]
        if nxt not in dn_lookup:
            raise RuntimeError(
                f"Reach {current} drains to {nxt}, which is not in this SWORD "
                f"continent file (topology break, or the river crosses into "
                f"another continent's file)."
            )
        current = nxt
        path.append(current)
    else:
        raise RuntimeError(
            f"Did not reach {ds_reach_id} within {max_iter} steps. "
            f"Possible topology loop or wrong endpoint pair."
        )

    if allowed_types is not None and "type" in by_id.columns:
        allowed = {int(t) for t in allowed_types}
        bad = [rid for rid in path if int(by_id.at[rid, "type"]) not in allowed]
        if bad:
            listed = "; ".join(_describe(rid, by_id.loc[rid]) for rid in bad[:10])
            more = f" (+{len(bad) - 10} more)" if len(bad) > 10 else ""
            needed = sorted(allowed | {int(by_id.at[rid, "type"]) for rid in bad})
            raise RuntimeError(
                f"The path ({len(path)} reaches) passes through {len(bad)} reach(es) "
                f"whose SWORD type is not in river_types {sorted(allowed)}: "
                f"{listed}{more}. If these are part of the river channel, add "
                f"their type to `sword: river_types:` in rivmapper.yaml "
                f"(e.g. river_types: {needed})."
            )
    return path


# ─────────────────────────────────────────────────────────────────────
# Per-river extraction
# ─────────────────────────────────────────────────────────────────────
def _parameters(river: River) -> dict:
    return {
        "us_sword_reach_id": river.us_reach_id,
        "ds_sword_reach_id": river.ds_reach_id,
        "sword_cont_abb": river.continent,
        "sword_version": river.sword.version,
        "river_types": list(river.sword.river_types),
    }


def load_continent_reaches(reaches_path: Path) -> gpd.GeoDataFrame:
    """Read a continental SWORD reaches gpkg (all reach types)."""
    return pyogrio.read_dataframe(reaches_path)


def extract_river(
    river: River,
    reaches: gpd.GeoDataFrame,
    nodes_path: Path,
    config_path: Path | None = None,
    check_outputs: bool = True,
) -> StepResult:
    """Trace one river through ``reaches`` and write its SWORD subset.

    Args:
        river: Resolved river settings.
        reaches: Continental SWORD reaches (already filtered to river types).
        nodes_path: Continental SWORD nodes geopackage (read with a filter).
        config_path: Recorded in the run record for provenance.
        check_outputs: If True, skip when current outputs already exist.
    """
    p = river.paths
    params = _parameters(river)
    if check_outputs:
        existing = check_existing(
            river.name,
            STEP,
            [p.sword_reaches, p.sword_reaches_csv, p.sword_nodes],
            p.run_record(STEP),
            params,
            river.overwrite,
        )
        if existing:
            return existing

    try:
        path = trace_reaches(
            river.us_reach_id,
            river.ds_reach_id,
            reaches,
            allowed_types=list(river.sword.river_types),
        )
    except RuntimeError as e:
        return StepResult(river.name, STEP, FAILED, str(e))

    order = {rid: i + 1 for i, rid in enumerate(path)}
    subset = reaches[reaches["reach_id"].isin(path)].copy()
    subset["reach_order"] = subset["reach_id"].map(order)
    subset = subset.sort_values("reach_order").reset_index(drop=True)

    id_list = ",".join(str(r) for r in path)
    nodes = pyogrio.read_dataframe(nodes_path, where=f"reach_id IN ({id_list})")
    nodes["reach_order"] = nodes["reach_id"].map(order)
    # dist_out decreases downstream, so descending within a reach = US→DS.
    nodes = nodes.sort_values(
        ["reach_order", "dist_out"], ascending=[True, False]
    ).reset_index(drop=True)
    nodes["node_order"] = range(1, len(nodes) + 1)

    p.sword_dir.mkdir(parents=True, exist_ok=True)
    subset.to_file(p.sword_reaches, driver="GPKG", engine="pyogrio")
    subset.drop(columns="geometry").to_csv(p.sword_reaches_csv, index=False)
    nodes.to_file(p.sword_nodes, driver="GPKG", engine="pyogrio")

    summary = {"n_sword_reaches": len(path), "n_nodes": len(nodes)}
    write_run_record(
        p.run_record(STEP),
        step=STEP,
        river=river.name,
        config_path=config_path,
        parameters=params,
        inputs={
            "sword_reaches": nodes_path.with_name(
                nodes_path.name.replace("_nodes_", "_reaches_")
            ),
            "sword_nodes": nodes_path,
        },
        outputs={
            "reaches": p.sword_reaches,
            "reaches_csv": p.sword_reaches_csv,
            "nodes": p.sword_nodes,
        },
        summary=summary,
    )
    return StepResult(
        river.name,
        STEP,
        OK,
        f"{len(path)} SWORD reaches, {len(nodes)} nodes",
        summary,
    )


def run(project: Project, rivers: list[River]) -> list[StepResult]:
    """Run step 01 for ``rivers``, loading each SWORD continent once."""
    groups: dict[tuple[Path, str], list[River]] = defaultdict(list)
    for r in rivers:
        groups[(r.sword_dir, r.continent)].append(r)

    results: list[StepResult] = []
    for (sword_dir, continent), group in groups.items():
        settings = group[0].sword
        # Skip loading SWORD entirely if every river here is already done.
        pending = []
        for r in group:
            p = r.paths
            done = check_existing(
                r.name,
                STEP,
                [p.sword_reaches, p.sword_reaches_csv, p.sword_nodes],
                p.run_record(STEP),
                _parameters(r),
                r.overwrite,
            )
            if done:
                results.append(done)
            else:
                pending.append(r)
        if not pending:
            continue

        try:
            reaches_path, nodes_path = ensure_sword(sword_dir, continent, settings)
            log.info("Loading %s", reaches_path.name)
            reaches = load_continent_reaches(reaches_path)
        except Exception as e:  # noqa: BLE001 — report per river, keep going
            results.extend(
                StepResult(r.name, STEP, FAILED, f"could not load SWORD: {e}")
                for r in pending
            )
            continue

        for r in pending:
            try:
                res = extract_river(
                    r, reaches, nodes_path, project.config_path, check_outputs=False
                )
            except Exception as e:  # noqa: BLE001
                log.exception("%s: unexpected error", r.name)
                res = StepResult(r.name, STEP, FAILED, f"{type(e).__name__}: {e}")
            results.append(res)
        del reaches

    order = {r.name: i for i, r in enumerate(rivers)}
    return sorted(results, key=lambda res: order[res.river])
