"""Step 01 — extract each river's SWORD reaches and nodes.

For every river, walks SWORD's ``rch_id_dn`` topology from the upstream
reach ID to the downstream reach ID and writes the traced reaches and
their nodes to ``{data_root}/SWORD/Extracted_Rivers/{river}/``.

Rivers are grouped by (SWORD folder, continent) so each continental
reaches geopackage is read once per run. Nodes are read with an SQL
filter on the traced reach IDs rather than loading the whole continent.
"""

from __future__ import annotations

import hashlib
import logging
import zipfile
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
# Download
# ─────────────────────────────────────────────────────────────────────
def ensure_sword_downloaded(
    sword_dir: Path, continent: str, settings: SwordSettings
) -> tuple[Path, Path]:
    """Return (reaches_gpkg, nodes_gpkg) for ``continent``, downloading if needed.

    The Zenodo deposit is a single zip holding every continent, so a
    download extracts all continental geopackages at once. The zip is
    downloaded to a ``.part`` file and only renamed once complete, so an
    interrupted download is never mistaken for a finished one.

    Raises:
        RuntimeError: If the MD5 checksum does not match ``settings.md5``.
    """
    reaches_path, nodes_path = sword_continent_files(
        sword_dir, continent, settings.version
    )
    if reaches_path.exists() and nodes_path.exists():
        return reaches_path, nodes_path

    import requests
    from tqdm.auto import tqdm

    sword_dir.mkdir(parents=True, exist_ok=True)
    zip_path = sword_dir / Path(settings.download_url).name
    part_path = zip_path.with_suffix(zip_path.suffix + ".part")
    log.info(
        "SWORD %s geopackages for '%s' not found in %s; downloading %s",
        settings.version,
        continent,
        sword_dir,
        settings.download_url,
    )

    with requests.get(settings.download_url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        md5 = hashlib.md5()
        with (
            open(part_path, "wb") as f,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading SWORD",
            ) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                md5.update(chunk)
                bar.update(len(chunk))

    if settings.md5 and md5.hexdigest() != settings.md5:
        raise RuntimeError(
            f"SWORD download checksum mismatch (expected {settings.md5}, got "
            f"{md5.hexdigest()}). The partial file was kept at {part_path}. "
            f"If Zenodo has republished the archive, update `sword: md5:` "
            f"(or set it to null) in rivmapper.yaml."
        )
    part_path.replace(zip_path)

    log.info("Extracting SWORD geopackages to %s", sword_dir)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name.endswith(".gpkg") and not name.startswith("._"):
                with zf.open(member) as src, open(sword_dir / name, "wb") as dst:
                    while block := src.read(1 << 20):
                        dst.write(block)
    zip_path.unlink()

    if not (reaches_path.exists() and nodes_path.exists()):
        raise RuntimeError(
            f"Download finished but {reaches_path.name} / {nodes_path.name} "
            f"were not in the archive. Check `sword: version:`."
        )
    return reaches_path, nodes_path


# ─────────────────────────────────────────────────────────────────────
# Tracing
# ─────────────────────────────────────────────────────────────────────
def trace_reaches(
    us_reach_id: int,
    ds_reach_id: int,
    reaches: gpd.GeoDataFrame,
    max_iter: int = 10_000,
) -> list[int]:
    """Walk SWORD topology from ``us_reach_id`` down to ``ds_reach_id``.

    At each reach the first non-zero entry of ``rch_id_dn`` is followed
    (a single-channel walk). ``rch_id_dn`` is a space-separated string
    when a reach has several downstream neighbours.

    Args:
        us_reach_id: Upstream end of the river (inclusive).
        ds_reach_id: Downstream end of the river (inclusive).
        reaches: SWORD reaches with ``reach_id``, ``rch_id_dn``, ``dist_out``.
        max_iter: Safety limit on the number of steps.

    Returns:
        Reach IDs ordered upstream → downstream.

    Raises:
        RuntimeError: If an endpoint is missing, the endpoints are swapped,
            or the walk leaves the network / never reaches ``ds_reach_id``.
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
                f"{label} {rid} is not a river reach in this SWORD continent. "
                f"Check sword_cont_abb, and that it is not a lake/dam reach."
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
            return path
        dn_raw = dn_lookup.get(current)
        if dn_raw is None:
            raise RuntimeError(
                f"Walk left the river network at reach {current} (after "
                f"{len(path) - 1} steps): its downstream neighbour is not a "
                f"kept reach type. Possible lake/dam reach or topology break."
            )
        dn_ids = [int(x) for x in str(dn_raw).split() if int(x) != 0]
        if not dn_ids:
            raise RuntimeError(
                f"Reach {current} has no downstream neighbour "
                f"(rch_id_dn={dn_raw!r}); the walk reached an outlet without "
                f"passing {ds_reach_id}. Check the endpoint pair."
            )
        current = dn_ids[0]
        path.append(current)

    raise RuntimeError(
        f"Did not reach {ds_reach_id} within {max_iter} steps. "
        f"Possible topology loop or wrong endpoint pair."
    )


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


def load_continent_reaches(
    reaches_path: Path, river_types: list[int]
) -> gpd.GeoDataFrame:
    """Read a continental SWORD reaches gpkg, keeping only ``river_types``."""
    types = ",".join(str(int(t)) for t in river_types)
    return pyogrio.read_dataframe(reaches_path, where=f"type IN ({types})")


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
        path = trace_reaches(river.us_reach_id, river.ds_reach_id, reaches)
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
            reaches_path, nodes_path = ensure_sword_downloaded(
                sword_dir, continent, settings
            )
            log.info("Loading %s", reaches_path.name)
            reaches = load_continent_reaches(reaches_path, settings.river_types)
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
