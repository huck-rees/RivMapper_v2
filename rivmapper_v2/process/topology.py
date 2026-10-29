"""Channel network topology for one window-year.

The mask's skeleton is split into **links** (skeleton runs between
junctions or free ends) and **nodes** (junction clusters, free ends), and
held as a networkx multigraph. Each link carries features measured along
its skeleton, in units of the channel width W, so rules work across rivers:

=================  ========================================================
row, col           centroid of the link's skeleton (window pixel coordinates)
length_w           skeleton length / W
width_w            median local width (2 × distance to the mask edge) / W
width_cv           variation of that width along the link
core_frac          share of the link's skeleton on the low-flow core
p_channel          mean P(channel) along the skeleton
prev_frac          share of the skeleton on last year's channel mask
line_dist_w        median distance to the SWORD reach line / W
terminal           the link touches a terminal zone (river enters/leaves)
=================  ========================================================

``select_links`` then decides which links are channel:

1. **Main path**: the least-cost route between the terminal zones, cost =
   length × a penalty that is low for channel-like links (on last year's
   channel, on the core, near the reach line, near W wide). Always kept.
2. **Alternative threads**: links on a cycle with the kept network (water
   can flow through them) are kept only if they look like channel:
   width within ``thread_width_w`` of W, mostly core, and not much wider in
   places than elsewhere (floodplain lakes and flooded forest are wide and
   ragged). In braided reaches the width floor is lower.
3. Everything else (dead-end subtrees, floodplain lattices attached at one
   or two points, isolated water) is dropped.

``mask_from_links`` maps the decision back to pixels: each mask pixel goes
to its nearest skeleton pixel, and is kept if that link is kept and the
pixel lies within the link's own channel half-width (+ a margin), so a
lake fused to the bank does not ride along with the channel it touches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

EIGHT = np.ones((3, 3), bool)
TINY_LINK_W = 1.0  # links this short may join the main path to itself
SHORT_LINK_W = 3.0
MAX_REF_W = 2.0
LATTICE_W = 10.0  # side-channel networks: at most one internal loop per this length (W)  # main-channel width reference is capped at this (W)  # links shorter than this (W) are not judged on their own
MINOR_LOOP_W = 1.5  # loops shorter than this (W) between kept links are kept
KEPT_DECISIONS = {"main", "thread", "terminal_zone", "spur", "minor_loop"}


def _neighbours(mask: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    return ndi.convolve(mask.astype(np.uint8), k, mode="constant")


@dataclass
class LinkGraph:
    skel: np.ndarray  # bool skeleton
    link_id: np.ndarray  # int32, skeleton pixel -> link id (0 = junction/none)
    node_id: np.ndarray  # int32, junction/end pixel -> node id (0 = none)
    graph: nx.MultiGraph  # nodes: node ids; edges keyed by link id
    links: pd.DataFrame  # one row per link (index = link id)
    term_nodes: dict[int, set[int]]  # terminal zone -> node ids inside it


def build(
    mask: np.ndarray,
    term: np.ndarray,
    width_px: float,
    *,
    p_channel: np.ndarray | None = None,
    core: np.ndarray | None = None,
    prev_mask: np.ndarray | None = None,
    centerline: np.ndarray | None = None,
) -> LinkGraph:
    """Link–node graph of ``mask`` with per-link features."""
    from skimage.morphology import skeletonize

    W = max(float(width_px), 1.0)
    skel = skeletonize(mask) & mask
    nb = _neighbours(skel)
    junction = skel & (nb >= 3)
    end = skel & (nb <= 1)
    nodes_px = junction | end
    node_id, n_nodes = ndi.label(nodes_px, structure=EIGHT)
    # links: skeleton runs between nodes. Ends belong to both a node and a
    # link (so a spur has length); junction pixels belong only to nodes.
    link_px = skel & ~junction
    link_id, n_links = ndi.label(link_px, structure=EIGHT)

    # which nodes each link touches (every 8-neighbour offset, so a link
    # pixel next to two junctions records both)
    padded = np.pad(node_id, 1)
    H, Wd = node_id.shape
    rows, cols = np.nonzero(link_px)
    lids = link_id[rows, cols]
    found = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nd = padded[rows + 1 + dr, cols + 1 + dc]
            found.append(np.stack([lids, nd]))
    pairs = np.unique(np.concatenate(found, axis=1), axis=1)
    pairs = pairs[:, (pairs[0] > 0) & (pairs[1] > 0)]
    # an end pixel is itself a node: count it
    own = np.unique(np.stack([link_id[end & link_px], node_id[end & link_px]]), axis=1)
    pairs = np.unique(np.concatenate([pairs, own], axis=1), axis=1)

    # per-link features
    idx = np.arange(1, n_links + 1)
    ones = np.ones_like(link_id, dtype=float)
    length = ndi.sum_labels(ones, link_id, idx)
    dt = ndi.distance_transform_edt(mask)
    width = 2 * dt  # local width at skeleton pixels (≈ diameter)
    lab_flat = link_id[link_px]
    w_flat = width[link_px]
    order = np.argsort(lab_flat, kind="stable")
    lab_s, w_s = lab_flat[order], w_flat[order]
    bounds = np.searchsorted(lab_s, np.arange(1, n_links + 2))
    med_w = np.empty(n_links)
    cv_w = np.empty(n_links)
    p90_w = np.empty(n_links)
    for i in range(n_links):
        w = w_s[bounds[i] : bounds[i + 1]]
        med_w[i] = np.median(w)
        p90_w[i] = np.percentile(w, 90)
        cv_w[i] = w.std() / max(w.mean(), 1e-6)

    def mean_on(arr):
        if arr is None:
            return np.full(n_links, np.nan)
        return ndi.mean(arr.astype(float), link_id, idx)

    if centerline is not None and centerline.any():
        d_line = ndi.distance_transform_edt(~centerline.astype(bool))
        line_d = ndi.median(d_line, link_id, idx) / W
    else:
        line_d = np.full(n_links, np.nan)

    # terminal zones: widened to reach the skeleton (it stops short of ends)
    reach = math.ceil(0.5 * W) + 1
    term_nodes: dict[int, set[int]] = {}
    term_link = np.zeros(n_links + 1, bool)
    in_term = np.zeros(n_links)
    if (term > 0).any():
        tz = ndi.grey_dilation(term, footprint=EIGHT, size=None, mode="constant")
        for _ in range(reach - 1):
            tz = np.where(tz > 0, tz, ndi.grey_dilation(tz, footprint=EIGHT))
        on = skel & (tz > 0)
        in_term = ndi.mean(on.astype(float), link_id, idx)
        for z in np.unique(tz[on]):
            sel = on & (tz == z)
            ns = set(np.unique(node_id[sel]).tolist()) - {0}
            ls = set(np.unique(link_id[sel]).tolist()) - {0}
            term_link[list(ls)] = True
            term_nodes[int(z)] = ns | {-int(z)}  # -z: virtual node for links
            for li in ls:
                pairs = np.concatenate([pairs, [[li], [-int(z)]]], axis=1)

    g = nx.MultiGraph()
    g.add_nodes_from(range(1, n_nodes + 1))
    by_link: dict[int, list[int]] = {}
    for li, nd in pairs.T:
        by_link.setdefault(int(li), []).append(int(nd))
    for li in range(1, n_links + 1):
        ends = by_link.get(li, [])
        real = [n for n in ends if n > 0]
        virt = [n for n in ends if n < 0]
        # a link has two ends; a closed loop with one junction touches it once
        if len(real) >= 2:
            a, b = real[0], real[1]
        elif len(real) == 1:
            a = b = real[0]
        else:  # isolated loop: give it its own node
            a = b = -(10_000_000 + li)
        g.add_edge(a, b, key=li)
        for v in virt:  # tie terminal links to the zone's virtual node, from
            # both ends; a path through this connector counts as using link li
            g.add_edge(a, v, key=-li)
            if b != a:
                g.add_edge(b, v, key=-li)

    if n_links:
        com = np.array(ndi.center_of_mass(ones, link_id, idx)).reshape(-1, 2)
    else:
        com = np.zeros((0, 2))
    links = pd.DataFrame(
        {
            "row": com[:, 0],
            "col": com[:, 1],
            "length_w": length / W,
            "width_w": med_w / W,
            "width_p90_w": p90_w / W,
            "width_cv": cv_w,
            "core_frac": mean_on(core),
            "p_channel": mean_on(p_channel),
            "prev_frac": mean_on(prev_mask),
            "line_dist_w": line_d,
            "terminal": term_link[1:],
            "in_terminal_frac": in_term,
        },
        index=pd.Index(idx, name="link"),
    )
    return LinkGraph(skel, link_id, node_id, g, links, term_nodes)


def link_cost(links: pd.DataFrame) -> pd.Series:
    """Per-link cost multiplier for the main-path search (≥ 1; 1 = ideal)."""
    pen = pd.Series(1.0, index=links.index)
    prev = links.prev_frac.fillna(0)
    core = links.core_frac.fillna(0)
    pen *= 1 + 4 * (1 - prev)
    pen *= 1 + 2 * (1 - core)
    pen *= 1 + np.clip(links.line_dist_w.fillna(0) - 1, 0, 10) / 2
    pen *= 1 + np.abs(np.log(np.clip(links.width_w, 0.1, 10)))
    return pen


def main_path(lg: LinkGraph) -> set[int]:
    """Links on the least-cost route between every pair of terminal zones."""
    links = lg.links
    if not lg.term_nodes or len(lg.term_nodes) < 2:
        return set()
    cost = link_cost(links) * np.maximum(links.length_w, 0.1)
    h = nx.Graph()
    best: dict[tuple, tuple[float, int]] = {}
    for a, b, k in lg.graph.edges(keys=True):
        c = 0.5 * float(cost.loc[-k]) if k < 0 else float(cost.loc[k])
        key = (min(a, b), max(a, b))
        if a == b:
            continue
        if key not in best or c < best[key][0]:
            best[key] = (c, k)
    for (a, b), (c, k) in best.items():
        h.add_edge(a, b, weight=c, link=k)
    zones = sorted(lg.term_nodes)
    kept: set[int] = set()
    for i, za in enumerate(zones):
        for zb in zones[i + 1 :]:
            sa, sb = -za, -zb
            if sa not in h or sb not in h:
                continue
            try:
                path = nx.shortest_path(h, sa, sb, weight="weight")
            except nx.NetworkXNoPath:
                continue
            for u, v in zip(path[:-1], path[1:], strict=False):
                kept.add(abs(int(h[u][v]["link"])))
    return kept


def through_links(lg: LinkGraph, kept: set[int]) -> set[int]:
    """Links that carry flow: on a cycle or between terminals, given the
    kept main path (i.e. not in a dead-end subtree)."""
    g = nx.MultiGraph()
    for a, b, k in lg.graph.edges(keys=True):
        g.add_edge(a, b, key=k)
    # repeatedly strip free ends (degree-1 nodes that are not terminal)
    virt = {-z for z in lg.term_nodes}
    while True:
        leaves = [n for n, d in g.degree() if d <= 1 and n not in virt]
        if not leaves:
            break
        g.remove_nodes_from(leaves)
    return {int(k) for _, _, k in g.edges(keys=True) if k > 0}


@dataclass
class ThreadRule:
    width_min_w: float = 0.3  # alternative threads at least this × W wide
    width_max_w: float = 1.6  # ... and at most this (median width)
    width_p90_max_w: float = 2.5  # no part much wider (lakes, flooded forest)
    min_core_frac: float = 0.5
    max_width_cv: float = 0.6
    persistence: bool = True  # last year's channel links pass looser limits


def _coarse_cycles(g: nx.MultiGraph, length_w: pd.Series) -> int:
    """Independent loops of ``g`` that are bigger than junction noise.

    Links shorter than TINY_LINK_W are contracted into their nodes, and
    loops of total length below 2 × MINOR_LOOP_W left on one node after
    that (holes, bars, junction clutter) are not counted.
    """
    parent: dict[int, int] = {n: n for n in g.nodes}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for a, b, k in g.edges(keys=True):
        if length_w[k] < TINY_LINK_W:
            parent[find(a)] = find(b)
    h = nx.MultiGraph()
    h.add_nodes_from({find(n) for n in g.nodes})
    for a, b, k in g.edges(keys=True):
        if length_w[k] < TINY_LINK_W:
            continue
        ra, rb = find(a), find(b)
        if ra == rb and length_w[k] < 2 * MINOR_LOOP_W:
            continue
        h.add_edge(ra, rb, key=k)
    return h.number_of_edges() - h.number_of_nodes() + nx.number_connected_components(h)


def thread_test(
    lg: LinkGraph,
    links: pd.DataFrame,
    main: set[int],
    rule: ThreadRule,
    braided: bool = False,
) -> pd.Series:
    """Which links look like channel (before the loop test).

    * Widths are judged against the main path's own width in this window
      (length-weighted median, between 1 and 2 W), so a river running
      wider than its width model is not penalised.
    * **Persistence**: a link mostly on last year's channel (≥ 50%) passes
      looser limits on the silting-up side only (min width × 0.5, core
      share 0.4 × the usual), so a narrowing, shallowing loop does not
      flicker in and out; maximum widths and width variation are not
      relaxed, so ragged floodplain water let in one wet year does not
      stick.
    * **Short links** are not judged on their own: links < 1 W
      (junctions), and links < 3 W on last year's channel (the narrowing
      ends of an existing loop). A connected group of them passes when it
      joins two passing links, at least one of them a second thread (a
      group of links < 1 W may also join the main path to itself), and is
      short itself. Short links off last year's channel are judged like any
      other, so scattered floodplain water cannot be stitched into loops.
    """
    if main:
        m = links.loc[sorted(main)]
        order = np.argsort(m.width_w.to_numpy())
        cum = np.cumsum(m.length_w.to_numpy()[order])
        ref = float(m.width_w.to_numpy()[order][np.searchsorted(cum, cum[-1] / 2)])
        ref = float(np.clip(ref, 1.0, MAX_REF_W))
    else:
        ref = 1.0
    wmin = rule.width_min_w * (0.5 if braided else 1.0)
    core = links.core_frac.fillna(1)

    def passes(f_min, f_max, f_core, f_cv):
        return (
            links.width_w.between(wmin * f_min, rule.width_max_w * ref * f_max)
            & (links.width_p90_w <= rule.width_p90_max_w * ref * f_max)
            & (core >= rule.min_core_frac * f_core)
            & (links.width_cv <= rule.max_width_cv * f_cv)
        )

    thread = passes(1, 1, 1, 1)
    was = links.prev_frac.fillna(0) >= 0.5
    if rule.persistence:
        thread |= was & passes(0.5, 1.0, 0.4, 1.0)

    # links < 1 W always; links < 3 W only where last year's channel ran
    # (otherwise the rule completes loops out of scattered floodplain water)
    p90_cap = rule.width_p90_max_w * ref
    eligible = links.length_w < TINY_LINK_W
    if rule.persistence:
        eligible |= (links.length_w < SHORT_LINK_W) & was
    short = eligible & ~thread & (links.width_p90_w <= p90_cap)
    if short.any():
        thr_ids = set(links.index[thread.to_numpy()].tolist()) - main
        main_nodes: dict[int, int] = {}
        thr_nodes: dict[int, int] = {}
        sg = nx.MultiGraph()
        for a, b, k in lg.graph.edges(keys=True):
            if k <= 0:
                continue
            if k in main:
                for n in {a, b}:
                    main_nodes[n] = main_nodes.get(n, 0) + 1
            elif k in thr_ids:
                for n in {a, b}:
                    thr_nodes[n] = thr_nodes.get(n, 0) + 1
            elif short.get(k, False):
                sg.add_edge(a, b, key=k)
        for comp in nx.connected_components(sg):
            ks = [k for _, _, k in sg.subgraph(comp).edges(keys=True)]
            n_thr = sum(thr_nodes.get(n, 0) for n in comp)
            n_main = sum(main_nodes.get(n, 0) for n in comp)
            all_tiny = bool((links.length_w[ks] < TINY_LINK_W).all())
            joins = n_thr + n_main >= 2 and (n_thr >= 1 or all_tiny)
            limit = 2 * (TINY_LINK_W if all_tiny else SHORT_LINK_W)
            if joins and links.length_w[ks].sum() <= limit:
                thread[ks] = True
    return thread


def select_links(
    lg: LinkGraph,
    rule: ThreadRule | None = None,
    braided: bool = False,
    spur_max_w: float = 3.0,
) -> tuple[set[int], pd.DataFrame]:
    """Kept link ids, and the link table with a ``decision`` column.

    Decisions: ``main``, ``thread``, ``terminal_zone`` (kept); ``spur`` (short dead end off the
    kept network, ≤ ``spur_max_w``: bank irregularities, bar heads; kept);
    ``floodplain_loop``, ``floodplain_lattice``, ``thread_not_in_loop``,
    ``dead_end_or_isolated`` (dropped).
    """
    rule = rule or ThreadRule()
    links = lg.links.copy()
    main = main_path(lg)
    through = through_links(lg, main)
    thread = thread_test(lg, links, main, rule, braided)
    dec = pd.Series("dead_end_or_isolated", index=links.index)
    th = links.index.isin(list(through))
    dec[th] = "floodplain_loop"
    dec[th & thread.to_numpy()] = "thread"
    dec[links.index.isin(list(main))] = "main"
    # Inside a terminal zone (where the river enters or leaves the window,
    # up to ~10 W across at the river's ends) the main path stops at the
    # zone's edge; channel-width links there are kept.
    in_zone = (
        (links.in_terminal_frac >= 0.5)
        & (links.width_p90_w <= rule.width_p90_max_w)
        & dec.isin(["floodplain_loop", "thread", "dead_end_or_isolated"])
    )
    dec[in_zone] = "terminal_zone"
    # A thread must belong to the kept network as a loop: keep the threads
    # connected to the main path through kept links, then strip free ends
    # again (a thread hanging off a dropped floodplain loop is a dead end).
    cand = set(links.index[dec.isin(["main", "thread", "terminal_zone"])].tolist())
    if main:
        g = nx.MultiGraph()
        for a, b, k in lg.graph.edges(keys=True):
            if abs(k) in cand:
                g.add_edge(a, b, key=k)
        virt = {-z for z in lg.term_nodes}
        while True:
            leaves = [n for n, d in g.degree() if d <= 1 and n not in virt]
            if not leaves:
                break
            g.remove_nodes_from(leaves)
        main_nodes = {n for a, b, k in lg.graph.edges(keys=True) if k in main
                      for n in (a, b)}
        ok: set[int] = set()
        for comp in nx.connected_components(g):
            if comp & main_nodes:
                ok |= {
                    abs(k) for a, b, k in g.subgraph(comp).edges(keys=True) if k > 0
                }
        # Lattice test: a side-channel network is (nearly) a simple path
        # between points on the main channel; floodplain water is a tangle
        # of small loops. Each connected group of second-thread links may
        # hold at most one internal loop per LATTICE_W widths of its length.
        tg = nx.MultiGraph()
        for a, b, k in lg.graph.edges(keys=True):
            if k > 0 and k in ok and dec[k] == "thread":
                tg.add_edge(a, b, key=k)
        for comp in nx.connected_components(tg):
            sub = tg.subgraph(comp)
            ks = [k for _, _, k in sub.edges(keys=True)]
            cycles = _coarse_cycles(sub, links.length_w)
            if cycles > links.length_w[ks].sum() / LATTICE_W:
                ok -= set(ks)
                dec[ks] = "floodplain_lattice"
        orphan = dec.eq("thread") & ~links.index.isin(list(ok))
        dec[orphan] = "thread_not_in_loop"
        # short spurs hanging directly off the kept network stay
        kept_ids = set(links.index[dec.isin(["main", "thread", "terminal_zone"])].tolist())
        kept_nodes = {
            n for a, b, k in lg.graph.edges(keys=True) if k in kept_ids for n in (a, b)
        }
        for a, b, k in lg.graph.edges(keys=True):
            if k <= 0:
                continue
            if (
                dec[k] == "dead_end_or_isolated"
                and links.length_w[k] <= spur_max_w
                and (a in kept_nodes or b in kept_nodes)
            ):
                dec[k] = "spur"
            # a short loop between two kept nodes: a hole or bar inside the
            # channel, not floodplain water
            elif (
                dec[k] in ("floodplain_loop", "thread_not_in_loop")
                and links.length_w[k] <= MINOR_LOOP_W
                and a in kept_nodes
                and b in kept_nodes
            ):
                dec[k] = "minor_loop"
    links["decision"] = dec
    kept = set(links.index[dec.isin(sorted(KEPT_DECISIONS))].tolist())
    return kept, links


def nearest_link(lg: LinkGraph) -> np.ndarray:
    """Per pixel, the id of the link owning the nearest skeleton pixel."""
    lid = lg.link_id.copy()
    if not lg.skel.any():
        return lid
    jun = lg.skel & (lid == 0)
    if jun.any() and (lid > 0).any():
        _, (ri, ci) = ndi.distance_transform_edt(lid == 0, return_indices=True)
        lid[jun] = lid[ri[jun], ci[jun]]
    _, (ri, ci) = ndi.distance_transform_edt(~lg.skel, return_indices=True)
    return lid[ri, ci]


def mask_from_links(
    mask: np.ndarray, lg: LinkGraph, kept: set[int], width_px: float,
    margin_w: float = 0.5,
) -> np.ndarray:
    """Mask pixels within reach of a kept link.

    A pixel is kept when its nearest *kept* skeleton pixel is no farther
    than the local half-width there (at least 0.5 W, at most the link's
    90th-percentile half-width) plus ``margin_w`` × W. So
    a lake fused to the bank is cut off at the channel's edge, and the
    pixels of a dropped branch that lie inside the kept channel (e.g. at a
    tributary mouth) stay.
    """
    W = max(float(width_px), 1.0)
    if not lg.skel.any() or not kept:
        return np.zeros_like(mask) if kept is not None and not kept else mask.copy()
    keep_l = np.zeros(int(lg.link_id.max()) + 1, bool)
    keep_l[np.array(sorted(kept))] = True
    on = lg.skel & keep_l[lg.link_id]
    # junction pixels next to a kept link count as kept skeleton
    jun = lg.skel & (lg.link_id == 0)
    if jun.any():
        on |= jun & ndi.binary_dilation(on, EIGHT)
    if not on.any():
        return np.zeros_like(mask)
    lid = np.where(on, lg.link_id, 0)
    if (on & (lid == 0)).any():  # junctions inherit a neighbouring kept link
        _, (ri, ci) = ndi.distance_transform_edt(lid == 0, return_indices=True)
        fix = on & (lid == 0)
        lid[fix] = lid[ri[fix], ci[fix]]
    d, (ri, ci) = ndi.distance_transform_edt(~on, return_indices=True)
    owner = lid[ri, ci]
    # The limit follows the channel's local half-width at the nearest kept
    # skeleton pixel, between 0.5 W and the link's 90th-percentile half-width
    # (so a lake bulging from the bank is still cut at the channel's edge).
    local = ndi.distance_transform_edt(mask)[ri, ci]
    cap = np.full(int(lg.link_id.max()) + 1, 0.5 * W)
    cap[lg.links.index.to_numpy()] = np.maximum(
        0.5 * lg.links.width_p90_w.to_numpy() * W, 0.5 * W
    )
    lim = np.clip(local, 0.5 * W, cap[owner]) + margin_w * W
    return mask & (d <= lim)
