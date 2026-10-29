"""Channel extraction for one window and one year.

Turns per-pixel probabilities from the HMM (``hmm.py``) into a binary
near-bankfull channel mask, keeping the river's connected channel and
removing everything else. Every length is a multiple of the channel width
W (pixels) of the window's centre reach, so the same settings work across
rivers. Steps (numbers match the implementation plan):

3. ``core`` = P(channel) ≥ core threshold; ``envelope`` = P(channel) +
   P(margin) ≥ envelope threshold.
4. Seeds: core pixels within 1 W of the SWORD reach lines, plus core
   pixels on last year's channel skeleton (so the mask follows a channel
   that has migrated away from SWORD's fixed line).
5. Allowed bridge pixels: some water evidence (P_env ≥ 0.2) or unobserved
   this year and last; never pixels observed dry at least twice.
6. Close: binary closing of the core (radius 0.5 W), only on allowed pixels.
7. Reconnect: starting from the seeded piece that touches most terminals,
   other pieces are joined by least-cost paths through allowed pixels, up
   to 5 W long and no longer than half the piece itself (repeated, so
   pieces can join in a chain).
8. Keep the pieces that carry the river through the window (touch two
   terminals: cut lines or river ends); everything else is dropped. If no
   piece does (gaps), all seeded pieces are kept. Grow into the envelope
   within 1 W.
9. Remove blobs wider than 2 × 1.25 W that are not on the centreline.
10. Prune dead-end branches longer than 3 W (cutoffs still joined at one
    end, tributary mouths) from the skeleton.
11. Drop pieces smaller than 4 W²; fill dry holes smaller than 0.25 W².
12. Core product = final mask ∩ core.

The corridor limits from the channel belt / HAND (step 2) and the wider
dead-end limit for braided reaches come in phase 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from ..config import ProcessSettings

EIGHT = np.ones((3, 3), bool)

# Decision codes: why each pixel is (or is not) channel. Written per
# reach-year to masks/diagnostics/ (see docs/methods/processing.md).
DECISIONS = {
    0: "not water: below both probability thresholds",
    1: "channel: low-flow core",
    2: "channel: grown into the high-stage envelope",
    3: "channel: gap closed or bridged",
    4: "channel: small dry hole filled",
    10: "dropped: water not connected to the river's through-flow network",
    11: "dropped: over-wide blob off the reach line",
    12: "dropped: dead-end branch",
    13: "dropped: piece too small",
    14: "dropped: by the stricter rerun of a high-area year",
    15: "dropped: floodplain water attached to the channel (off the main path "
    "and not a channel-like loop)",
}


@dataclass
class ChannelInputs:
    """Everything channel extraction needs for one window-year (window grid).

    Attributes:
        p_channel, p_margin: HMM posteriors, float in [0, 1].
        n_obs, n_wet_3: This year's raw counts (0 outside the polygons).
        n_obs_prev: Last year's clear looks (zeros for the first year).
        inside: Pixels inside the window's reach polygons.
        centerline: SWORD reach lines rasterised (no dilation).
        prev_mask: Last year's final mask, or None.
        width_px: Channel width W of the centre reach, in pixels.
        terminals: Labelled zones where the river enters/leaves the window
            (``RiverLayout.terminals``); None = derive from the polygon edge.
        dead_end_px: Dead-end branches longer than this are removed
            (default ``dead_end_widths`` × W, or ``dead_end_widths_braided``
            in braided reaches).
        braided: The centre reach is braided (most SWORD nodes in it have
            n_chan_mod > 1): lower width floor for second threads, longer
            dead-end limit.
    """

    p_channel: np.ndarray
    p_margin: np.ndarray
    n_obs: np.ndarray
    n_wet_3: np.ndarray
    n_obs_prev: np.ndarray
    inside: np.ndarray
    centerline: np.ndarray
    prev_mask: np.ndarray | None
    width_px: float
    terminals: np.ndarray | None = None
    dead_end_px: float | None = None
    braided: bool = False


@dataclass
class ChannelResult:
    channel: np.ndarray  # near-bankfull mask
    core: np.ndarray  # low-flow core mask
    stats: dict = field(default_factory=dict)
    decision: np.ndarray | None = None  # uint8 codes, see DECISIONS
    links: object = None  # pandas DataFrame from topology.select_links, or None


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= r * r


def _keep_touching(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Components of ``mask`` (8-connected) that touch ``seeds``."""
    lab, n = ndi.label(mask, structure=EIGHT)
    if n == 0:
        return mask.copy()
    hit = np.zeros(n + 1, bool)
    hit[np.unique(lab[seeds & mask])] = True
    hit[0] = False
    return hit[lab]


def _drop_small(mask: np.ndarray, min_px: float) -> np.ndarray:
    lab, n = ndi.label(mask, structure=EIGHT)
    if n == 0:
        return mask
    size = np.bincount(lab.ravel())
    keep = size >= min_px
    keep[0] = False
    return keep[lab]


def _fill_small_holes(mask: np.ndarray, max_px: float) -> np.ndarray:
    lab, n = ndi.label(~mask)  # 4-connected dry regions
    if n == 0:
        return mask
    size = np.bincount(lab.ravel())
    # regions touching the array edge are not holes
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    fill = size <= max_px
    fill[0] = False
    fill[edge] = False
    return mask | fill[lab]


def _reconnect(
    core: np.ndarray,
    seeded: np.ndarray,
    allowed: np.ndarray,
    p_env: np.ndarray,
    max_len_px: float,
    min_piece_px: float = 0,
    nrbc: bool = True,
) -> tuple[np.ndarray, int]:
    """Join core pieces to the network (``seeded``) by least-cost paths.

    Pieces smaller than ``min_piece_px`` are not joined (noise; they would
    only add false water).

    Returns (bridge pixels, number of pieces joined).
    """
    from skimage.graph import MCP_Geometric

    orphans = (
        _drop_small(core & ~seeded, min_piece_px) if min_piece_px else core & ~seeded
    )
    if not orphans.any() or not seeded.any():
        return np.zeros_like(core), 0
    # only search within max_len_px of the network (bounds the work)
    reach = ndi.distance_transform_edt(~seeded) <= max_len_px + 1
    cost = np.where((allowed | core) & reach, 1.0 - 0.9 * np.clip(p_env, 0, 1), np.inf)
    mcp = MCP_Geometric(cost, fully_connected=True)
    starts = np.argwhere(seeded)
    cum, _ = mcp.find_costs(starts)
    lab, n = ndi.label(orphans, structure=EIGHT)
    bridges = np.zeros_like(core)
    joined = 0
    reachable = np.isfinite(ndi.minimum(cum, lab, index=np.arange(1, n + 1)))
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if not reachable[i - 1]:
            continue
        piece = lab[sl] == i
        c = np.where(piece, cum[sl], np.inf)
        k = np.unravel_index(np.argmin(c), c.shape)
        if not np.isfinite(c[k]):
            continue
        end = (k[0] + sl[0].start, k[1] + sl[1].start)
        path = np.array(mcp.traceback(end))
        length = np.hypot(*np.diff(path, axis=0).T).sum() if len(path) > 1 else 0.0
        # NRBC rule (Zeng et al. 2015): a piece must be at least twice as
        # long as the gap it bridges; piece length = its bounding diagonal.
        piece_len = math.hypot(*piece.shape)
        if length <= max_len_px and (not nrbc or length <= 0.5 * piece_len):
            bridges[path[:, 0], path[:, 1]] = True
            joined += 1
    return bridges, joined


def terminals(inside: np.ndarray, centerline: np.ndarray, W: float) -> np.ndarray:
    """Labelled zones where the river enters or leaves the window.

    Where the reach line crosses the edge of the window's polygons: the
    cut lines to the reaches outside the window, or the river's ends.
    Returns an int array, 0 = not a terminal, 1..n = terminal zones.
    """
    edge = inside & ndi.binary_dilation(~inside, EIGHT, iterations=2)
    near = ndi.distance_transform_edt(~centerline) <= max(W, 2.0)
    zone = edge & near
    lab, _ = ndi.label(ndi.binary_dilation(zone, EIGHT), structure=EIGHT)
    return np.where(zone, lab, 0)


def _main_network(
    net: np.ndarray, inp: ChannelInputs, W: float
) -> tuple[np.ndarray, bool]:
    """Keep the components of ``net`` that carry the river through the window.

    A component is kept when it touches two or more terminal zones (water
    enters and leaves through it). Disconnected water inside the window —
    oxbows, ponds, a channel abandoned by avulsion — touches at most one
    and is dropped. If no component reaches two terminals (gaps from
    clouds or a channel too narrow to see everywhere), every component is
    kept and ``through`` is False.
    """
    term = (
        inp.terminals
        if inp.terminals is not None
        else terminals(inp.inside, inp.centerline, W)
    )
    lab, n = ndi.label(net, structure=EIGHT)
    if n == 0:
        return net, False
    pairs = np.unique(np.stack([lab[term > 0], term[term > 0]]), axis=1)
    pairs = pairs[:, pairs[0] > 0]
    hits = np.bincount(pairs[0], minlength=n + 1)
    keep = hits >= 2
    through = bool(keep[1:].any())
    if not through:
        keep = np.ones(n + 1, bool)
    keep[0] = False
    return keep[lab], through


def _start_piece(mask: np.ndarray, seeds: np.ndarray, term: np.ndarray) -> np.ndarray:
    """The seeded piece the network grows from: the one touching the most
    terminal zones, then the largest."""
    lab, n = ndi.label(mask, structure=EIGHT)
    if n == 0:
        return mask
    seeded = np.zeros(n + 1, bool)
    seeded[np.unique(lab[seeds & mask])] = True
    seeded[0] = False
    if not seeded.any():
        return np.zeros_like(mask)
    hits = np.zeros(n + 1)
    if (term > 0).any():
        pairs = np.unique(np.stack([lab[term > 0], term[term > 0]]), axis=1)
        pairs = pairs[:, pairs[0] > 0]
        hits = np.bincount(pairs[0], minlength=n + 1).astype(float)
    size = np.bincount(lab.ravel(), minlength=n + 1)
    score = np.where(seeded, hits * size.max() * 10 + size, -1)
    score[0] = -1
    return lab == int(np.argmax(score))


def _touches_two_terminals(mask: np.ndarray, term: np.ndarray) -> bool:
    """True if one connected piece of ``mask`` touches two terminal zones."""
    lab, n = ndi.label(mask, structure=EIGHT)
    if n == 0 or not (term > 0).any():
        return False
    pairs = np.unique(np.stack([lab[term > 0], term[term > 0]]), axis=1)
    pairs = pairs[:, pairs[0] > 0]
    return bool((np.bincount(pairs[0], minlength=n + 1) >= 2).any())


def _neighbours(mask: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    return ndi.convolve(mask.astype(np.uint8), k, mode="constant")


def prune_dead_ends(
    mask: np.ndarray, term: np.ndarray, max_len_px: float, width_px: float = 1.0
) -> tuple[np.ndarray, int]:
    """Remove dead-end branches longer than ``max_len_px``.

    The mask's skeleton is peeled from its free ends (ends not in a
    terminal zone) until only paths between terminals and closed loops
    remain: the through-flow network. Each peeled branch is measured along
    the skeleton; mask pixels closest to a branch longer than the limit are
    removed. Shorter spurs (bank irregularities, bar heads) are kept.

    Returns (pruned mask, number of branches removed).
    """
    from skimage.morphology import skeletonize

    if not mask.any() or not (term > 0).any():
        return mask, 0
    skel = skeletonize(mask)
    # the skeleton stops short of the mask's ends; widen the zones to reach it
    W_reach = math.ceil(0.5 * width_px) + 1
    protected = skel & ndi.binary_dilation(term > 0, EIGHT, iterations=W_reach) & mask
    if not protected.any():
        return mask, 0
    core = skel.copy()
    while True:
        ends = core & (_neighbours(core) <= 1) & ~protected
        if not ends.any():
            break
        core &= ~ends
    spurs = skel & ~core
    if not spurs.any():
        return mask, 0
    lab, n = ndi.label(spurs, structure=EIGHT)
    length = ndi.sum_labels(np.ones_like(lab), lab, index=np.arange(n + 1))
    long_ = length > max_len_px
    long_[0] = False
    if not long_.any():
        return mask, 0
    # assign every mask pixel to its nearest skeleton pixel
    _, (ri, ci) = ndi.distance_transform_edt(~skel, return_indices=True)
    owner = lab[ri, ci]
    drop = mask & long_[owner] & ~core[ri, ci]
    if drop.sum() > 0.5 * mask.sum():
        # never strip most of the channel: a terminal was probably missed
        return mask, 0
    return mask & ~drop, int(long_.sum())


def _select_network(grown, inp, s, core, term, W):
    """Step 10 by link selection (topology.py).

    Returns (kept mask, dead-end pixels, floodplain pixels, stats, link
    table). Falls
    back to the input unchanged when no main path is found or when the
    selection would remove most of the mask (a terminal was probably
    missed).
    """
    from . import topology

    lg = topology.build(
        grown,
        term,
        W,
        p_channel=inp.p_channel,
        core=core,
        prev_mask=inp.prev_mask,
        centerline=inp.centerline,
    )
    rule = topology.ThreadRule(
        width_min_w=s.thread_min_widths,
        width_max_w=s.thread_max_widths,
        width_p90_max_w=s.thread_max_p90_widths,
        min_core_frac=s.thread_min_core,
        max_width_cv=s.thread_max_width_cv,
        persistence=s.thread_persistence,
    )
    if inp.dead_end_px is not None:
        max_len = inp.dead_end_px / W
    else:
        max_len = s.dead_end_widths_braided if inp.braided else s.dead_end_widths
    kept, links = topology.select_links(
        lg, rule, braided=inp.braided, spur_max_w=max_len
    )
    stats_links = links
    empty = np.zeros_like(grown)
    counts = links.decision.value_counts()
    stats = {
        "links": len(links),
        "links_main": int(counts.get("main", 0)),
        "links_thread": int(counts.get("thread", 0)),
        "links_floodplain": int(
            counts.get("floodplain_loop", 0) + counts.get("thread_not_in_loop", 0)
        ),
        "links_dead_end": int(counts.get("dead_end_or_isolated", 0)),
    }
    if not stats["links_main"]:
        return grown, empty, empty, stats, stats_links
    out = topology.mask_from_links(grown, lg, kept, W, s.link_margin_widths)
    out = _keep_touching(out, out & _main_pixels(lg, links))
    if out.sum() < 0.3 * grown.sum():
        return grown, empty, empty, stats, stats_links  # never strip most
    removed = grown & ~out
    # dead-end vs floodplain: by the link that owns each removed pixel
    dead_ids = links.index[links.decision == "dead_end_or_isolated"].to_numpy()
    dead = removed & np.isin(topology.nearest_link(lg), dead_ids)
    return out, dead, removed & ~dead, stats, stats_links


def _main_pixels(lg, links) -> np.ndarray:
    ids = links.index[links.decision == "main"].to_numpy()
    sel = np.zeros(int(lg.link_id.max()) + 1, bool)
    sel[ids] = True
    return lg.skel & sel[lg.link_id]


def extract(
    inp: ChannelInputs, s: ProcessSettings, envelope_threshold: float
) -> ChannelResult:
    """Channel and core masks for one window-year."""
    W = max(float(inp.width_px), 1.0)
    inside = inp.inside
    p_env = np.clip(inp.p_channel + inp.p_margin, 0, 1)
    core = (inp.p_channel >= s.core_threshold) & inside
    env = (p_env >= envelope_threshold) & inside
    env |= core

    # 4. seeds
    near_line = ndi.distance_transform_edt(~inp.centerline) <= max(W, 1.5)
    seeds = core & near_line
    if inp.prev_mask is not None and inp.prev_mask.any():
        from skimage.morphology import skeletonize

        seeds |= core & ndi.binary_dilation(skeletonize(inp.prev_mask), EIGHT)

    # 5. allowed bridge pixels; "observed dry" = ≥ 2 clear looks, ≤ 10% wet
    observed_dry = (inp.n_obs >= 2) & (inp.n_wet_3 * 10 <= inp.n_obs)
    allowed = (
        inside
        & ~observed_dry
        & ((p_env >= 0.2) | ((inp.n_obs == 0) & (inp.n_obs_prev == 0)))
    )

    # 6. close small gaps
    r = math.ceil(s.bridge_closing_widths * W)
    closed = core
    if r > 0:
        closed = core | (ndi.binary_closing(core, _disk(r)) & allowed)

    # 7. reconnect: grow a network from the best seeded piece, joining the
    # other pieces (seeded or not) one bridge at a time
    term = (
        inp.terminals
        if inp.terminals is not None
        else terminals(inside, inp.centerline, W)
    )
    start = _start_piece(closed, seeds, term)
    bridges = np.zeros_like(closed)
    joined = 0
    for _ in range(8):
        reached = _keep_touching(closed | bridges, start)
        b, j = _reconnect(
            closed, reached, allowed, p_env, s.bridge_max_widths * W, W * W
        )
        if j == 0:
            break
        bridges |= b
        joined += j
    n_bridge_px = int(bridges.sum())
    joined_px = closed & ~core  # pixels added by closing
    if joined and W >= 2:
        # give bridges the channel's width (a 1-px path would read as a
        # constriction), still only through allowed pixels
        bridges = ndi.binary_dilation(bridges, _disk(int(W // 2))) & (allowed | closed)
    net = _keep_touching(closed | bridges, seeds)
    net, through = _main_network(net, inp, W)

    # 8. grow from the connected core into the envelope (geodesic, ≤ grow W)
    n_grow = math.ceil(s.grow_widths * W)
    grown = net
    if n_grow > 0:
        grown = ndi.binary_dilation(net, EIGHT, iterations=n_grow, mask=env | net)

    # 9. remove over-wide blobs off the centreline
    rp = s.max_inscribed_widths * W
    dt = ndi.distance_transform_edt(grown)
    thick = dt > rp
    removed_wide = 0
    wide_mask = np.zeros_like(grown)
    if thick.any():
        blob = grown & (ndi.distance_transform_edt(~thick) <= rp + 1)
        on_line = ndi.binary_dilation(inp.centerline, EIGHT, iterations=2)
        blob_off = blob & ~_keep_touching(blob, on_line)
        candidate = _keep_touching(grown & ~blob_off, net & ~blob_off)
        if through and blob_off.any() and not _touches_two_terminals(candidate, term):
            # The removal cut the river (a wet patch lying on the channel):
            # bridge across the removed pixels (any length; only through the
            # patch itself) by the least-cost route, at
            # channel width, then keep the rest of the patch removed.
            rest = grown & ~blob_off
            start = _start_piece(rest, rest, term)
            bridge = np.zeros_like(rest)
            for _ in range(4):
                reached = _keep_touching(rest | bridge, start)
                # no NRBC length rule: the piece beyond the patch can be short
                b, j = _reconnect(rest, reached, blob_off, p_env, 50 * W, W, nrbc=False)
                if j == 0:
                    break
                bridge |= b
            if bridge.any() and W >= 2:
                bridge = ndi.binary_dilation(bridge, _disk(int(W // 2))) & blob_off
            blob_off &= ~bridge
            candidate = _keep_touching(grown & ~blob_off, net & ~blob_off)
        if through and not _touches_two_terminals(candidate, term):
            blob_off[:] = False  # still disconnected: undo the removal
            candidate = grown
        removed_wide = int(blob_off.sum())
        wide_mask = grown & ~candidate
        grown = candidate

    # 10. dead ends (cutoffs still joined at one end, tributary mouths) and
    # floodplain water attached to the channel: link selection on the
    # skeleton graph (topology.py), or phase 1 dead-end pruning
    pruned = 0
    dead_mask = np.zeros_like(grown)
    flood_mask = np.zeros_like(grown)
    link_stats: dict = {}
    link_table = None
    if through and s.network_selection:
        grown, dead_mask, flood_mask, link_stats, link_table = _select_network(
            grown, inp, s, core, term, W
        )
        pruned = link_stats.get("links_dead_end", 0)
    elif through:
        term = (
            inp.terminals
            if inp.terminals is not None
            else terminals(inside, inp.centerline, W)
        )
        max_len = (
            inp.dead_end_px
            if inp.dead_end_px is not None
            else (s.dead_end_widths_braided if inp.braided else s.dead_end_widths) * W
        )
        before = grown
        grown, pruned = prune_dead_ends(grown, term, max_len, W)
        dead_mask = before & ~grown

    # 11. size and holes
    kept = _drop_small(grown, s.min_component_w2 * W * W)
    small_mask = grown & ~kept
    final = _fill_small_holes(kept, s.fill_hole_w2 * W * W) & inside
    filled = final & ~kept
    core_out = _drop_small(final & core, s.min_component_w2 * W * W)

    stats = {
        "seed_px": int(seeds.sum()),
        "bridges_joined": joined,
        "bridge_px": n_bridge_px,
        "removed_wide_px": removed_wide,
        "through": through,
        "dead_ends_removed": pruned,
        "channel_px": int(final.sum()),
        "core_px": int(core_out.sum()),
        "not_connected_px": 0,
        "dead_end_px": int(dead_mask.sum()),
        "small_px": int(small_mask.sum()),
        "filled_px": int(filled.sum()),
        "floodplain_px": int((flood_mask & ~final).sum()),
        "braided": bool(inp.braided),
        **link_stats,
    }
    # 12. record each pixel's fate (later codes overwrite earlier ones)
    d = np.zeros(final.shape, np.uint8)
    water = (core | env) & inside
    d[water & ~final] = 10
    d[wide_mask & ~final] = 11
    d[dead_mask & ~final] = 12
    d[flood_mask & ~final] = 15
    d[small_mask & ~final] = 13
    d[final] = 2
    d[final & ((bridges | joined_px) & ~core)] = 3
    d[core_out] = 1
    d[filled & ~core] = 4
    stats["not_connected_px"] = int((d == 10).sum())
    return ChannelResult(
        channel=final, core=core_out, stats=stats, decision=d, links=link_table
    )
