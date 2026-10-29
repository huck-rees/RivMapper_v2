# Mask processing (step `process_masks`)

**Status: in development.** Phases 0 (foundations), 1 (core pipeline) and 2 (scene screening at extraction, stage flag from modelled discharge) are built; phase 2 is not yet checked on re-extracted Ok Tedi data. The step writes annual channel masks, a QA table and QA figures. Calibration of the envelope threshold and the channel-belt corridor come in phase 3. The full design and build plan are in the *RivMapper mask processing plan* (shared doc, Oct 2026).

## Goal

Turn the raw annual DSWE counts ([masks.md](masks.md)) into one co-registered binary channel mask per reach and year that

- approximates the **near-bankfull wetted channel** (channel plus its high-stage margins), defined the same way every year;
- keeps only the river's **through-flowing channel** (main stem, anabranches, flowing side channels) and removes oxbows, floodplain lakes, overbank water, ponds and tributaries;
- keeps **real change** (migration, cutoffs, avulsions) while suppressing year-to-year noise from stage, cloud and observation density;
- runs **without manual editing**: every threshold is fitted per river or scaled by the reach's channel width W.

A low-flow core mask and per-pixel probabilities are written alongside.

## Design in brief

| Stage | What | Phase |
|---|---|---|
| Windows | Each reach is processed with its upstream and downstream neighbours on the river grid; only the centre reach's result is kept | 0 ✅ |
| Scene screening | Extraction drops scenes with widespread false water, and writes a scene manifest (raw product v2; [masks.md](masks.md)) | 2 ✅ |
| Temporal model | Per-pixel hidden Markov model on the counts (dry / high-stage margin / channel) gives a channel probability every year | 1 ✅ |
| Channel extraction | Seeds (SWORD + last year's channel), width-scaled gap bridging, through-flow connectivity, removal of over-wide and dead-end water | 1 ✅ (corridor: 3) |
| Year check | Reach area against its 5-year running median; outliers rerun or flagged | 1 ✅ |
| Calibration | Envelope threshold matched to bankfull width | 3 |
| Stage flag | Modelled discharge (GloFAS reanalysis; GEOGLOWS v2 optional) on scene dates ([discharge.md](discharge.md)) | 2 ✅ |

## Reach windows ✅

`rivmapper_v2/process/windows.py`.

- The river grid is the one extraction used: origin at the top-left of the union of reach polygons, snapped to 30 m; each reach raster is the smallest window of it covering the polygon ([conventions.md](conventions.md)).
- A **window** is a reach plus one neighbour each side, on one grid. Results are kept for the centre reach only, so each pixel is decided once.
- **Cut lines** are the shared polygon boundaries with the reaches just outside a window. They run across the valley (nearest-valley partition), and are where water may enter or leave the window. The river's ends have no cut line.
- A pixel belongs to the reach whose polygon contains its centre: the same rule extraction used. On the Ok Tedi this reproduces every raw file's footprint exactly, and the 20 windows' centre reaches cover all 5,028,468 pixels once, with no gaps or overlaps.

## Synthetic rivers ✅

`rivmapper_v2/process/synthetic.py` builds rivers with a known channel every year, written in the raw-file format so processing reads them like real data.

- **Truth:** a meandering channel of width W migrating downstream; near-bankfull (`truth`) and low-flow (`truth_core`) extents.
- **Scenarios:** neck cutoff (leaves an oxbow), avulsion (abandoned course stays wet for 3 years), braided reach (a second thread switching every 6 years), floodplain lake, one overbank flood year, a tributary.
- **Observation:** per year, scenes with spatially coherent cloud and random stage; each clear pixel classified into DSWE classes with realistic confusion (mixed edge pixels, false water on land, occasional haze blocks). Clear looks per pixel follow the Ok Tedi's year-by-year medians, including the empty years (1984–86, 2012).

These are used for unit tests and the phase 1 checks below, and in phase 4 to check that processing recovers known reworking and overlap curves.

## Temporal model ✅

`rivmapper_v2/process/hmm.py`. Each pixel's yearly counts are treated as noisy evidence about a hidden state that rarely changes.

- **States:** `dry`, `margin` (wet only at higher stage) and `channel`. If the margin state ends up holding under 0.5% of pixel-years the model is refitted with two states, and the run record says so.
- **Observation:** a year's clear looks split into dry (`n_obs` − `n_wet_3`), strict wet (`n_wet_2`, DSWE 1–2) and loose-only wet (`n_wet_3` − `n_wet_2`, DSWE 3), multinomial given the state. A year with no clear looks carries no evidence, so the model fills it from the neighbouring years.
- **Change:** one yearly transition matrix; every move is allowed, including dry straight to channel (avulsions).
- **Fitting:** Baum–Welch (expectation–maximisation) per river on 50,000 sampled pixels, half of them within 2 W of the reach lines, with no labels. States are ordered by their strict-wet probability, so labels cannot swap. Posteriors for every pixel and year come from the forward–backward algorithm, in chunks of pixels, and are cached per reach in `masks/model/` (P(channel) and P(margin) × 100, one band per year each).
- Reference for the approach on annual land-cover maps: Lamarche et al. (2022), *IEEE TGRS*, doi:10.1109/TGRS.2021.3123738.

## Year status ✅

From the share of a reach's pixels with at least one clear look that year (coverage):

| Status | Rule | Mask written |
|---|---|---|
| `observed` | coverage ≥ 0.5 | yes |
| `sparse` | 0.1–0.5 | yes, flagged |
| `inferred` | < 0.1, with a year ≥ 0.1 within `max_infer_gap` (2) years on both sides | yes, from the model alone, flagged |
| `no_data` | anything else (e.g. 1984–86 on the Ok Tedi) | all 255 |

Landsat 7 SLC-off scenes are only used in `le07_gap_years` (phase 2), so 2012 on the Ok Tedi is `inferred` from 2011 and 2013.

## Channel extraction ✅

`rivmapper_v2/process/channel.py`, per window and year, years in order. Every length is a multiple of the centre reach's channel width W (`width_ref`, pixels).

1. **Core and envelope.** Core = P(channel) ≥ 0.5; envelope = P(channel) + P(margin) ≥ the envelope threshold (0.5 until calibration lands in phase 3).
2. **Seeds.** Core pixels within max(W, 1.5 px) of the SWORD reach lines, plus core pixels on last year's channel skeleton, so the mask follows a channel that has moved away from SWORD's fixed line.
3. **Allowed bridge pixels.** Inside the window, some water evidence (P_env ≥ 0.2) or unobserved this year and last, and never *observed dry* (≥ 2 clear looks, ≤ 10% of them wet).
4. **Close** the core with a disk of radius 0.5 W, on allowed pixels only.
5. **Reconnect.** Starting from the seeded piece touching the most terminals (then the largest), other pieces are joined by least-cost paths through allowed pixels (cost 1 − 0.9 P_env), each at most 5 W long and at most half the length of the piece it joins (the NRBC rule of Zeng et al. 2015). Repeated so pieces can join in a chain. Bridges are widened to the channel width.
6. **Through-flow.** *Terminals* are where water may enter or leave the window: its cut lines, and at the river's ends a disk of radius 10 W around the end of the reach line. Pieces touching two or more terminals carry the river and are kept; everything else (ponds, oxbows, abandoned courses no longer joined) is dropped. If no piece reaches two terminals — gaps from cloud, or a channel too narrow to see everywhere — all seeded pieces are kept and the QA table records `through = False`.
7. **Grow** from that network into the envelope, at most 1 W (geodesic dilation).
8. **Over-wide blobs.** Parts wider than 2 × 1.25 W not on the reach line (floodplain lakes and flooded fields joined to the channel) are removed. If that removal cuts the river from one of its terminals (a wet patch lying on the channel itself), the gap is bridged back by the least-cost route through the removed pixels, at channel width, and only the rest of the patch stays removed; if the river is still cut, nothing is removed. (Ok Tedi reach 9, 2025: a wet patch on the channel near the downstream cut was removed with the channel through it, and half the reach was then pruned as a dead end. An earlier fix that kept a corridor along the whole skeleton also kept floodplain lakes' skeletons and was replaced.)
9. **Network selection** (`process/topology.py`, phase 3; `network_selection: true`). The mask's skeleton is split into *links* (runs between junctions or free ends) joined at *nodes*, and each link is measured along its skeleton, in units of W: length, median width and its 90th percentile, width variation (CV), share on the core, share on last year's channel, distance from the reach line. Then:
   - **Main path.** The least-cost route through the link graph between each pair of terminals, where a link's cost is its length times a penalty that is 1 for an ideal link and grows when the link is off last year's channel, off the core, far from the reach line, or far from W wide. Always kept.
   - **Second threads.** Links on a loop with the kept network (water can pass through them) are kept if they look like channel: median width from 0.3 W (0.15 W in braided reaches) to 1.6 × the main channel's own width in the window (taken between 1 and 2 W), no part wider than 2.5 × that width, at least half on the core, width CV ≤ 0.6, and still part of a loop once the rejected links are gone. Widths are judged against the main channel rather than W, so a river running wider than its width model is not penalised.
     - *Persistence* (`thread_persistence: true`): a link mostly (≥ 50%) on last year's channel passes looser limits on the silting-up side only (minimum width halved, core share 0.2 instead of 0.5; maximum widths and width variation are not relaxed, so ragged floodplain water let in one wet year does not stick). A side channel or a cut-off loop stays channel while it is wet and open at both ends, rather than flickering in and out on marginal detections; a new loop must pass the normal test. Flicker would read as a full channel's worth of erosion and deposition each time.
     - *Short links* are not judged on their own: links under 1 W (junctions), and links under 3 W on last year's channel (the narrowing ends of an existing loop). A connected group of them passes when it joins two passing links, at least one a second thread, so a narrow or ragged mouth does not by itself break an existing loop. Short links off last year's channel are judged like any other, so scattered floodplain water cannot be stitched into loops.
     - *Braided reaches*: most SWORD nodes in the reach (`braided_min_chan`, 50%) have `n_chan_mod` > 1. On the Ok Tedi: reaches 6 and 7.

     This keeps anabranches, braids and cut-off loops still carrying water at both ends (a loop leaves the channel when one end closes and it becomes a dead end), and drops floodplain lakes, flooded forest and lattices of overbank water that touch the channel at two or more points, which dead-end pruning alone cannot remove.
   - **Kept besides:** short spurs (≤ 3 W) off the kept network (bar heads, bank irregularities), short loops (≤ 1.5 W) between kept links (holes and bars inside the channel), and channel-width links inside a terminal zone (at the river's ends the zone is 10 W across and the main path stops at its edge).
   - Everything else is dropped: dead-end subtrees (code 12) and floodplain water (code 15). Mask pixels are kept when they lie within reach of a kept link: the channel's local half-width at the nearest kept skeleton pixel (at least 0.5 W, at most the link's 90th-percentile half-width) plus 0.5 W. So a lake fused to the bank is cut off at the channel's edge, while pixels of a dropped branch that lie inside the kept channel (at a tributary mouth, say) stay.
   - Safeguards: if no main path is found, or the selection would remove more than 70% of the mask, nothing is removed by this step. Each kept and dropped link, with its measurements and decision, is written to `masks/diagnostics/{river}_r{NNN}_{year}_links.csv`.

   With `network_selection: false`, phase 1's dead-end pruning runs instead: the skeleton is peeled from free ends down to the through-flow network and branches longer than 3 W are removed with the pixels nearest them (never more than half the mask).
10. **Clean.** Drop pieces smaller than 4 W²; fill dry holes smaller than 0.25 W² (holes bigger than that are bars and islands, kept).
11. **Core product** = channel ∩ core.

Anabranches joined to the network at both ends are channel by design, so a newly abandoned course or cut-off loop that is still wet and connected at both ends is mapped until it stops being so (a deliberate choice: it keeps side channels from flickering, at the cost of a gradual rather than a sharp abandonment signal; the links tables record each loop's width, core share and connection by year).

## Year check ✅

Each reach's channel area in observed and sparse years is compared with its 5-year running median: score = (A − median) / (1.4826 × MAD), with the MAD floored at 5% of the median area, so a steady river only flags changes of about 15% or more. Years scoring above +3 (too much water: usually false water or a flood) are rerun once with the envelope threshold raised by 0.2 and the rerun is kept if it scores closer to the median. Years still beyond ±3 are flagged `area_high` or `area_low`; their masks are written, so analyses can choose to drop them.

## Outputs ✅

| File | Content |
|---|---|
| `masks/processed/{river}_r{NNN}_{year}.tif` | 4 bands, uint8, 255 outside the reach polygon and in `no_data` years: `channel` (0/1), `core` (0/1), `p_envelope` and `p_core` (× 100). Tags: status and flags |
| `masks/model/{river}_r{NNN}_posterior.tif` | HMM posteriors, P(channel) then P(margin) per year (× 100) |
| `masks/{river}_process_masks_qa.csv` | One row per reach-year (columns below) |
| `masks/diagnostics/{river}_r{NNN}_{year}.tif` | Decision codes: why each pixel is or is not channel (table below). uint8, 255 outside the polygon; the codes are also in the file tags |
| `masks/diagnostics/{river}_r{NNN}_{year}_links.csv` | One row per network link whose centre lies in the reach: map position (x, y), length, widths, core and last-year shares, distance from the reach line, and the decision (`main`, `thread`, `spur`, `minor_loop`, `terminal_zone` kept; `floodplain_loop`, `thread_not_in_loop`, `dead_end_or_isolated` dropped) |
| `masks/model/{river}_r{NNN}_window.tif` | Window context, on the window grid: reach owner (`ds_order`), terminal zones, rasterised reach line. uint16 |
| `masks/model/{river}_hmm_{key}.json`, `…_coverage_{key}.json` | Fitted HMM and per-reach coverage (also used to resume an interrupted run) |
| `masks/qa/{river}_r{NNN}_qa.png` | Area by year with status and outliers, and six mask thumbnails |
| `masks/qa/{river}_overview.png` | Channel area by reach and year relative to each reach's median |
| `masks/{river}_process_masks_run.yaml` | Run record: every setting, git commit, package versions, fitted HMM, summary |

Diagnostics and window context are written when `keep_diagnostics: true` (default). Switching it doesn't change the masks or force a rerun.

**Decision codes** (`masks/diagnostics/`):

| Code | Meaning |
|---|---|
| 0 | Not water: below both probability thresholds |
| 1 | Channel: low-flow core |
| 2 | Channel: grown into the high-stage envelope |
| 3 | Channel: gap closed or bridged |
| 4 | Channel: small dry hole filled |
| 10 | Dropped: water not connected to the river's through-flow network (ponds, oxbows, unseeded pieces, envelope beyond the 1 W growth limit) |
| 11 | Dropped: over-wide blob off the reach line |
| 12 | Dropped: dead-end branch |
| 13 | Dropped: piece too small |
| 14 | Dropped: by the stricter rerun of a high-area year |
| 15 | Dropped: floodplain water attached to the channel (off the main path and not a channel-like loop) |

Codes 1–4 are exactly the `channel` band. Every count is also in the QA table.

**QA table columns:** `ds_order`, `year`, `status`, `coverage`, `channel_km2`, `core_km2`, `score` (area vs 5-year running median, in robust SDs), `rerun`, `channel_km2_before_rerun`, `envelope_threshold` (used for the kept mask), pixel counts `seed_px`, `bridge_px`, `channel_px`, `core_px`, `not_connected_px`, `removed_wide_px`, `dead_end_px`, `small_px`, `filled_px`, `floodplain_px`, counts `bridges_joined`, `links`, `links_main`, `links_thread`, `links_floodplain`, `links_dead_end`, `dead_ends_removed`, `through` (a piece linked the window's terminals), `mean_width_m` (area / reach length), `width_over_W` (relative to `width_ref`), `q_scene_median`, `q_scene_pctile` and `stage_flag` (from the `discharge` step, when it has run), `flags`.

Flags: `sparse`, `inferred`, `marginal` (W < 3 px), `unresolved` (W < 2 px), `area_high`, `area_low`, `low_stage`, `high_stage`, `no_discharge`.

## Checks (phase 1)

**Synthetic rivers** (300 × 450 px, 1984–2026, Ok Tedi-like observation density; IoU against the known near-bankfull channel in observed years; A_R = new area visited within Δt years relative to the channel):

| Scenario | IoU mean (min) | False channel, mean | A_R at Δt = 2, 5, 10 (processed / truth) |
|---|---|---|---|
| Migration only, W = 8 px | 0.94 (0.86) | 0.6% | 0.111/0.105, 0.278/0.262, 0.559/0.526 |
| Neck cutoff, W = 8 px | 0.93 (0.86) | 0.7% | 0.118/0.112, 0.298/0.281, 0.597/0.564 |
| Narrow, W = 3 px | 0.96 (0.88) | 1.2% | 0.234/0.229, 0.586/0.572, 1.172/1.146 |
| Avulsion, W = 8 px | 0.91 (0.64) | 2.8% | 0.127/0.124, 0.323/0.313, 0.664/0.647 |

The oxbow is removed from the first year after the cutoff. The avulsion's new course is mapped in its first year (89% of it); the low IoU years are the three in which the abandoned course is still wet and joined at both ends, which is mapped by design (anabranch). A single-year threshold on the same synthetic river scores IoU 0.75.

**Ok Tedi** (20 reaches, 1984–2026, 860 reach-years: 758 observed, 16 sparse, 26 inferred, 60 no data; about 75 min on 2 CPUs). Against a single-year baseline (wet in ≥ 50% of clear looks, components touching the reach line), over observed years:

| | Baseline | Processed |
|---|---|---|
| Area CV per reach, median of 20 | 0.56 | 0.17 |
| Median year-to-year area change | 26% | 5% |

27 reach-years are flagged as area outliers. Most are stage, not mapping error: the lower reaches (15–19) are high in 1999–2000 and low in 2015, the El Niño drought year when the Fly was too low for shipping.

Known limits, for phases 2–3:

- The upper Ok Tedi (reaches 1–4, W = 3.5–3.7 px) never forms one path between its window's terminals, so the fallback (all seeded pieces) applies. Masks are clean (typically 1–4 pieces against 10–30 for the baseline) but not strictly continuous.
- Floodplain water joined to the channel through a neighbouring reach, or reaching a cut line, survives in some wet years on the lower Ok Tedi (e.g. reach 16 in 2009–2011 and 2017–2019). The channel-belt / HAND corridor (phase 3) targets this; the stage flag (phase 2) will separate stage-driven area changes from mapping errors.
- A tributary entering at the top of reach 2 is kept in some years (tributary handling: phase 3).

## Settings

The `process:` section of `rivmapper.yaml` (all overridable per river); see the template written by `rivmapper init`.
