# Mask processing (step `process_masks`)

**Status: in development.** Phases 0 (foundations) and 1 (core pipeline) are done: the step writes annual channel masks, a QA table and QA figures. Scene screening, discharge, calibration of the envelope threshold and the channel-belt corridor come in phases 2–3. The full design and build plan are in the *RivMapper mask processing plan* (shared doc, Oct 2026).

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
| Scene screening | Extraction drops scenes with widespread false water, and writes a scene manifest (raw product v2) | 2 |
| Temporal model | Per-pixel hidden Markov model on the counts (dry / high-stage margin / channel) gives a channel probability every year | 1 ✅ |
| Channel extraction | Seeds (SWORD + last year's channel), width-scaled gap bridging, through-flow connectivity, removal of over-wide and dead-end water | 1 ✅ (corridor: 3) |
| Year check | Reach area against its 5-year running median; outliers rerun or flagged | 1 ✅ |
| Calibration | Envelope threshold matched to bankfull width | 3 |
| Stage flag | Modelled discharge (GEOGLOWS v2) on scene dates | 2 |

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
8. **Over-wide blobs.** Parts wider than 2 × 1.25 W not on the reach line (floodplain lakes and flooded fields joined to the channel) are removed.
9. **Dead ends.** The skeleton is peeled from free ends (not at a terminal) down to the through-flow network; branches longer than 3 W (a cutoff still joined at one end, a tributary mouth) are removed with the mask pixels closest to them. Shorter spurs (bar heads, bank irregularities) stay. Never more than half of the mask is removed this way.
10. **Clean.** Drop pieces smaller than 4 W²; fill dry holes smaller than 0.25 W² (holes bigger than that are bars and islands, kept).
11. **Core product** = channel ∩ core.

Anabranches joined to the network at both ends are channel by design, so a newly abandoned course that is still wet and connected at both ends is mapped until it stops being so.

## Year check ✅

Each reach's channel area in observed and sparse years is compared with its 5-year running median: score = (A − median) / (1.4826 × MAD), with the MAD floored at 5% of the median area, so a steady river only flags changes of about 15% or more. Years scoring above +3 (too much water: usually false water or a flood) are rerun once with the envelope threshold raised by 0.2 and the rerun is kept if it scores closer to the median. Years still beyond ±3 are flagged `area_high` or `area_low`; their masks are written, so analyses can choose to drop them.

## Outputs ✅

| File | Content |
|---|---|
| `masks/processed/{river}_r{NNN}_{year}.tif` | 4 bands, uint8, 255 outside the reach polygon and in `no_data` years: `channel` (0/1), `core` (0/1), `p_envelope` and `p_core` (× 100). Tags: status and flags |
| `masks/model/{river}_r{NNN}_posterior.tif` | HMM posteriors, P(channel) then P(margin) per year (× 100) |
| `masks/{river}_process_masks_qa.csv` | One row per reach-year: status, coverage, channel and core area (km²), area score, rerun, through, bridges joined, dead ends removed, flags |
| `masks/qa/{river}_r{NNN}_qa.png` | Area by year with status and outliers, and six mask thumbnails |
| `masks/qa/{river}_overview.png` | Channel area by reach and year relative to each reach's median |
| `masks/{river}_process_masks_run.yaml` | Run record, with the fitted HMM parameters |

Flags: `sparse`, `inferred`, `marginal` (W < 3 px), `unresolved` (W < 2 px), `area_high`, `area_low`.

## Checks (phase 1)

PHASE1_RESULTS

## Settings

The `process:` section of `rivmapper.yaml` (all overridable per river); see the template written by `rivmapper init`.
