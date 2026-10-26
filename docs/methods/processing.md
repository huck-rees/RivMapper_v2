# Mask processing (step `process_masks`)

**Status: in development.** Phase 0 (foundations) is done; the step currently checks its inputs and writes a run record, and reports `skipped`. The full design and build plan are in the *RivMapper mask processing plan* (shared doc, Oct 2026). This page grows as each phase lands.

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
| Temporal model | Per-pixel hidden Markov model on the counts (dry / high-stage margin / channel) gives a channel probability every year | 1 |
| Channel extraction | Corridor, seeds (SWORD + last year's channel), width-scaled gap bridging, connectivity, removal of over-wide and dead-end water | 1, 3 |
| Year check | Reach area against its 5-year running median; outliers rerun or flagged | 1 |
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

These are used for unit tests now, and in phase 4 to check that processing recovers known reworking and overlap curves.

## Settings

The `process:` section of `rivmapper.yaml` (all overridable per river); see the template written by `rivmapper init`.
