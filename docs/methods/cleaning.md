# Learned mask cleaning (step `clean_masks`)

The rule-based extraction (`processing.md`) keeps some water that a person
immediately sees is not river channel: flood water and lakes fused to the
bank, floodplain water draped along the channel, scraggly bank-side water
from short wetting events, and oxbows or partly abandoned loops that are
still joined to the river. In years with few clear looks it also leaves
gaps in the main channel. `clean_masks` removes such pieces with a
classifier trained on human labels, drops tiny bits of water not seen in
either neighbouring year, and reconnects a broken main channel.
`masks/cleaned/` is the final channel product; `masks/processed/` is kept
unchanged for comparison.

Run order for a river: `process_masks` → `export_rgb` (imagery for the
labelling pages) → `clean_masks` without a model (piece tables) → `label`
→ `ml train` → set `process.clean_model` → `clean_masks` → `label --review`
→ `ml train` again → `clean_masks` → `gif` to inspect.

## Pieces (`process/units.py`)

Each processed reach-year mask is split into

- the **main corridor**: the least-cost pixel route between the points where
  the channel crosses the reach edge, keeping about one typical half-width
  `h` from the bank (cost `1 + 4·|d − h| / h`, cheaper on the low-flow core),
  and every channel pixel within `h + 0.25 W` of it. It follows the river's
  normal-width channel and does not detour through lobes fused to the bank.
  Ground costs 10,000 per pixel: the route stays on connected channel
  wherever it exists and crosses ground only where the mask is broken
  (cloud, a narrow stretch below 30 m), by the fewest ground pixels. So the
  main channel beyond a gap is not left as a removable piece, and the route
  never shortcuts a meander neck. The raster border counts as a reach edge.
  Where the mask stops short of the reach edge (a narrow upper river, cloud),
  the two ends of the channel along the reach's long axis are route ends too,
  so the whole main channel is in the corridor.
  The corridor is then smoothed: ragged bank-side fringe (protrusions
  narrower than about 0.7 W that reach more than 0.25 W beyond the smoothed
  channel; opening radius 0.35 W) is left out of it and becomes a candidate
  piece or tiny bit instead of being kept unconditionally. Pixels within
  half a half-width of the route always stay, so the corridor stays
  connected. On Ok Tedi reach 15 this cut the share of one- or two-year
  bank-side water inside the corridor from 37% to 12% and moved 1% of
  channel that is present in 10 or more years out of it (to be judged as a
  piece; persistence keeps it). On the Bermejo, whose corridor is a single
  narrow thread, it changes nothing.
  It is never removed, so cleaning cannot break the river.
- **pieces**: connected groups of the remaining channel pixels of at least
  0.5 W². One contact with the corridor: a bank lobe, dead-end arm or bar
  head; two or more: a loop (side channel, braid thread, cut-off loop,
  floodplain water along the bank).

Each piece is measured in units of W: area, skeleton length, mean and
maximum width, width relative to the main channel, compactness,
raggedness, contacts with the corridor, share of its edge on the corridor,
farthest distance from the corridor, share on the low-flow core, mean
probabilities, persistence (share of the piece that was channel 1 and 2
years before and after), its pixels' channel frequency over the whole
record, its share of the reach-year's channel, and its main-channel history:
the share of the piece that was in the main corridor in any earlier year
(`former_main`) and in the previous 5 years (`recent_main`). An oxbow is
former main channel now off the main path, which the shape and persistence
features alone did not separate from an active side channel. Context: stage
percentile, braided reach, sparse/inferred year. The channel width in
pixels is left out on purpose: on the Ok Tedi it acted as a proxy for
position along the river (alone it scored AUC 0.89, because the wide lowland
reaches held most of the floodplain water) and would not transfer to other
rivers. Without it, leave-one-reach-out accuracy was 0.87 (AUC 0.94) on 276
Ok Tedi labels.
Every piece and its features are written to
`masks/pieces/{river}_r{NNN}_pieces.csv`.

## Labels (`rivmapper label`)

`rivmapper label rivmapper.yaml --n 300` samples pieces stratified by reach,
loop vs lobe and size, and writes an offline page
(`{river}/labels/bundle/index.html`). Each image shows the year's
true-colour Landsat composite (`rivmapper run … --steps export_rgb`; median
of the year's kept scenes, red/green/blue, 0–0.3 reflectance) with the piece
outlined, and the channel masks of the year before, the year and the year
after. Keys 1 / 2 / 3: channel / not channel / unsure. Labels are exported
as `{river}_labels.csv` with a pixel inside each piece, so they still match
if the piece code changes.

Channel = wetted, active river: the main channel and side channels or braid
threads carrying flow. Not channel = floodplain or flood water, lakes,
abandoned or partly abandoned oxbows (even where still joined to the river),
dry bars, artefacts. Pieces that are half one, half the other: unsure.

**Review rounds.** `rivmapper label rivmapper.yaml --review --n 150`
samples only pieces the current model keeps (at least 2 W², not yet
labelled) into `labels/review/`, exported as `{river}_labels_review.csv`.
Random sampling rarely shows the model's costly keep mistakes (an oxbow
kept as channel); a review round targets them. Training reads every
`{river}_labels*.csv` in `labels/`.

## Model (`rivmapper ml train`)

Gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`,
balanced classes; missing values coded −1). Accuracy is estimated by
grouped cross-validation holding out whole reaches, so the score reflects
reaches the model has not seen. The model file carries the feature list,
label counts and cross-validated metrics (also written as JSON beside it).

## Cleaning

With `process.clean_model` set, `clean_masks`, for every reach-year:

1. predicts P(channel) for every piece and removes pieces below
   `process.clean_threshold` (0.5);
2. removes **tiny bits** (channel off the corridor in groups under 0.5 W²,
   too small to be pieces) unless at least half of the bit was channel the
   year before or after; a reach-year without either neighbour keeps them;
3. **reconnects** a main channel broken into segments: the route between
   segments is traced through ground with evidence of water (wet in at least
   one look that year, DSWE classes 1–3, or channel in the year before or
   after) at 100 per pixel (10,000 without evidence; connected channel costs
   1–5 per pixel, so the route never shortcuts a meander neck), and the gap
   stretches are filled out to the typical half-width on evidence pixels
   only.

It writes `masks/cleaned/`: the four bands of `processed/` (`core`
restricted to the cleaned channel) plus a fifth, `filled` (1 = reconnected
channel, also set in `channel`), so observed and reconnected channel stay
distinguishable. The piece tables gain `p_channel` and `removed`; the QA
table gains `channel_km2_cleaned` (filled pixels included),
`channel_km2_filled`, `pieces_removed`, `tiny_removed_km2`,
`n_obs_median` (median clear Landsat looks over the reach, from the raw
counts) and `low_looks` (below half the river's median `n_obs_median` over
all its reach-years, or below 3). The rule is relative because rivers differ
far more than years do: the median reach-year has 4 clear looks per pixel on
the cloudy Ok Tedi and 11 on the Bermejo; a fixed limit of 8 flagged 721 of
the Ok Tedi's 800 reach-years. Low-look years are where the
main channel breaks up (Bermejo reaches 1 and 3 in 1991 and 1994: 3–6 clear
looks, so a pixel wet in 1–2 of them reads as an uncertain wet share);
reconnecting filled 2.3 and 2.5 km² there. `processed/` is left untouched
for comparison. Without a model the step only writes the piece tables
(labelling mode). `rivmapper gif` shows reconnected pixels in teal.

Synthetic check (`tests/test_clean.py`): with labels taken from the known
truth, cleaning removes more false-channel than true-channel pixels, and
the only pixels it adds are the reconnected ones in the `filled` band.

## Results (Ok Tedi–Fly and Bermejo, Nov 2026)

**Labels.** Per river, 300 pieces from a stratified random sample (by reach,
lobe or loop, and size) and 150 from a review round of kept pieces: 908
labelled channel or not channel across both rivers (pieces marked unsure are
not used). 16 early Ok Tedi labels no longer match a piece, because the
corridor fixes (gap bridging, channel ends, smoothing) moved those stretches
into the main channel.

**Model.** One model for both rivers (`models/pieces_v3.joblib`; metrics in
the JSON beside it). Before the review labels, grouped cross-validation
gave accuracy 0.84 and AUC 0.91 on 534 labels (v2, without the history
features: 0.83 and 0.90). By river (v2): Ok Tedi 0.88 (AUC 0.94), Bermejo
0.82 (AUC 0.89). Trained on one river and scored on the other: Ok Tedi to
Bermejo 0.66 (AUC 0.76; 37% of true channel pieces removed), Bermejo to Ok
Tedi 0.79 (AUC 0.86). Training on both rivers improved each.

**Cleaning** (model v3, before the review labels):

| | Ok Tedi | Bermejo |
|---|---|---|
| Channel area removed, median reach-year | 0.9% | 1.9% |
| 90th percentile | 7.8% | 9.9% |
| Of the removed area, tiny bits | 2% | 2% |
| Reconnected (filled) | 9 km² in 111 of 800 reach-years | 179 km² in 230 of 896 reach-years |
| Area CV per reach, median: processed → cleaned | 0.094 → 0.072 | 0.116 → 0.101 |

On the Ok Tedi the floodplain reaches (14–18) gain most: their CV falls by a
third. The Bermejo's variability is mostly in the main channel itself,
which cleaning does not touch.

## Known limits

- **What remains.** Occasional flicker of cut-off loops and side channels
  (e.g. the Ok Tedi reach 19 oxbow returns in some years), bank-side water
  in flood years, and bar pixels attached to the main thread in braided
  reaches (inside the corridor, so never removed). These are short-lived
  relative to the multi-year windows over which reworking is measured and
  are expected to average out in the migration timescale T_CM; they were
  accepted rather than tuned away (Nov 2026).
- **Reconnected channel is inferred**, not observed: it is set in `channel`
  and marked in `filled`. Analyses should be checked with and without it,
  especially on the Bermejo's braided reaches, where it is largest.
- **New rivers need labels.** Transfer between two rivers was weak (0.66–
  0.79 accuracy), so label each new river (about 150 random and 150 review
  pieces, roughly half an hour) and retrain on all rivers. Once several
  rivers are labelled, the held-out-river score will show whether new
  rivers still need their own labels.
- **Label rule.** Channel = the wetted, active river (main channel, flowing
  side channels and braid threads); oxbows count as not channel even where
  still joined. This differs from the permissive rule in `process_masks`
  (cut-off loops open at both ends are kept there), on purpose: the
  extraction keeps candidates and the cleaning judges them.
