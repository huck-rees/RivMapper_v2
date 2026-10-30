# Learned mask cleaning (step `clean_masks`)

The rule-based extraction (`processing.md`) keeps some water that a person
immediately sees is not river channel: flood water and lakes fused to the
bank, floodplain water draped along the channel, and cut-off loops that
flicker in and out with detection. `clean_masks` removes such pieces with a
classifier trained on human labels.

## Pieces (`process/units.py`)

Each processed reach-year mask is split into

- the **main corridor**: the least-cost pixel route between the points where
  the channel crosses the reach edge, keeping about one typical half-width
  `h` from the bank (cost `1 + 4·|d − h| / h`, cheaper on the low-flow core),
  and every channel pixel within `h + 0.25 W` of it. It follows the river's
  normal-width channel and does not detour through lobes fused to the bank.
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
record, and its share of the reach-year's channel. Context: stage
percentile, braided reach, channel width in pixels, sparse/inferred year.
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

Channel = part of the active river, including side channels and cut-off
loops still open at both ends (the same rule as the network selection).
Not channel = floodplain or flood water, lakes, artefacts.

## Model (`rivmapper ml train`)

Gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`,
balanced classes; missing values coded −1). Accuracy is estimated by
grouped cross-validation holding out whole reaches, so the score reflects
reaches the model has not seen. The model file carries the feature list,
label counts and cross-validated metrics (also written as JSON beside it).

## Cleaning

With `process.clean_model` set, `clean_masks` predicts P(channel) for every
piece, removes pieces below `process.clean_threshold` (0.5) and writes
`masks/cleaned/` (same four bands as `processed/`; `core` restricted to the
cleaned channel). The piece tables gain `p_channel` and `removed`; the QA
table gains `channel_km2_cleaned` and `pieces_removed`. `processed/` is
left untouched for comparison. Without a model the step only writes the
piece tables (labelling mode).

Synthetic check (`tests/test_clean.py`): with labels taken from the known
truth, cleaning removes more false-channel than true-channel pixels and
never adds water.
