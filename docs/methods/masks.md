# Water masks (step `extract_masks`)

How RivMapper v2 extracts annual Landsat water masks from Google Earth Engine. Code: [`rivmapper_v2/gee.py`](../../rivmapper_v2/gee.py) (Earth Engine) and [`rivmapper_v2/masks.py`](../../rivmapper_v2/masks.py) (step logic). Grid and value conventions: [conventions.md](conventions.md).

Mask processing (`raw/` → `processed/`) is a separate, later step.

## Imagery

- **Sources:** Landsat Collection 2, Level 2 surface reflectance, Tier 1.
  - Landsat 5 TM, 8 OLI and 9 OLI-2: full record.
  - Landsat 7 ETM+: only 15 Apr 1999 – 30 May 2003, before the scan-line-corrector failure.
- **Bands:** harmonised to Blue, Green, Red, NIR, SWIR1, SWIR2. Reflectance = DN × 0.0000275 − 0.2.
- **Pixel masking:** `QA_PIXEL` bits 0 (fill), 1 (dilated cloud), 2 (cirrus), 3 (cloud), 4 (cloud shadow) and 5 (snow) are masked.
  - *This fixes a bug in the v1 and June-2026 notebooks.* They used Collection 1 bit positions (3 = shadow, 5 = cloud) on Collection 2 data. As a result they masked cloud (bit 3) and snow (bit 5), but never cloud shadow or dilated cloud. Shadows are dark and can be classified as water.

## Water classification

Each clear observation is classified with the Dynamic Surface Water Extent (DSWE) tests of Jones (2019). The classes are 0 not water, 1 high-confidence water, 2 moderate-confidence water, 3 potential wetland, and 4 low-confidence water.

A pixel counts as water in that scene if its class is between 1 and `dswe_level`.

## Annual composite

For a target year *Y* and window half-width *w*:

- **Water:** across all clear scenes in years *Y − w* to *Y + w*, take the per-pixel `percentile` (default 50, the median) of the binary water series. A pixel is water if that value is ≥ 0.5, and dry otherwise.
- **No observation:** pixels with no clear observation get 255 (no data).
- **Observation count:** the number of clear observations is kept as band `n_obs`.

**Choosing the window:** `composite_windows` (default `[0, 1, 2]`) are tried in order, and the first that passes both checks is used. The checks are run at 60 m over the reach polygon:

1. **Clear coverage:** at least `min_clear_fraction` (default 0.5) of the polygon has one or more clear observations.
2. **Plausible water area:** the detected water area is at least `min_water_fraction` (default 0.5) of the expected wetted area, `width_ref × reach_len` from the reaches step. This guards against under-detection in persistently cloudy regions such as the Ok Tedi.

If no window passes, the window with the best clear coverage is exported and the year is marked `flagged`. Years with no scenes at any window are marked `no_imagery` and no file is written. The window used is stored in each file and in the log.

## Download

- **Grid:** each reach is downloaded on the river's fixed 30 m grid in the analysis CRS ([conventions.md](conventions.md)), using Earth Engine `computePixels` with an explicit pixel grid.
- **Chunking:** large reaches are fetched in chunks of `chunk_px` × `chunk_px` pixels and assembled locally. Chunks are aligned to the grid, so no merging or resampling is needed.
- **Polygon mask:** pixels whose centre lies outside the reach polygon are set to 255.

## Outputs

`{data_root}/{river}/masks/`:

| File | Contents |
|---|---|
| `raw/{river}_r{NNN}_{year}.tif` | uint8, two bands. `water` is 0 dry, 1 water, 255 no data. `n_obs` is the clear-observation count. Tags hold the settings, composite window and years, scene count and check result. |
| `{river}_extract_masks_log.csv` | One row per reach-year: window, scenes, clear fraction, water and expected area (km²), and status (`ok`, `flagged`, `no_imagery`, `failed`) |
| `{river}_extract_masks_run.yaml` | Run record: parameters, analysis CRS, grid origin, counts |

**Re-runs:**
- Files already made with the same settings are skipped, so an interrupted run resumes.
- If settings change after some files exist, the step reports `stale` rather than mixing old and new files. Use `--overwrite` to rebuild.

## Parameters (`masks:` in `rivmapper.yaml`; all can be set per river)

| Parameter | Default | Meaning |
|---|---|---|
| `year_start`, `year_end` | — (required) | Years to extract |
| `reach_start`, `reach_end` | all | Reach range (`ds_order`) |
| `dswe_level` | — (required) | Water = DSWE class 1..`dswe_level` |
| `percentile` | 50 | Per-pixel percentile of per-scene water |
| `composite_windows` | `[0, 1, 2]` | Window half-widths (years) tried in order |
| `min_clear_fraction` | 0.5 | Clear-coverage check |
| `min_water_fraction` | 0.5 | Water-area check (fraction of expected) |
| `chunk_px` | 2048 | Download chunk size (pixels per side) |

## References

- Jones, J. W. (2019). Improved automated detection of subpixel-scale inundation: Revised Dynamic Surface Water Extent (DSWE) partial surface water tests. *Remote Sensing*, 11(4), 374. https://doi.org/10.3390/rs11040374
- U.S. Geological Survey. *Landsat Collection 2 Level-2 Science Products* (QA_PIXEL bit definitions and surface reflectance scaling).
