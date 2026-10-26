# Water masks (step `extract_masks`)

How RivMapper v2 extracts annual Landsat water observations from Google Earth Engine. Code: [`rivmapper_v2/gee.py`](../../rivmapper_v2/gee.py) (Earth Engine) and [`rivmapper_v2/masks.py`](../../rivmapper_v2/masks.py) (step logic). Grid and value conventions: [conventions.md](conventions.md).

**Design.** The raw product is a faithful record of what Earth Engine provides for each calendar year. Raw files hold **counts**: how many clear observations each pixel had that year, and how many of those were water at each DSWE level. Raw extraction applies no compositing window, DSWE threshold or percentile. Those choices are made later, in mask processing, from the counts, so they can be compared or calibrated per river without downloading again.

## Imagery

- **Sources:** Landsat Collection 2, Level 2 surface reflectance, Tier 1.
  - Landsat 5 TM, 8 OLI and 9 OLI-2: full record.
  - Landsat 7 ETM+: only 15 Apr 1999 – 30 May 2003, before the scan-line-corrector failure.
- **Bands:** harmonised to Blue, Green, Red, NIR, SWIR1, SWIR2. Reflectance = DN × 0.0000275 − 0.2.
- **Pixel masking:** `QA_PIXEL` bits 0 (fill), 1 (dilated cloud), 2 (cirrus), 3 (cloud), 4 (cloud shadow) and 5 (snow) are masked.
  - *This fixes a bug in the v1 and June-2026 notebooks.* They used Collection 1 bit positions (3 = shadow, 5 = cloud) on Collection 2 data. As a result they masked cloud (bit 3) and snow (bit 5), but never cloud shadow or dilated cloud. Shadows are dark and can be classified as water.

## Water classification

Each clear observation is classified with the Dynamic Surface Water Extent (DSWE) tests of Jones (2019). The classes are 0 not water, 1 high-confidence water, 2 moderate-confidence water, 3 potential wetland, and 4 low-confidence or partial water.

## Annual counts (the raw product)

For each reach and calendar year *Y*, over every clear scene acquired from 1 January to 31 December of *Y*:

| Band | Count |
|---|---|
| `n_obs` | Clear observations (not masked by `QA_PIXEL`) |
| `n_wet_1` | Observations with DSWE class 1 |
| `n_wet_2` | Observations with DSWE class 1–2 |
| `n_wet_3` | Observations with DSWE class 1–3 |
| `n_wet_4` | Observations with DSWE class 1–4 |

The counts are nested: `n_wet_1 ≤ n_wet_2 ≤ n_wet_3 ≤ n_wet_4 ≤ n_obs`. Counts are capped at 254. A year with no scenes still gets a file, with `n_obs = 0` everywhere inside the polygon, and is logged as `no_imagery`.

**From counts to a water mask (processing, planned).** For DSWE level *k*, a pixel's wet frequency in year *Y* is *f* = `n_wet_k / n_obs`. A per-pixel percentile *p* of the binary wet/dry series is then a threshold on *f*. For example, the median (*p* = 50) means water where *f* ≥ 0.5. Pixels with `n_obs = 0` are no data. Choosing *k* and *p* per river (by width matching, SWOT-era calibration, sensitivity plateaus or sensor-transition consistency) is a processing-step decision.

## Download

- **Grid:** each reach-year is downloaded on the river's fixed 30 m grid in the analysis CRS ([conventions.md](conventions.md)), using Earth Engine `computePixels` with an explicit pixel grid.
- **Chunking:** large reaches are fetched in chunks of `chunk_px` × `chunk_px` pixels and assembled locally. Chunks are aligned to the grid, so no merging or resampling is needed.
- **Adaptive tiling:** Earth Engine's memory use grows with pixels × scenes, so a chunk that is fine in most years can fail on a big reach in a dense year (e.g. 2022–23, when Landsat 8 and 9 overlap on adjacent paths). If a chunk fails with a memory, request-size or time-limit error, it is split into quadrants and retried, recursively, down to `min_chunk_px`. Any other error (authentication, bad geometry) fails straight away. Sub-tiles sit on the same 30 m grid, so the file is identical to a single download. The log's `n_tiles` column records how many requests each reach-year needed.
- **Per-tile scene selection:** each tile only pulls the Landsat scenes whose footprint touches that tile (plus a 300 m margin), not every scene touching the whole reach. Scenes that miss a tile contribute zero counts there, so results are unchanged; Earth Engine just does less work. `n_scenes` in the log still counts scenes touching the whole reach.
- **Polygon mask:** pixels whose centre lies outside the reach polygon are set to 255 in every band.

## Outputs

`{data_root}/{river}/masks/`:

| File | Contents |
|---|---|
| `raw/{river}_r{NNN}_{year}.tif` | uint8, five bands (`n_obs`, `n_wet_1` … `n_wet_4`), nodata 255. Tags hold the product version, settings and scene count. |
| `{river}_extract_masks_log.csv` | One row per reach-year: scenes, clear fraction (share of the polygon with `n_obs` ≥ 1), median and 10th-percentile `n_obs`, and status (`ok`, `no_imagery`, `failed`) |
| `{river}_extract_masks_run.yaml` | Run record: parameters, analysis CRS, grid origin, counts |

**Re-runs:**
- Files already made with the same product version and grid are skipped, so an interrupted run resumes.
- Changing `dswe_level` or `percentile` does not invalidate raw files.
- If the product or grid changes after files exist, the step reports `stale`. Use `--overwrite` to rebuild.

## Parameters (`masks:` in `rivmapper.yaml`; all can be set per river)

| Parameter | Default | Used by |
|---|---|---|
| `year_start`, `year_end` | — (required) | Extraction: calendar years to extract |
| `reach_start`, `reach_end` | all | Extraction: reach range (`ds_order`) |
| `chunk_px` | 2048 | Extraction: download chunk size (pixels per side) |
| `min_chunk_px` | 256 | Extraction: smallest tile a failing chunk is split down to |
| `dswe_level` | — | Processing: water = DSWE class 1..`dswe_level` |
| `percentile` | 50 | Processing: per-pixel percentile of the wet/dry series |
| `min_pixels` | 200 | Processing |

Settings from the earlier windowed design (`composite_windows`, `min_clear_fraction`, `min_water_fraction`) are ignored, with a warning.

## References

- Jones, J. W. (2019). Improved automated detection of subpixel-scale inundation: Revised Dynamic Surface Water Extent (DSWE) partial surface water tests. *Remote Sensing*, 11(4), 374. https://doi.org/10.3390/rs11040374
- U.S. Geological Survey. *Landsat Collection 2 Level-2 Science Products* (QA_PIXEL bit definitions and surface reflectance scaling).
