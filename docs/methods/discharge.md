# Discharge on scene dates (step `discharge`)

**Status: built (phase 2); GloFAS source added Oct 2026, tested offline only, not yet run on real data.** Code: [`rivmapper_v2/discharge.py`](../../rivmapper_v2/discharge.py) (join, stage statistics, step) and [`rivmapper_v2/glofas.py`](../../rivmapper_v2/glofas.py) (GloFAS access).

## Why

An annual mask is built from whatever scenes were clear that year. In one year they may all fall in the dry season, in another mostly during floods. Channel area then changes with water level, not with the river. This step measures that: for every reach and year, the modelled discharge on the dates of the scenes that went into its counts, compared with the reach's own record. Years mapped mostly at low or high water are flagged, so their areas can be read, or left out, accordingly.

On the Ok Tedi, phase 1 flagged the lower reaches as unusually wet in 1999–2000 and unusually dry in 2015. In 2015 the Fly was too low for shipping during the El Niño drought. Those are the years this step should explain.

## Data

### GloFAS historical reanalysis (default, `source: glofas`)

The Global Flood Awareness System (GloFAS) is part of the Copernicus Emergency Management Service. Its historical reanalysis runs the LISFLOOD hydrological model forced by the ERA5 atmospheric reanalysis, and gives daily river discharge on a global 0.05° grid (about 5.5 km at the equator; 90° N to 60° S), from 1 January 1979 to near the present.

We use GloFAS rather than GEOGLOWS because it is the more widely used and cited global discharge reanalysis, and it is what our lab already uses.

- **Dataset:** `cems-glofas-historical` on the CEMS Early Warning Data Store (EWDS), <https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical>, DOI [10.24381/cds.a4fdd6b9](https://doi.org/10.24381/cds.a4fdd6b9). Licence: CEMS-FLOODS datasets licence.
- **Variable:** river discharge in the last 24 hours (m³/s), NetCDF variable `dis24`.
- **Version** (`glofas_version`, default `4.0`): each GloFAS release comes with its own reanalysis. Versions 4.0 and 5.0 (released July 2026) are both offered. Keep one version for a whole project; it is recorded in each run record.
- **Product type** (`glofas_product_type`): `consolidated` is forced by ERA5 and updated monthly. `intermediate` is forced by ERA5T and covers the most recent months, before the consolidated data exist.
- **Upstream-area map:** the area draining to each grid cell (m²), used for the join. It is registered as dataset `glofas_uparea` and downloaded once (~0.1 GB) into the shared datasets folder (`data/GloFAS/`). It comes from the JRC's *LISFLOOD static and parameter maps for GloFAS* (CC BY 4.0): `upArea_repaired.nc` for v4.0 and `upArea_repaired_correctedmetadata_3000.nc` for v5.0. If `glofas_dir` holds a file whose name contains `uparea` (for example `uparea_glofas_v4_0.nc` from the CEMS auxiliary data page), that file is used instead.

### GEOGLOWS v2 (alternative, `source: geoglows_v2`)

GEOGLOWS v2 retrospective simulation: daily discharge from 1940 to the present on the TDX-Hydro river network, published under CC BY 4.0 on AWS Open Data (`s3://geoglows-v2/retrospective/daily.zarr`). It needs no account. Each reach is joined to a river link (`LINKNO`) using the GEOGLOWS model table (dataset `geoglows_table`). Everything below applies to both sources unless it says GloFAS.

GRADES-hydroDL is kept as a possible cross-check only.

## Access: one-time setup for GloFAS (Windows)

GloFAS data are free but need an EWDS account. RivMapper downloads with the official EWDS Python client, `ecmwf-datastores-client` (installed by `environment.yml`).

1. **Create an account** at <https://ewds.climate.copernicus.eu> (top right, *Login / Register*), then log in.
2. **Accept the licence.** Open the [dataset's *Download* tab](https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical?tab=download) and scroll to the bottom. Under *Terms of use*, accept the CEMS-FLOODS licence. You do this once per account; without it every request fails, and the step says so.
3. **Copy your API key.** Open your profile page (<https://ewds.climate.copernicus.eu/profile>, or click your name) and copy the *Personal Access Token*.
4. **Save the key in a file.** In Notepad, create a file with exactly these two lines, using your own token:
   ```
   url: https://ewds.climate.copernicus.eu/api
   key: 01234567-89ab-cdef-0123-456789abcdef
   ```
   Save it as `.ecmwfdatastoresrc` in your home folder, `C:\Users\<your user name>\.ecmwfdatastoresrc`. In Notepad's *Save as* box, set *Save as type* to *All files* and type the name in quotes, `".ecmwfdatastoresrc"`, so Notepad does not add `.txt`. To keep the file somewhere else, set the environment variable `ECMWF_DATASTORES_RC_FILE` to its path.
5. **Install the client** if your environment predates this step: `conda install -c conda-forge ecmwf-datastores-client xarray netcdf4` (in the `rivmapper_v2` environment).

RivMapper always sends requests to the EWDS address. The key alone decides who you are, so a key in the environment variable `ECMWF_DATASTORES_KEY` also works. If the key is missing, the step names the file to create. If the licence has not been accepted, the step links the page where you accept it.

## Workflow

For each river (needs the reach lines and the scene manifest from `extract_masks`, raw product v2):

### 1. Join each reach to one model river

- Take the reach's downstream end: the end of its reach line nearer the next reach (for the last reach, the end farther from the previous one).
- Candidates are GloFAS grid cells (cell centres) or GEOGLOWS link outlets within `search_km` (default 5 km) of that point.
- Keep the candidates whose upstream area agrees with the reach's (SWORD `facc`, km²) within a factor `area_tolerance` (default 2), and take the nearest. On a 0.05° grid, several cells lie within a few kilometres of any point, and next to a confluence the cell holding the reach end may belong to a tributary. The area test rejects it in favour of the mainstem cell (`area_matched`).
- If none matches on area, take the best area match within twice the radius (`best_area_2x_radius`). Failing that, the nearest candidate (`nearest_unmatched`). Check these reaches in the join table.
- **Manual override** (both sources): `manual_lon` and `manual_lat` join every reach to the model river at that point. For GloFAS, that is the grid cell containing the point. For GEOGLOWS, it is the link whose outlet is nearest to it. The join method is then `manual`. For GEOGLOWS, `river_id` (a `LINKNO`) still works too.

A GloFAS cell is identified by its row and column on the global 0.05° grid (row 0 at 90° N, column 0 at 180° W), written `r{row}c{col}`, for example `r1916c6425`. The id does not depend on which file or bounding box the cell was read from.

### 2. Get daily discharge (GloFAS)

For each year from `masks: year_start` to `year_end`, from the first of these that has it:

1. **Files already on disk** in `glofas_dir`. Any NetCDF file in that folder (searched recursively) that holds GloFAS discharge (`dis24`), covers the joined cells and covers the whole year is used. Files may span several years and be larger than the river. Common coordinate names (`valid_time`/`time`, `latitude`/`lat`, `longitude`/`lon`) are understood.
2. **This river's download cache**, `{river}/discharge/glofas/{river}_glofas_v4_0_consolidated_{year}.nc`.
3. **A new request to EWDS** for that year, saved to the cache. The file is written under a temporary name and renamed when complete, so an interrupted download is never mistaken for a finished one.

A rerun, or a run resumed after an interruption, therefore downloads only the years it does not already have.

**One request per river per year.** Each request asks for one variable, all days of one year (`year`, `month` 01–12, `day` 01–31, variable `average_river_discharge_in_the_last_24_hours` with `timespan: time_mean`; if EWDS rejects that form, the older CDS form (`hyear`/`hmonth`/`hday`, `river_discharge_in_the_last_24_hours`) is tried once and the accepted form is recorded in the run record), NetCDF, clipped with `area` [N, W, S, E] to the joined cells plus a 0.1° margin, rounded outward to the nearest 0.5°. Rounding keeps the area the same when a rerun changes the join slightly, so the cache stays valid. Yearly requests are the usual pattern for this dataset (EWDS limits the size of single requests) and match the cache's resume unit. A cached file that no longer covers the joined cells is downloaded again.

**Why clip, and why cache per river.** A global year of daily 0.05° discharge is 7200 × 3000 cells × 365 days, about 31 GB uncompressed (about 1.5 TB for 1979–2025). A river's clip of a few degrees is around a megabyte a year. The clipped files are specific to one river's area, so they live with the river's other outputs, under `{river}/discharge/glofas/`, not in the shared `data/` folder. A shared folder of clips keyed by bounding box would grow with every river and be hard to tell apart. The shared, reusable copy of GloFAS is a lab archive, which `glofas_dir` points to. The upstream-area map, which every river needs, is in `data/GloFAS/`.

The values of the joined cells are written to `{river}_discharge_daily.csv`.

### 3. Time alignment

GloFAS stores "discharge in the last 24 hours". The time stamp (`valid_time`, 00 UTC) marks the **end** of the 24-hour averaging period (Harrigan et al., 2020). The value stamped 2005-07-02 00:00 is therefore the mean discharge over 2005-07-01, 00–24 UTC. RivMapper shifts every GloFAS time stamp back by one day, so `{river}_discharge_daily.csv` labels each value with the UTC day it averages.

Scene dates in the scene manifest are UTC acquisition dates (Landsat passes at about 10:00–10:30 local solar time, so the UTC date is the local date except near the date line). A scene on 2005-07-01 is matched to the mean over 2005-07-01. Depending on how EWDS stamps a year's file, the last day of the final mask year may lack a value; that scene date then has no discharge and is left out of the statistics.

GEOGLOWS daily values are used on their stated dates.

### 4. Stage statistics per reach-year

From the *kept* scenes in the scene manifest:

- `q_scene_median`: median discharge over the scene dates, each scene weighted by its clear fraction over the reach (a scene that saw most of the reach counts more);
- `q_scene_min`, `q_scene_max`, `n_scene_dates`;
- `q_year_median` (all days of that year) and `q_record_median` (all days of the mask years);
- `q_scene_pctile`: the percentile of `q_scene_median` in the reach's daily record over the mask years;
- `stage_flag`: `low_stage` below `low_stage_pctile` (default 20), `high_stage` above `high_stage_pctile` (default 80).

`process_masks` joins `q_scene_median`, `q_scene_pctile` and `stage_flag` into its QA table and adds the flag to `flags`. Without this step's outputs, the columns are empty and `flags` says `no_discharge`.

The step is **optional**: if it fails (no network, no EWDS key, no scene manifest), later steps still run.

## Outputs (`{river}/discharge/`)

| File | Content |
|---|---|
| `{river}_discharge_join.csv` | One row per reach: `source`, `river_id` (GloFAS cell id or GEOGLOWS `LINKNO`), `lon`/`lat` (cell centre or link outlet), `cell_row`/`cell_col` (GloFAS), `dist_km` from the reach's downstream end, `model_area_km2` and `reach_area_km2` (upstream areas), `area_ratio`, join `method` |
| `{river}_discharge_daily.csv` | Daily mean discharge (m³/s): `date` (UTC day), one column per joined `river_id` |
| `{river}_stage.csv` | One row per reach-year: the statistics above |
| `{river}_discharge_run.yaml` | Run record: settings, GloFAS request details (dataset, `system_version`, product type, date shift), upstream-area file, and which files were read locally, from the cache or downloaded |
| `glofas/` | GloFAS download cache, one NetCDF per year (GloFAS only) |

## Parameters (`discharge:`; all can be set per river)

| Parameter | Default | Meaning |
|---|---|---|
| `source` | `glofas` | `glofas` or `geoglows_v2` |
| `glofas_version` | `4.0` | GloFAS reanalysis version, `4.0` or `5.0` |
| `glofas_product_type` | `consolidated` | `consolidated` (ERA5) or `intermediate` (ERA5T, recent months) |
| `glofas_dir` | — | Folder of GloFAS NetCDF files already on disk, read before downloading |
| `search_km` | 5 | Search radius at the reach's downstream end |
| `area_tolerance` | 2 | Model and reach upstream areas must agree within this factor |
| `low_stage_pctile` | 20 | Scene-date discharge below this percentile → `low_stage` |
| `high_stage_pctile` | 80 | … above this percentile → `high_stage` |
| `manual_lon`, `manual_lat` | — | Use the model river at this point for every reach (both sources) |
| `river_id` | — | GEOGLOWS only: force one `LINKNO` for every reach |

## Limitations

- **Modelled, not gauged.** GloFAS and GEOGLOWS are global models. What matters here is their timing and relative magnitudes, since the flag is a percentile within the river's own record. Small, regulated or heavily abstracted rivers will be less reliable.
- **Grid resolution.** A 0.05° cell is about 5.5 km across. Rivers much narrower than that, or with an upstream area below a few hundred km², may not be represented by a distinct line of cells. The area match and the join table show where that happens.
- **Different river networks.** GloFAS upstream area comes from its own drainage network (`ldd`), SWORD's `facc` from MERIT Hydro. The two can differ, especially in deltas, flat terrain and multi-channel reaches; `area_tolerance` absorbs ordinary differences.
- **One cell or link per reach.** Fine for a mainstem reach. On rivers with large tributaries joining within a reach, discharge changes along it.
- **Versions differ.** GloFAS v4.0 and v5.0 are separate reanalyses with different calibrations. Do not mix them within a project; `glofas_dir` files must be of the configured version, which cannot be checked from the files themselves.
- **Recent years.** Consolidated data lag a few months behind real time. For the current year use `glofas_product_type: intermediate`, or end the mask years earlier. A cached year is reused as it is: if it was downloaded before the year was complete, delete its file in `{river}/discharge/glofas/` (and rerun with `--overwrite`) to fetch it again.

## References

- Harrigan, S., Zsoter, E., Alfieri, L., Prudhomme, C., Salamon, P., Wetterhall, F., Barnard, C., Cloke, H., & Pappenberger, F. (2020). GloFAS-ERA5 operational global river discharge reanalysis 1979–present. *Earth System Science Data*, 12, 2043–2060. https://doi.org/10.5194/essd-12-2043-2020
- Copernicus Emergency Management Service (2019). River discharge and related historical data from the Global Flood Awareness System. Early Warning Data Store (EWDS). https://doi.org/10.24381/cds.a4fdd6b9
- Alfieri, L., Burek, P., Dutra, E., Krzeminski, B., Muraro, D., Thielen, J., & Pappenberger, F. (2013). GloFAS – global ensemble streamflow forecasting and flood early warning. *Hydrology and Earth System Sciences*, 17, 1161–1175. https://doi.org/10.5194/hess-17-1161-2013
- Choulga, M., Beck, H., Moschini, F., Mazzetti, C., Disperati, J., Grimaldi, S., Salamon, P., & Prudhomme, C. (2023). LISFLOOD static and parameter maps for GloFAS. European Commission, Joint Research Centre (JRC) [Dataset]. http://data.europa.eu/89h/68050d73-9c06-499c-a441-dc5053cb0c86
- Hales, R. C., et al. Advancing global hydrologic modeling with the GEOGloWS ECMWF streamflow service. *Journal of Flood Risk Management*. https://doi.org/10.1111/jfr3.12859
- GEOGLOWS Hydrological Model Version 2, accessed from https://registry.opendata.aws/geoglows-v2 (CC BY 4.0).
