# Discharge on scene dates (step `discharge`)

**Status: built (phase 2); not yet run on real data.** Code: [`rivmapper_v2/discharge.py`](../../rivmapper_v2/discharge.py).

## Why

An annual mask is built from whatever scenes were clear that year. In one year they may all fall in the dry season, in another mostly during floods. Channel area then changes with water level, not with the river. This step measures that: for every reach and year, the modelled discharge on the dates of the scenes that went into its counts, compared with the reach's own record. Years mapped mostly at low or high water are flagged, so their areas can be read, or left out, accordingly.

On the Ok Tedi, phase 1 flagged the lower reaches as unusually wet in 1999–2000 and unusually dry in 2015. In 2015 the Fly was too low for shipping during the El Niño drought. Those are the years this step should explain.

## Data

GEOGLOWS v2 retrospective simulation: daily discharge from 1940 to the present on the TDX-Hydro river network, with global coverage. It is published under CC BY 4.0 through the AWS Open Data Program (`s3://geoglows-v2/retrospective/daily.zarr`, variable `Q`, dimensions `river_id` × `time`). It is read anonymously, with no account needed. The model table (`v2-model-table.parquet`: `LINKNO`, outlet `lat`/`lon`, drainage area) is registered as dataset `geoglows_table` and downloaded once into the shared datasets folder.

GRADES-hydroDL is kept as a possible cross-check only.

## Workflow

For each river (needs the reach lines and the scene manifest from `extract_masks`, raw product v2):

1. **Join each reach to one GEOGLOWS river.**
   - Take the reach's downstream end: the end of its reach line nearer the next reach (for the last reach, the end farther from the previous one).
   - Among GEOGLOWS links whose outlet lies within `search_km` (default 5 km), keep those whose drainage area agrees with the reach's (SWORD `facc`) within a factor `area_tolerance` (default 2). Take the nearest. The area test stops a small tributary joining near the confluence from being chosen over the mainstem.
   - If none matches on area, take the best area match within twice the radius (`best_area_2x_radius`). Failing that, the nearest outlet (`nearest_unmatched`).
   - `river_id` in the settings overrides the join for every reach.
2. **Fetch** daily discharge for the joined links over the mask years.
3. **Stage statistics per reach-year**, from the *kept* scenes in the scene manifest:
   - `q_scene_median`: median discharge over the scene dates, each scene weighted by its clear fraction over the reach (a scene that saw most of the reach counts more);
   - `q_scene_min`, `q_scene_max`, `n_scene_dates`;
   - `q_year_median` (all days of that year) and `q_record_median` (all days of the mask years);
   - `q_scene_pctile`: the percentile of `q_scene_median` in the reach's daily record over the mask years;
   - `stage_flag`: `low_stage` below `low_stage_pctile` (default 20), `high_stage` above `high_stage_pctile` (default 80).

`process_masks` joins `q_scene_median`, `q_scene_pctile` and `stage_flag` into its QA table and adds the flag to `flags`. Without this step's outputs, the columns are empty and `flags` says `no_discharge`.

The step is **optional**: if it fails (no network, no scene manifest), later steps still run.

## Outputs (`{river}/discharge/`)

| File | Content |
|---|---|
| `{river}_geoglows_join.csv` | One row per reach: `river_id`, outlet lon/lat, distance (km), link and reach drainage areas (km²), their ratio, join `method` |
| `{river}_geoglows_daily.csv` | Daily discharge (m³/s): `date`, one column per joined `river_id` |
| `{river}_stage.csv` | One row per reach-year: the statistics above |
| `{river}_discharge_run.yaml` | Run record |

## Parameters (`discharge:`; all can be set per river)

| Parameter | Default | Meaning |
|---|---|---|
| `search_km` | 5 | Search radius at the reach's downstream end |
| `area_tolerance` | 2 | Link and reach drainage areas must agree within this factor |
| `low_stage_pctile` | 20 | Scene-date discharge below this percentile → `low_stage` |
| `high_stage_pctile` | 80 | … above this percentile → `high_stage` |
| `river_id` | — | Force one GEOGLOWS river for every reach |

## Limitations

- **Modelled, not gauged.** GEOGLOWS is a global model. Its timing and relative magnitudes are what matter here, since the flag is a percentile within the river's own record, but small and regulated rivers will be less reliable.
- **One link per reach.** Fine for a mainstem reach; on rivers with large tributaries joining within a reach, discharge changes along it.
- **Scene dates are UTC acquisition dates** matched to daily means.

## References

- Hales, R. C., et al. Advancing global hydrologic modeling with the GEOGloWS ECMWF streamflow service. *Journal of Flood Risk Management*. https://doi.org/10.1111/jfr3.12859
- GEOGLOWS Hydrological Model Version 2, accessed from https://registry.opendata.aws/geoglows-v2 (CC BY 4.0).
