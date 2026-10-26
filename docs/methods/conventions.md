# Coordinate reference systems and data conventions

The conventions every RivMapper v2 product follows, so areas, lengths and pixel overlays are consistent across steps. Keep this page up to date as steps are added.

Status: ✅ implemented · 🔜 agreed, not yet implemented

## Rule

Each river has **one projected analysis CRS**. Every product used for measurement (lengths, areas, overlays) is stored in that CRS. External source data stays in its native CRS and is reprojected when read.

## Products

| Product | Location | CRS | Status |
|---|---|---|---|
| Continental SWORD | `data/SWORD/` | EPSG:4326, as distributed (never modified) | ✅ |
| Extracted SWORD reaches and nodes | `{river}/sword/` | EPSG:4326. An exact subset of the source: its `x`/`y` attributes are lon/lat. Steps that use it reproject on read. | ✅ |
| Reach lines, reach polygons, valley line | `{river}/reaches/` | River analysis CRS | ✅ |
| Raw masks (GEE export) | `{river}/masks/raw/` | River analysis CRS, on the river's fixed 30 m grid | ✅ |
| Processed masks | `{river}/masks/processed/` | River analysis CRS, same grid as raw | 🔜 |
| HMM probability cache | `{river}/masks/model/` | River analysis CRS, same grid as raw | 🔜 |
| Channel belt (Nyberg et al., 2023, or mapped by hand), JRC occurrence | inputs | Reprojected into the river analysis CRS before any intersection or area calculation | 🔜 |
| Summary tables | CSV / YAML | No CRS; lengths in m, areas in m² or km², all computed in the analysis CRS | ✅ |

## Choosing the analysis CRS

1. **Default: UTM** for the zone containing the centre of the river's bounding box. That's EPSG:326zz north of the equator and 327zz south. ✅
   - Within about 3° of the zone's central meridian, UTM scale error is under about 0.1% and area error under about 0.2%.
   - Examples: Ok Tedi–Fly is in zone 54 (EPSG:32754); the Beni test reach is in EPSG:32719.
2. **Long rivers: Lambert azimuthal equal-area (LAEA) centred on the river** ✅
   - Used when the river extends more than about 3° of longitude from the chosen UTM zone's central meridian (e.g. the Yukon, about 141–165°W).
   - It preserves area exactly. Length distortion stays small over about 1,000 km.
   - It has no EPSG code, so it is stored as WKT, which GeoPackage, GeoTIFF, rasterio, QGIS and Earth Engine all accept.
3. **Chosen once per river and reused** ✅
   - The reaches step picks the CRS, writes the reach products in it, and records it (with the reason) in `{river}_reaches_run.yaml`.
   - Later steps take the CRS from the reach polygons, so every product for a river shares one CRS.
   - Override per river with `reaches: analysis_crs:` in the YAML, or an `analysis_crs` column in the rivers table (e.g. `EPSG:32754`).

## Raster grid ✅

- **Pixel size:** 30 m (Landsat).
- **Grid origin:** snapped to multiples of 30 m in the analysis CRS, one grid per river: the top-left corner of the union of the reach polygons, rounded outward. Each reach raster is the smallest window of that grid covering its polygon, so every reach and every year lines up pixel for pixel. Downloads use Earth Engine `computePixels` with an explicit pixel grid (CRS + affine transform) to guarantee this.
- **One reprojection only:** imagery is reprojected once, in Earth Engine, at export, by nearest-neighbour. Local processing never resamples.

## Raster values

**Raw masks** (`masks/raw/`) ✅: five uint8 count bands, `n_obs` and `n_wet_1` … `n_wet_4`. These are clear observations in the calendar year, and how many of them had DSWE class 1..k. 255 = outside the reach polygon (nodata), and counts are capped at 254. `n_obs = 0` inside the polygon means no clear observation that year.

**Processed masks** (`masks/processed/`) 🔜:

| Value | Meaning |
|---|---|
| 0 | Dry (observed, not water) |
| 1 | Water |
| 255 | No data: outside the reach polygon, or no clear observation |

uint8, nodata 255. Keeping "dry" and "no data" separate is essential for area and mobility metrics. v1-era masks used 0 for both.

## Units and naming

- **Lengths in m; areas in km²** in vector attributes (`reach_ar`, `us_drain_ar`). SWORD `facc` is in km².
- **Folders and files are lowercase**, and every file name starts with the river name (`{river}_...`).
- **Each step writes `{river}_{step}_run.yaml`**, recording parameters, versions and diagnostics, including the analysis CRS.
