"""Starter ``rivmapper.yaml`` / ``rivers.csv`` text for ``rivmapper init``."""

from __future__ import annotations

from typing import Any

import yaml

from .config import (
    DischargeSettings,
    MaskSettings,
    ProcessSettings,
    ReachSettings,
    SwordSettings,
)


def _v(value: Any) -> str:
    """Format a scalar as YAML (null for None)."""
    return yaml.safe_dump(value, default_flow_style=True).strip().removesuffix("\n...")


def render_config(
    data_root: str = "./output",
    rivers_table: str = "rivers.csv",
    gee_project: str | None = None,
    sword_dir: str | None = None,
    reaches: dict[str, Any] | None = None,
    masks: dict[str, Any] | None = None,
    process: dict[str, Any] | None = None,
) -> str:
    """Return a commented rivmapper.yaml with the given values filled in."""
    r = {**ReachSettings().model_dump(), **(reaches or {})}
    m = {**MaskSettings().model_dump(), **(masks or {})}
    pr = {**ProcessSettings().model_dump(), **(process or {})}
    d = DischargeSettings().model_dump()
    s = SwordSettings()
    return f"""\
# RivMapper v2 project configuration
# ────────────────────────────────────────────────────────────────────
# Settings here apply to every river in the rivers table. Any setting in
# the `reaches:`, `masks:`, `process:` or `discharge:` sections can be
# overridden for one river by adding a column with the same name to the
# rivers table; blank cells use the value below. Relative paths are
# relative to this file.
#
# Check it:  rivmapper validate rivmapper.yaml
# Run it:    rivmapper run rivmapper.yaml [--steps sword,reaches] [--rivers Beni]

# This project's outputs go under this folder: one subfolder per river,
# with one subfolder per step ({{river}}/sword/, {{river}}/reaches/, ...).
data_root: {_v(data_root)}

# One row per river: river_name, sword_cont_abb, us_sword_reach_id,
# ds_sword_reach_id, plus any per-river overrides.
rivers_table: {_v(rivers_table)}

# Rebuild outputs even if they already exist (CLI: --overwrite).
overwrite: false

sword:
  version: {_v(s.version)}
  # Folder with the continental SWORD geopackages, shared by all projects.
  # null = the shared datasets folder (<repo>/data/SWORD in a git checkout;
  # `rivmapper data path` shows it). Downloaded (~2 GB) on first use, or
  # up front with `rivmapper data fetch sword`.
  dir: {_v(sword_dir)}
  # SWORD `type` codes the tracer may walk through (1 = river).
  river_types: [1]

# Google Earth Engine (step extract_masks). Sign in once with `rivmapper gee auth`,
# then test with `rivmapper gee check rivmapper.yaml`.
gee:
  project: {_v(gee_project)}         # Google Cloud project registered for Earth Engine
  service_account: null      # headless servers only: service-account email
  key_file: null             # ...and its JSON key (relative to this file)

# Step 02: reach creation (see docs/methods/reaches.md).
reaches:
  # valley_widths: reaches have valley length = widths_per_reach x W(A), measured
  #   along a smoothed valley line; W(A) is a width-vs-drainage-area power law
  #   fitted to SWORD widths. drainage_regression: v1 rule on channel length.
  length_method: {_v(r["length_method"])}
  widths_per_reach: {_v(r["widths_per_reach"])}       # k: 100 W = ~9 meander wavelengths, ~20 bends
  buffer_width: {_v(r["buffer_width"])}        # buffer half-width as a fraction of reach (valley) length (required)
  valley_sigma_widths: {_v(r["valley_sigma_widths"])}    # valley-line smoothing scale (channel widths)
  valley_sinuosity_factor: {_v(r["valley_sinuosity_factor"])} # smoothing in sinuous reaches = this x local sinuosity (widths)
  final_merge_fraction: {_v(r["final_merge_fraction"])}   # merge a last reach shorter than this x nominal
  # drainage_regression only: reach_len_m = length_modulator * len_slope * facc + len_intercept
  length_modulator: {_v(r["length_modulator"])}
  len_slope: {_v(r["len_slope"])}            # v1 regression (Ganges, HydroRIVERS UPLAND_SKM)
  len_intercept: {_v(r["len_intercept"])}
  ds_trim_fraction: {_v(r["ds_trim_fraction"])}       # drop a last reach shorter than this x the one above it
  # Projected CRS for every measured product: auto = UTM, or LAEA for rivers
  # spanning > 3 deg from the UTM central meridian; or e.g. "EPSG:32754".
  analysis_crs: {_v(r["analysis_crs"])}
  # Geometry details
  gap_tolerance_m: {_v(r["gap_tolerance_m"])}       # warn on gaps between SWORD reaches larger than this
  smoothing_window_multiplier: {_v(r["smoothing_window_multiplier"])}  # cut-line direction window = this x reach length
  n_tangent_samples: {_v(r["n_tangent_samples"])}
  cut_extent_factor: {_v(r["cut_extent_factor"])}      # cut-line half-length = this x largest buffer
  # How neighbouring polygons are separated: nearest_valley (closest position
  # along the valley line; boundaries perpendicular to the valley), nearest_centerline
  # (closest reach channel centreline) or cut_lines (perpendicular cuts at each
  # boundary; can overlap where the channel loops back).
  boundary_method: {_v(r["boundary_method"])}
  partition_spacing_m: {_v(r["partition_spacing_m"])}   # sample spacing for nearest_valley / nearest_centerline

# Step 03: raw annual DSWE observation counts from Earth Engine (extract_masks).
# See docs/methods/masks.md.
masks:
  year_start: {_v(m["year_start"])}         # required: first calendar year to extract
  year_end: {_v(m["year_end"])}           # required: last calendar year to extract
  reach_start: {_v(m["reach_start"])}         # null = first reach
  reach_end: {_v(m["reach_end"])}           # null = last reach
  chunk_px: {_v(m["chunk_px"])}           # download chunk size (pixels per side)
  min_chunk_px: {_v(m["min_chunk_px"])}        # chunks that hit EE memory/time limits are split down to this
  workers: {_v(m["workers"])}               # reach-years requested from Earth Engine in parallel
  # Scene screening (raw product v2): scenes with widespread false water on
  # reference-dry pixels (far from the channel, never water in JRC GSW) are dropped.
  screen_scenes: {_v(m["screen_scenes"])}
  max_false_water2: {_v(m["max_false_water2"])}    # drop a scene above this share of DSWE 1-2 on reference-dry pixels
  max_false_water3: {_v(m["max_false_water3"])}     # ... or above this share of DSWE 1-3
  ref_dry_widths: {_v(m["ref_dry_widths"])}       # reference-dry pixels are at least this many channel widths from the reach line
  min_ref_px: {_v(m["min_ref_px"])}           # clear reference pixels needed to judge by the two rates
  gross_false_water2: {_v(m["gross_false_water2"])}   # with fewer, still drop above this DSWE 1-2 share ...
  gross_min_ref_px: {_v(m["gross_min_ref_px"])}      # ... given at least this many reference pixels
  screen_scale_m: {_v(m["screen_scale_m"])}       # scale of the screening statistics (m)
  le07_gap_years: {_v(m["le07_gap_years"])}     # years in which SLC-off Landsat 7 is also used
  screen_only: {_v(m["screen_only"])}         # true = write the scene manifest only (quick check, no downloads)
  # Used later by mask processing (not by extraction):
  dswe_level: {_v(m["dswe_level"])}          # water = DSWE class 1..dswe_level (1-4)
  percentile: {_v(m["percentile"])}          # per-pixel percentile of per-scene water (50 = median)
  min_pixels: {_v(m["min_pixels"])}

# Step 04: modelled discharge on scene dates → stage flags (discharge; optional).
# GloFAS (default) needs a free EWDS account and API key; GEOGLOWS needs only
# network access to AWS. See docs/methods/discharge.md.
discharge:
  source: {_v(d["source"])}             # glofas (GloFAS reanalysis, EWDS) or geoglows_v2 (GEOGLOWS v2, AWS)
  glofas_version: {_v(d["glofas_version"])}      # GloFAS reanalysis version: 4.0 or 5.0 (kept fixed for a project)
  glofas_product_type: {_v(d["glofas_product_type"])}  # consolidated (ERA5) or intermediate (ERA5T, last few months)
  glofas_dir: {_v(d["glofas_dir"])}           # folder of GloFAS NetCDF files already on disk (read first; null = none)
  search_km: {_v(d["search_km"])}            # search radius for GloFAS cells / GEOGLOWS links at each reach's downstream end
  area_tolerance: {_v(d["area_tolerance"])}       # model and reach upstream areas must agree within this factor
  low_stage_pctile: {_v(d["low_stage_pctile"])}    # scene-date discharge below this percentile → low_stage
  high_stage_pctile: {_v(d["high_stage_pctile"])}   # ... above this percentile → high_stage
  manual_lon: {_v(d["manual_lon"])}           # manual override: use the model river at this point for every reach
  manual_lat: {_v(d["manual_lat"])}           # (GloFAS: the cell containing it; GEOGLOWS: the nearest link outlet)
  river_id: {_v(d["river_id"])}             # GEOGLOWS only: force one river id (LINKNO) for every reach

# Step 05: raw counts → annual channel masks (process_masks; in development).
# Spatial settings are multiples of the reach's channel width W. See
# docs/methods/processing.md.
process:
  states: {_v(pr["states"])}                     # HMM states: 3 = dry/margin/channel, 2 = dry/channel
  fit_sample_px: {_v(pr["fit_sample_px"])}           # pixels sampled to fit the HMM per river
  envelope_threshold: {_v(pr["envelope_threshold"])}   # near-bankfull mask threshold, or calibrate
  core_threshold: {_v(pr["core_threshold"])}            # low-flow core mask threshold on P(channel)
  max_infer_gap: {_v(pr["max_infer_gap"])}              # empty years within this many years of data are inferred
  use_le07_gap_years: {_v(pr["use_le07_gap_years"])}       # count SLC-off Landsat 7 gap years as observed
  bridge_closing_widths: {_v(pr["bridge_closing_widths"])}     # gap-closing radius (W)
  bridge_max_widths: {_v(pr["bridge_max_widths"])}         # longest reconnecting path (W)
  grow_widths: {_v(pr["grow_widths"])}               # growth from core into the envelope (W)
  max_inscribed_widths: {_v(pr["max_inscribed_widths"])}     # wider blobs off the centreline are removed (W, radius)
  dead_end_widths: {_v(pr["dead_end_widths"])}           # dead-end branches longer than this are pruned (W)
  dead_end_widths_braided: {_v(pr["dead_end_widths_braided"])}  # same, braided reaches
  network_selection: {_v(pr["network_selection"])}       # keep main path + channel-like loops; drop attached floodplain water
  thread_min_widths: {_v(pr["thread_min_widths"])}         # a second thread is kept if its median width is >= this (W) ...
  thread_max_widths: {_v(pr["thread_max_widths"])}         # ... and <= this x the main channel's width ...
  thread_max_p90_widths: {_v(pr["thread_max_p90_widths"])}     # ... with no part wider than this x main width ...
  thread_min_core: {_v(pr["thread_min_core"])}           # ... mostly on the low-flow core (share) ...
  thread_max_width_cv: {_v(pr["thread_max_width_cv"])}       # ... and an even width (CV)
  thread_persistence: {_v(pr["thread_persistence"])}      # last year's channel links pass looser limits (less flicker)
  braided_min_chan: {_v(pr["braided_min_chan"])}         # braided reach: share of SWORD nodes with n_chan_mod > 1
  link_margin_widths: {_v(pr["link_margin_widths"])}        # pixels kept up to a link's half-width + this (W)
  min_component_w2: {_v(pr["min_component_w2"])}          # smallest kept component (W²)
  fill_hole_w2: {_v(pr["fill_hole_w2"])}             # largest dry hole filled (W²)
  outlier_mad: {_v(pr["outlier_mad"])}               # year-to-year area check (MADs)
  use_channel_belt: {_v(pr["use_channel_belt"])}        # limit to the Nyberg et al. (2023) channel belt
  max_hand_m: {_v(pr["max_hand_m"])}               # drop water higher than this above the channel (m)
  resolvable_px: {_v(pr["resolvable_px"])}             # reaches narrower than this many pixels are flagged
  keep_diagnostics: {_v(pr["keep_diagnostics"])}         # write per-pixel decision rasters (masks/diagnostics/)
"""


RIVERS_TEMPLATE = """\
river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id,buffer_width,notes
Beni,sa,62266500601,62266100011,,example row - replace with your rivers
"""
