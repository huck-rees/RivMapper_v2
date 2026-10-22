"""Starter ``rivmapper.yaml`` / ``rivers.csv`` text for ``rivmapper init``."""

from __future__ import annotations

from typing import Any

import yaml

from .config import MaskSettings, ReachSettings, SwordSettings


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
) -> str:
    """Return a commented rivmapper.yaml with the given values filled in."""
    r = {**ReachSettings().model_dump(), **(reaches or {})}
    m = {**MaskSettings().model_dump(), **(masks or {})}
    s = SwordSettings()
    return f"""\
# RivMapper v2 project configuration
# ────────────────────────────────────────────────────────────────────
# Settings here apply to every river in the rivers table. Any setting in
# the `reaches:` or `masks:` sections can be overridden for one river by
# adding a column with the same name to the rivers table; blank cells use
# the value below. Relative paths are relative to this file.
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

# Step 03: raw water masks from Earth Engine (see docs/methods/conventions.md).
masks:
  year_start: {_v(m["year_start"])}         # required
  year_end: {_v(m["year_end"])}           # required
  reach_start: {_v(m["reach_start"])}         # null = first reach
  reach_end: {_v(m["reach_end"])}           # null = last reach
  dswe_level: {_v(m["dswe_level"])}          # required: water = DSWE class 1..dswe_level (1-4)
  percentile: {_v(m["percentile"])}          # per-pixel percentile of per-scene water (50 = median)
  composite_windows: {_v(m["composite_windows"])}       # [0] = each year on its own (no widening); [0, 1, 2] widens on failed checks
  min_clear_fraction: {_v(m["min_clear_fraction"])}  # check: share of reach polygon with >= 1 clear observation
  min_water_fraction: {_v(m["min_water_fraction"])}  # check: detected / expected (width_ref x reach_len) water area
  chunk_px: {_v(m["chunk_px"])}           # download chunk size (pixels per side)
  min_pixels: {_v(m["min_pixels"])}          # used later by mask processing
"""


RIVERS_TEMPLATE = """\
river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id,buffer_width,notes
Beni,sa,62266500601,62266100011,,example row - replace with your rivers
"""
