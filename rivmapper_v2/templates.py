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

# This project's outputs go under this folder (extracted SWORD reaches,
# RiverMapping/, run records, ...).
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

gee:
  project: {_v(gee_project)}

# Step 02: reach creation. reach_len_m = length_modulator * len_slope * facc + len_intercept
reaches:
  length_modulator: {_v(r["length_modulator"])}
  buffer_width: {_v(r["buffer_width"])}        # buffer half-width as a fraction of reach length (required)
  len_slope: {_v(r["len_slope"])}            # v1 regression (Ganges, HydroRIVERS UPLAND_SKM)
  len_intercept: {_v(r["len_intercept"])}
  ds_trim_fraction: {_v(r["ds_trim_fraction"])}       # drop a last reach shorter than this × the one above it
  gap_tolerance_m: {_v(r["gap_tolerance_m"])}       # warn on gaps between SWORD reaches larger than this
  smoothing_window_multiplier: {_v(r["smoothing_window_multiplier"])}  # valley-direction window = this × reach length
  n_tangent_samples: {_v(r["n_tangent_samples"])}
  cut_extent_factor: {_v(r["cut_extent_factor"])}      # cut-line half-length = this × largest buffer
  # How neighbouring polygons are separated: nearest_centerline (each point goes
  # to the closest reach centreline; never overlaps) or cut_lines (perpendicular
  # cuts at each boundary; can overlap where the channel loops back).
  boundary_method: {_v(r["boundary_method"])}
  partition_spacing_m: {_v(r["partition_spacing_m"])}   # centreline sample spacing for nearest_centerline

# Step 03: water masks (not yet ported to the package; validated now so the
# rivers table can already carry these columns).
masks:
  year_start: {_v(m["year_start"])}
  year_end: {_v(m["year_end"])}
  reach_start: {_v(m["reach_start"])}         # null = first reach
  reach_end: {_v(m["reach_end"])}           # null = last reach
  dswe_level: {_v(m["dswe_level"])}          # DSWE class threshold, 1-4
  percentile: {_v(m["percentile"])}
  min_pixels: {_v(m["min_pixels"])}
  tile_size_km2: {_v(m["tile_size_km2"])}
"""


RIVERS_TEMPLATE = """\
river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id,buffer_width,notes
Beni,sa,62266500601,62266100011,,example row - replace with your rivers
"""
