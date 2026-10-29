"""Project configuration for RivMapper v2.

A RivMapper project is defined by two files:

``rivmapper.yaml``
    Settings shared by every river: where data lives, the SWORD version,
    the GEE project, and default parameters for each pipeline step.

``rivers.csv``
    One row per river. Required columns identify the river and its SWORD
    endpoints. Any parameter from the ``reaches`` or ``masks`` sections
    can also appear as a column; a filled cell overrides the YAML default
    for that river, and a blank cell falls back to it.

``Project.load()`` reads both, validates every value up front, and
returns one fully resolved ``River`` per row, so pipeline steps never
look anything up in the raw files.

Example:
    >>> project = Project.load("projects/oktedi/rivmapper.yaml")
    >>> river = project.river("OkTedi")
    >>> river.reaches.buffer_width
    0.5
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .paths import RiverPaths

log = logging.getLogger(__name__)

CONTINENTS = ("af", "as", "eu", "na", "oc", "sa")

# Characters that are illegal in Windows file names; river_name becomes
# part of every output path.
_BAD_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class ConfigError(ValueError):
    """Raised when the YAML config or rivers table is invalid.

    Collects every problem found, so the user can fix them in one pass.
    """

    def __init__(self, problems: list[str], source: Path | None = None):
        self.problems = problems
        where = f" in {source}" if source else ""
        lines = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"{len(problems)} problem(s){where}:\n{lines}")


# ─────────────────────────────────────────────────────────────────────
# Settings sections
# ─────────────────────────────────────────────────────────────────────
class _Section(BaseModel):
    """Base for YAML sections: unknown keys are errors, not silently ignored."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SwordSettings(_Section):
    """Where SWORD lives and which reaches the tracer may walk through."""

    version: str = "v17b"
    dir: Path | None = Field(
        None,
        description="Folder holding the continental SWORD geopackages. "
        "Defaults to the shared datasets folder (see rivmapper_v2.data).",
    )
    river_types: list[int] = Field(
        default_factory=lambda: [1],
        description="SWORD `type` codes kept for tracing (1 = river).",
    )


class GeeSettings(_Section):
    """Google Earth Engine settings."""

    project: str | None = Field(
        None, description="Google Cloud project registered for Earth Engine."
    )
    service_account: str | None = Field(
        None, description="Service-account email (headless servers); else user login."
    )
    key_file: Path | None = Field(None, description="JSON key for service_account.")


class ReachSettings(_Section):
    """Parameters for step 02 (create reaches). All can be set per river."""

    analysis_crs: str = Field(
        "auto",
        description="Projected CRS for all measured products: 'auto' (UTM, or "
        "LAEA for rivers spanning > 3 deg from the UTM central meridian) or "
        "any CRS string, e.g. 'EPSG:32754'.",
    )

    length_method: Literal["valley_widths", "drainage_regression"] = Field(
        "valley_widths",
        description="valley_widths: reach valley length = widths_per_reach × W(A); "
        "drainage_regression: v1 channel-length regression on drainage area.",
    )
    widths_per_reach: float = Field(
        100.0, gt=0, description="k: reach valley length in channel widths."
    )
    valley_sigma_widths: float = Field(
        15.0, gt=0, description="Base valley-line smoothing scale (channel widths)."
    )
    valley_sinuosity_factor: float = Field(
        6.0,
        ge=0,
        description="Smoothing scale in sinuous reaches = this × local sinuosity "
        "(widths), if larger than valley_sigma_widths.",
    )
    final_merge_fraction: float = Field(
        0.5,
        ge=0,
        le=1,
        description="valley_widths: merge a last reach shorter than this × its "
        "nominal length into the one above.",
    )
    length_modulator: float = Field(1.0, gt=0)
    buffer_width: float | None = Field(
        None, gt=0, description="Buffer half-width as a fraction of reach length."
    )
    len_slope: float = Field(0.0668, gt=0)
    len_intercept: float = 14316.0
    ds_trim_fraction: float = Field(0.8, ge=0, le=1)
    gap_tolerance_m: float = Field(10.0, ge=0)
    smoothing_window_multiplier: float = Field(1.5, gt=0)
    n_tangent_samples: int = Field(21, ge=3)
    cut_extent_factor: float = Field(3.0, gt=0)
    boundary_method: Literal["nearest_valley", "nearest_centerline", "cut_lines"] = (
        Field(
            "nearest_valley",
            description="How neighbouring reach polygons are separated: "
            "nearest_valley (closest position along the valley line; boundaries "
            "perpendicular to the valley), nearest_centerline (closest reach "
            "channel centreline) or cut_lines (perpendicular cuts at each boundary).",
        )
    )
    partition_spacing_m: float = Field(
        100.0,
        gt=0,
        description="Sample spacing along centrelines for nearest_centerline (m).",
    )


class MaskSettings(_Section):
    """Parameters for step 03 (water masks). All can be set per river."""

    year_start: int | None = Field(None, ge=1982)
    year_end: int | None = Field(None, ge=1982)
    reach_start: int | None = Field(None, ge=1)
    reach_end: int | None = Field(None, ge=1)
    dswe_level: int | None = Field(None, ge=1, le=4)
    percentile: float = Field(50.0, ge=0, le=100)
    chunk_px: int = Field(
        2048,
        ge=256,
        le=8192,
        description="Download chunk size (pixels per side) for Earth Engine.",
    )
    min_chunk_px: int = Field(
        256,
        ge=32,
        le=8192,
        description=(
            "Smallest tile (pixels per side) a chunk is split down to when "
            "Earth Engine runs out of memory or time on it."
        ),
    )
    min_pixels: int = Field(200, ge=0)
    # ── scene screening (raw product v2) ──
    screen_scenes: bool = Field(
        True, description="Drop scenes with widespread false water before counting."
    )
    max_false_water2: float = Field(
        0.05,
        gt=0,
        le=1,
        description=(
            "Drop a scene when more than this share of its clear reference-dry "
            "pixels is DSWE 1-2 (the blocky haze/cloud artefacts)."
        ),
    )
    max_false_water3: float = Field(
        0.20,
        gt=0,
        le=1,
        description=(
            "... or more than this share is DSWE 1-3 (class-3-only artefacts; "
            "set well above the persistent DSWE 3 of wet ground)."
        ),
    )
    ref_dry_widths: float = Field(
        10.0,
        gt=0,
        description=(
            "Reference-dry pixels are at least this many channel widths from "
            "the reach line (and never water in JRC Global Surface Water)."
        ),
    )
    min_ref_px: int = Field(
        500,
        ge=1,
        description="Clear reference pixels needed to judge a scene by the rates above.",
    )
    gross_false_water2: float = Field(
        0.25,
        gt=0,
        le=1,
        description=(
            "Scenes with fewer reference pixels are still dropped above this "
            "DSWE 1-2 share (gross artefacts) ..."
        ),
    )
    gross_min_ref_px: int = Field(
        100, ge=1, description="... if they have at least this many reference pixels."
    )
    screen_scale_m: float = Field(60.0, ge=30, description="Scale for screening stats.")
    le07_gap_years: list[int] = Field(
        default_factory=lambda: [2012],
        description=(
            "Years in which Landsat 7 SLC-off scenes are also used (too few "
            "other scenes). Landsat 7 is otherwise used only to May 2003."
        ),
    )
    workers: int = Field(
        2,
        ge=1,
        le=16,
        description="Reach-years requested from Earth Engine in parallel.",
    )
    screen_only: bool = Field(
        False,
        description=(
            "Write the scene manifest only (no downloads): a quick look at "
            "screening before a full extraction."
        ),
    )

    @field_validator("le07_gap_years", mode="before")
    @classmethod
    def _years_from_text(cls, v):
        # rivers.csv cells arrive as text, e.g. "2012" or "2012;2013"
        if isinstance(v, str):
            parts = [x for x in v.replace(",", ";").split(";") if x.strip()]
            return [int(float(x)) for x in parts]
        if isinstance(v, int | float):
            return [int(v)]
        return v

    @model_validator(mode="after")
    def _ranges_ordered(self) -> MaskSettings:
        if self.min_chunk_px > self.chunk_px:
            raise ValueError(
                f"masks.min_chunk_px ({self.min_chunk_px}) must not exceed "
                f"masks.chunk_px ({self.chunk_px})"
            )
        if self.year_start and self.year_end and self.year_start > self.year_end:
            raise ValueError(
                f"year_start ({self.year_start}) is after year_end ({self.year_end})"
            )
        if self.reach_start and self.reach_end and self.reach_start > self.reach_end:
            raise ValueError(
                f"reach_start ({self.reach_start}) is after reach_end "
                f"({self.reach_end})"
            )
        return self


class ProcessSettings(_Section):
    """Parameters for step ``process_masks`` (counts → channel masks).

    Spatial settings are multiples of the reach's channel width W
    (``width_ref`` in the reach polygons), so defaults carry across rivers.
    See docs/methods/processing.md. All can be set per river.
    """

    states: int = Field(
        3,
        ge=2,
        le=3,
        description="HMM states: 3 = dry/margin/channel, 2 = dry/channel.",
    )
    fit_sample_px: int = Field(
        50_000, ge=1000, description="Pixels sampled to fit the HMM per river."
    )
    envelope_threshold: float | Literal["calibrate"] = Field(
        "calibrate",
        description="Threshold on P(channel)+P(margin), or 'calibrate' to match "
        "bankfull width.",
    )
    core_threshold: float = Field(0.5, gt=0, lt=1)
    max_infer_gap: int = Field(
        2, ge=0, description="Years an unobserved year may sit from observed ones."
    )
    use_le07_gap_years: bool = True
    bridge_closing_widths: float = Field(0.5, ge=0)
    bridge_max_widths: float = Field(5.0, ge=0)
    grow_widths: float = Field(1.0, ge=0)
    max_inscribed_widths: float = Field(1.25, gt=0)
    dead_end_widths: float = Field(3.0, ge=0)
    dead_end_widths_braided: float = Field(10.0, ge=0)
    network_selection: bool = Field(
        True,
        description="Phase 3 link selection: keep the main path and channel-like "
        "loops, drop floodplain water attached to the channel (else the phase 1 "
        "dead-end pruning).",
    )
    thread_min_widths: float = Field(
        0.3, ge=0, description="Second threads at least this × W wide (median)."
    )
    thread_max_widths: float = Field(
        1.6, gt=0, description="... and at most this × W wide (median)."
    )
    thread_max_p90_widths: float = Field(
        2.5, gt=0, description="... with no part (90th percentile) wider than this × W."
    )
    thread_min_core: float = Field(
        0.5, ge=0, le=1, description="... on the low-flow core for at least this share."
    )
    thread_max_width_cv: float = Field(
        0.6, gt=0, description="... and width varying by at most this (CV)."
    )
    link_margin_widths: float = Field(
        0.5,
        ge=0,
        description="Pixels kept up to a link's half-width plus this × W from it.",
    )
    min_component_w2: float = Field(4.0, ge=0)
    fill_hole_w2: float = Field(0.25, ge=0)
    outlier_mad: float = Field(3.0, gt=0)
    use_channel_belt: bool = False
    max_hand_m: float | None = Field(None, gt=0)
    resolvable_px: float = Field(
        3.0, gt=0, description="Reaches narrower than this many pixels are flagged."
    )
    keep_diagnostics: bool = Field(
        True,
        description="Write per-pixel decision rasters and window context files.",
    )

    @model_validator(mode="after")
    def _threshold_range(self) -> ProcessSettings:
        t = self.envelope_threshold
        if t != "calibrate" and not 0 < t < 1:
            raise ValueError(
                f"envelope_threshold must be between 0 and 1 or 'calibrate' (got {t})"
            )
        return self


GLOFAS_VERSIONS = ("4.0", "5.0")


class DischargeSettings(_Section):
    """Step ``discharge``: modelled discharge on scene dates (stage flag)."""

    source: Literal["glofas", "geoglows_v2"] = Field(
        "glofas",
        description=(
            "glofas: GloFAS historical reanalysis (CEMS EWDS; needs an account). "
            "geoglows_v2: GEOGLOWS v2 retrospective (AWS, anonymous)."
        ),
    )
    glofas_version: str = Field(
        "4.0", description=f"GloFAS reanalysis version: one of {GLOFAS_VERSIONS}."
    )
    glofas_product_type: Literal["consolidated", "intermediate"] = Field(
        "consolidated",
        description=(
            "consolidated (ERA5, updated monthly) or intermediate (ERA5T, recent "
            "months only)."
        ),
    )
    glofas_workers: int = Field(
        6,
        ge=1,
        le=16,
        description=(
            "GloFAS years requested from EWDS at once (they queue and run in "
            "parallel on the server; EWDS caps how many of yours run together)."
        ),
    )
    glofas_dir: Path | None = Field(
        None,
        description=(
            "Folder of GloFAS NetCDF files already on disk (e.g. a lab archive). "
            "Read first; only what they lack is downloaded."
        ),
    )
    search_km: float = Field(
        5.0,
        gt=0,
        description="Search radius for model rivers/cells around a reach end.",
    )
    area_tolerance: float = Field(
        2.0,
        gt=1,
        description=(
            "A model river or cell matches a reach when their drainage areas "
            "differ by less than this factor."
        ),
    )
    low_stage_pctile: float = Field(
        20.0,
        ge=0,
        le=100,
        description="Scene-date discharge below this percentile → low_stage.",
    )
    high_stage_pctile: float = Field(
        80.0,
        ge=0,
        le=100,
        description="Scene-date discharge above this percentile → high_stage.",
    )
    manual_lon: float | None = Field(
        None,
        ge=-180,
        le=180,
        description=(
            "Manual override (both sources): use the model river at this point "
            "for every reach (GloFAS: the grid cell containing it; GEOGLOWS: "
            "the link outlet nearest to it). Set with manual_lat."
        ),
    )
    manual_lat: float | None = Field(None, ge=-90, le=90)
    river_id: int | None = Field(
        None,
        description="GEOGLOWS only: use this river id (LINKNO) for every reach.",
    )

    @field_validator("glofas_version", mode="before")
    @classmethod
    def _norm_version(cls, v: Any) -> str:
        """Accept 4, 4.0, "4", "v4.0" or "version_4_0" for "4.0"."""
        text = str(v).strip().lower().removeprefix("version_").removeprefix("v")
        text = text.replace("_", ".")
        try:
            text = f"{float(text):.1f}"
        except ValueError:
            pass
        if text not in GLOFAS_VERSIONS:
            raise ValueError(
                f"glofas_version must be one of {list(GLOFAS_VERSIONS)} (got {v!r})"
            )
        return text

    @model_validator(mode="after")
    def _manual_override(self) -> DischargeSettings:
        if (self.manual_lon is None) != (self.manual_lat is None):
            raise ValueError("set both manual_lon and manual_lat, or neither")
        if self.river_id is not None:
            if self.source != "geoglows_v2":
                raise ValueError(
                    "river_id is a GEOGLOWS link id; for GloFAS use "
                    "manual_lon/manual_lat"
                )
            if self.manual_lon is not None:
                raise ValueError("set river_id or manual_lon/manual_lat, not both")
        return self

    @property
    def glofas_system_version(self) -> str:
        """EWDS request value, e.g. ``version_4_0``."""
        return "version_" + self.glofas_version.replace(".", "_")


_SECTION_KEYS = ("sword", "gee", "reaches", "masks", "process", "discharge")

# Sections whose fields may be overridden per river in the rivers table.
OVERRIDABLE_SECTIONS: dict[str, type[_Section]] = {
    "reaches": ReachSettings,
    "masks": MaskSettings,
    "process": ProcessSettings,
    "discharge": DischargeSettings,
}
OVERRIDE_COLUMNS: dict[str, str] = {
    field: section
    for section, model in OVERRIDABLE_SECTIONS.items()
    for field in model.model_fields
}

# Old master-CSV column names → new names. Accepted with a warning.
LEGACY_COLUMN_ALIASES: dict[str, str] = {
    "working_directory": "data_root",
    "reach_length_modulator": "length_modulator",
    "DSWE_threshold_level": "dswe_level",
    "mask_percentile": "percentile",
    "continent_abbrev": "sword_cont_abb",
    "us_reach_id": "us_sword_reach_id",
    "ds_reach_id": "ds_sword_reach_id",
}

# Settings from earlier versions that are accepted with a warning and ignored.
RETIRED_SETTINGS: dict[tuple[str, str], str] = {
    ("masks", "tile_size_km2"): "downloads are now chunked by masks.chunk_px",
    ("sword", "download_url"): "SWORD download details live in datasets.yaml",
    ("sword", "md5"): "SWORD download details live in datasets.yaml",
    ("masks", "composite_windows"): "raw masks are single-year DSWE counts",
    ("masks", "min_clear_fraction"): "raw masks are single-year DSWE counts",
    ("masks", "min_water_fraction"): "raw masks are single-year DSWE counts",
    ("masks", "max_false_water"): (
        "replaced by max_false_water2 (DSWE 1-2) and max_false_water3 (DSWE 1-3)"
    ),
}

RIVER_ID_COLUMNS = (
    "river_name",
    "sword_cont_abb",
    "us_sword_reach_id",
    "ds_sword_reach_id",
)
# Columns users may keep for their own bookkeeping; ignored by the pipeline.
NOTE_COLUMNS = ("notes",)


class ProjectSettings(_Section):
    """Top level of ``rivmapper.yaml``."""

    @model_validator(mode="before")
    @classmethod
    def _empty_sections_use_defaults(cls, data):
        # A section header with nothing (or only comments) under it, e.g.
        # "process:", reads as None in YAML: treat it as "all defaults".
        if isinstance(data, dict):
            data = {
                k: ({} if v is None and k in _SECTION_KEYS else v)
                for k, v in data.items()
            }
        return data

    data_root: Path
    rivers_table: Path = Path("rivers.csv")
    overwrite: bool = False
    sword: SwordSettings = Field(default_factory=SwordSettings)
    gee: GeeSettings = Field(default_factory=GeeSettings)
    reaches: ReachSettings = Field(default_factory=ReachSettings)
    masks: MaskSettings = Field(default_factory=MaskSettings)
    process: ProcessSettings = Field(default_factory=ProcessSettings)
    discharge: DischargeSettings = Field(default_factory=DischargeSettings)


# ─────────────────────────────────────────────────────────────────────
# Resolved, per-river view
# ─────────────────────────────────────────────────────────────────────
class River(BaseModel):
    """One river with every setting resolved (YAML defaults + row overrides)."""

    model_config = ConfigDict(frozen=True)

    name: str
    continent: str
    us_reach_id: int
    ds_reach_id: int
    data_root: Path
    sword: SwordSettings
    gee: GeeSettings
    reaches: ReachSettings
    masks: MaskSettings
    process: ProcessSettings = Field(default_factory=ProcessSettings)
    discharge: DischargeSettings = Field(default_factory=DischargeSettings)
    overwrite: bool = False
    row_number: int | None = Field(
        None, description="Spreadsheet row in rivers.csv (header = row 1)."
    )

    @property
    def paths(self) -> RiverPaths:
        """Standard input/output locations for this river."""
        return RiverPaths(self.data_root, self.name)

    @property
    def sword_dir(self) -> Path:
        """Folder holding the continental SWORD geopackages.

        ``sword: dir:`` if set, otherwise the shared datasets folder
        (``<repo>/data/SWORD`` in a git checkout; see ``rivmapper_v2.data``).
        """
        if self.sword.dir:
            return self.sword.dir
        from .data import dataset_dir, get_dataset

        return dataset_dir(get_dataset("sword", self.sword.version))

    def require(self, section: str, *fields: str) -> None:
        """Raise ConfigError if any of ``fields`` in ``section`` is unset.

        Steps call this so a missing parameter fails with a clear message
        naming the river and the YAML key / CSV column to fill.
        """
        settings = getattr(self, section)
        missing = [f for f in fields if getattr(settings, f) is None]
        if missing:
            raise ConfigError(
                [
                    f"{self.name}: '{f}' is not set — add `{section}: {f}:` "
                    f"to rivmapper.yaml or a `{f}` column to the rivers table"
                    for f in missing
                ]
            )


# ─────────────────────────────────────────────────────────────────────
# Project loading
# ─────────────────────────────────────────────────────────────────────
class Project:
    """A loaded, validated RivMapper project.

    Attributes:
        config_path: The YAML file this project was loaded from.
        settings: Parsed YAML (project-wide defaults).
        rivers: One resolved ``River`` per row of the rivers table.
    """

    def __init__(
        self, config_path: Path, settings: ProjectSettings, rivers: list[River]
    ):
        self.config_path = config_path
        self.settings = settings
        self.rivers = rivers

    # ── construction ────────────────────────────────────────────────
    @classmethod
    def load(cls, config_path: str | Path) -> Project:
        """Load and validate ``rivmapper.yaml`` and its rivers table.

        Relative paths in the YAML (``data_root``, ``rivers_table``,
        ``sword.dir``, ``discharge.glofas_dir``) are resolved against the YAML
        file's folder; a relative ``glofas_dir`` in the rivers table against
        the table's folder.

        Raises:
            FileNotFoundError: If the YAML or rivers table does not exist.
            ConfigError: Listing every validation problem found.
        """
        config_path = Path(config_path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        settings = load_settings(config_path)

        table_path = settings.rivers_table
        if not table_path.exists():
            raise FileNotFoundError(
                f"Rivers table not found: {table_path} "
                f"(set `rivers_table:` in {config_path.name})"
            )
        table = read_rivers_table(table_path)
        rivers = resolve_rivers(settings, table, source=table_path)
        return cls(config_path, settings, rivers)

    # ── access ──────────────────────────────────────────────────────
    @property
    def river_names(self) -> list[str]:
        return [r.name for r in self.rivers]

    def river(self, name: str) -> River:
        """Return one river by name (case-sensitive)."""
        for r in self.rivers:
            if r.name == name:
                return r
        raise KeyError(f"No river named {name!r}. Available: {self.river_names}")

    def select(self, names: Iterable[str] | None) -> list[River]:
        """Return rivers in table order, optionally restricted to ``names``."""
        if not names:
            return list(self.rivers)
        names = list(names)
        unknown = [n for n in names if n not in self.river_names]
        if unknown:
            raise KeyError(
                f"Unknown river(s): {unknown}. Available: {self.river_names}"
            )
        wanted = set(names)
        return [r for r in self.rivers if r.name in wanted]

    def __repr__(self) -> str:
        return (
            f"Project({self.config_path.name}, {len(self.rivers)} rivers, "
            f"data_root={self.settings.data_root})"
        )


def load_settings(config_path: Path) -> ProjectSettings:
    """Parse and validate the YAML file, resolving relative paths."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError(["top level must be a mapping of settings"], config_path)
    for (section, key), note in RETIRED_SETTINGS.items():
        if isinstance(raw.get(section), dict) and key in raw[section]:
            raw[section].pop(key)
            log.warning(
                "%s: `%s: %s` is no longer used (%s); you can delete it.",
                config_path.name,
                section,
                key,
                note,
            )
    try:
        settings = ProjectSettings.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(_format_pydantic(e), config_path) from None

    base = config_path.parent
    settings.data_root = _resolve(settings.data_root, base)
    settings.rivers_table = _resolve(settings.rivers_table, base)
    if settings.sword.dir is not None:
        settings.sword.dir = _resolve(settings.sword.dir, base)
    if settings.gee.key_file is not None:
        settings.gee.key_file = _resolve(settings.gee.key_file, base)
    if settings.discharge.glofas_dir is not None:
        settings.discharge.glofas_dir = _resolve(settings.discharge.glofas_dir, base)
    return settings


def read_rivers_table(path: Path) -> pd.DataFrame:
    """Read the rivers CSV as strings, blank rows dropped, legacy names mapped."""
    table = pd.read_csv(path, dtype=str, skip_blank_lines=True)
    table.columns = [c.strip() for c in table.columns]
    table = table.dropna(how="all").reset_index(drop=True)

    renames = {}
    for old, new in LEGACY_COLUMN_ALIASES.items():
        if old in table.columns:
            if new in table.columns:
                raise ConfigError(
                    [f"both '{old}' (old name) and '{new}' columns present"], path
                )
            renames[old] = new
    if renames:
        log.warning(
            "Rivers table uses old column names; please rename: %s",
            ", ".join(f"{o} → {n}" for o, n in renames.items()),
        )
        table = table.rename(columns=renames)
    return table


def resolve_rivers(
    settings: ProjectSettings, table: pd.DataFrame, source: Path | None = None
) -> list[River]:
    """Merge YAML defaults with each table row and validate the result."""
    problems: list[str] = []

    missing = [c for c in RIVER_ID_COLUMNS if c not in table.columns]
    if missing:
        raise ConfigError([f"rivers table is missing column(s): {missing}"], source)

    known = set(RIVER_ID_COLUMNS) | set(OVERRIDE_COLUMNS) | {"data_root"}
    unknown = [
        c
        for c in table.columns
        if c not in known and c not in NOTE_COLUMNS and not c.startswith("_")
    ]
    if unknown:
        problems.append(
            f"unknown column(s) {unknown}. Per-river overrides must be one of "
            f"{sorted(OVERRIDE_COLUMNS)}; prefix a column with '_' or name it "
            f"'notes' to keep it for your own reference"
        )

    names = table["river_name"].str.strip()
    dupes = sorted(set(names[names.duplicated()].dropna()))
    if dupes:
        problems.append(f"duplicate river_name(s): {dupes}")

    rivers: list[River] = []
    for idx, row in table.iterrows():
        row_no = int(idx) + 2  # spreadsheet row; header is row 1
        cells = {k: _clean_cell(v) for k, v in row.items() if k in known}
        label = f"row {row_no} ({cells.get('river_name') or 'no name'})"
        try:
            rivers.append(_resolve_row(settings, cells, row_no))
        except ValidationError as e:
            problems.extend(f"{label}: {m}" for m in _format_pydantic(e))
        except ValueError as e:
            problems.append(f"{label}: {e}")

    if problems:
        raise ConfigError(problems, source)
    return rivers


def _resolve_row(
    settings: ProjectSettings, cells: dict[str, Any], row_no: int
) -> River:
    name = cells.get("river_name")
    if not name:
        raise ValueError("river_name is blank")
    if _BAD_NAME_CHARS.search(name):
        raise ValueError(
            f"river_name {name!r} contains characters not allowed in file names"
        )
    continent = (cells.get("sword_cont_abb") or "").lower()
    if continent not in CONTINENTS:
        raise ValueError(
            f"sword_cont_abb {cells.get('sword_cont_abb')!r} must be one of "
            f"{list(CONTINENTS)}"
        )
    us_id = _parse_reach_id(cells.get("us_sword_reach_id"), "us_sword_reach_id")
    ds_id = _parse_reach_id(cells.get("ds_sword_reach_id"), "ds_sword_reach_id")
    if us_id == ds_id:
        raise ValueError("us_sword_reach_id and ds_sword_reach_id are identical")

    data_root = settings.data_root
    if cells.get("data_root"):
        data_root = _resolve(Path(cells["data_root"]), settings.data_root)

    sections: dict[str, _Section] = {}
    for section, model in OVERRIDABLE_SECTIONS.items():
        base = getattr(settings, section).model_dump()
        overrides = {
            f: cells[f] for f in model.model_fields if cells.get(f) is not None
        }
        if section == "discharge" and overrides.get("glofas_dir"):
            overrides["glofas_dir"] = _resolve(
                Path(overrides["glofas_dir"]), settings.rivers_table.parent
            )
        sections[section] = model.model_validate({**base, **overrides})

    return River(
        name=name,
        continent=continent,
        us_reach_id=us_id,
        ds_reach_id=ds_id,
        data_root=data_root,
        sword=settings.sword,
        gee=settings.gee,
        overwrite=settings.overwrite,
        row_number=row_no,
        **sections,
    )


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
_INT_LIKE_FLOAT = re.compile(r"^-?\d+\.0*$")


def _clean_cell(value: Any) -> Any:
    """Blank/NaN → None; strip text; '2005.0' → '2005' (Excel artefact)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    if _INT_LIKE_FLOAT.match(text):
        text = text.split(".")[0]
    return text


def _parse_reach_id(value: Any, column: str) -> int:
    if value is None:
        raise ValueError(f"{column} is blank")
    text = str(value)
    if "e" in text.lower():
        raise ValueError(
            f"{column} {text!r} looks like scientific notation — Excel has "
            f"probably rounded the ID. Format the column as Number with 0 "
            f"decimals (or Text) and re-enter it"
        )
    if not text.isdigit() or len(text) != 11:
        raise ValueError(f"{column} {text!r} is not an 11-digit SWORD reach ID")
    return int(text)


def _resolve(path: Path, base: Path) -> Path:
    path = Path(str(path)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _format_pydantic(err: ValidationError) -> list[str]:
    out = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "value"
        msg = e["msg"].removeprefix("Value error, ")
        if e["type"] == "extra_forbidden":
            msg = "unknown setting (check spelling)"
        out.append(
            f"{loc}: {msg}"
            + (
                f" (got {e['input']!r})"
                if e["type"] not in ("extra_forbidden", "missing")
                and not isinstance(e.get("input"), dict)
                else ""
            )
        )
    return out
