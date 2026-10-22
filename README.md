# RivMapper v2

**Next-generation channel mobility, sediment storage, and transit time analysis for alluvial rivers.**

RivMapper v2 is a ground-up rewrite of [RivMapper (v1)](https://github.com/huck-rees/RivMapper), designed around new input datasets (SWORD v17b in place of HydroRIVERS and GRWL) and a config-driven workflow that scales to long lists of rivers.

> **Note:** RivMapper v1 lives at <https://github.com/huck-rees/RivMapper> and is the version associated with the in-review manuscript. This repository (v2) is under active development and not yet ready for production use.

## Setup

```bash
git clone https://github.com/huck-rees/RivMapper_v2.git
cd RivMapper_v2
conda env create -f environment.yml
conda activate rivmapper_v2
rivmapper data fetch sword        # optional: download SWORD now (~2 GB) instead of on first run
```

`environment.yml` installs the package in editable mode, which also provides the `rivmapper` command. In an existing environment, run `pip install -e ".[dev]"` from the repo root.

## Folder layout

Everything lives inside the repo folder. `data/` and `projects/` are ignored by git, so nothing large is ever committed:

```
RivMapper_v2/
├── rivmapper_v2/          code
├── data/                  shared external datasets, downloaded once
│   └── SWORD/             all continents (rivmapper data fetch sword)
└── projects/              your projects
    └── oktedi/
        ├── rivmapper.yaml
        ├── rivers.csv
        └── output/        everything this project produces
```

## External datasets

External datasets are listed once in [`rivmapper_v2/datasets.yaml`](rivmapper_v2/datasets.yaml), with each one's URL, version, checksum, expected files, licence and citation.

```bash
rivmapper data list               # what's registered, what's installed, where
rivmapper data fetch sword        # download, check checksum, unpack, record a manifest
rivmapper data verify sword       # check files on disk still match the manifest
rivmapper data cite               # citations for the data availability statement
rivmapper data path               # where datasets are stored
```

Pipeline steps fetch a missing dataset automatically, so `fetch` is only needed to download up front.

Datasets are stored in the first of these that applies:
1. A project's `sword: dir:` setting.
2. The `RIVMAPPER_DATA_DIR` environment variable, for example one shared copy on a lab server.
3. `data/` inside the repo, when running from a git checkout.
4. The user cache folder, for pip installs.

## How a project is set up

A project is two files in its own folder, typically `projects/<name>/` (create them with `rivmapper init projects/<name>`):

| File | What goes in it |
|---|---|
| `rivmapper.yaml` | Settings shared by every river: `data_root` (where this project's outputs go, default `./output`), SWORD version, GEE project, and default parameters for each step |
| `rivers.csv` | One row per river: `river_name`, `sword_cont_abb`, `us_sword_reach_id`, `ds_sword_reach_id`, plus optional per-river overrides |

Any parameter in the `reaches:` or `masks:` sections of the YAML can also be a column in `rivers.csv`. A filled cell overrides the default for that river, and a blank cell uses the YAML value. Columns named `notes`, or starting with `_`, are kept for your own reference and ignored by the pipeline. Any other unrecognised column is reported as an error, so a typo can't silently fall back to a default.

```csv
river_name,sword_cont_abb,us_sword_reach_id,ds_sword_reach_id,buffer_width,notes
Beni,sa,62266500601,62266100011,,uses the YAML default
Bermejo,sa,<us_id>,<ds_id>,1.0,overrides buffer_width for this river only
```

See [`examples/beni/`](examples/beni/) for a complete example.

## Running

```bash
rivmapper init projects/oktedi              # write starter rivmapper.yaml + rivers.csv
rivmapper validate projects/oktedi/rivmapper.yaml
rivmapper run projects/oktedi/rivmapper.yaml                  # all steps, all rivers
rivmapper run projects/oktedi/rivmapper.yaml --steps sword    # one step
rivmapper run projects/oktedi/rivmapper.yaml --rivers OkTedi --overwrite
```

`validate` checks every row before anything runs: continent codes, 11-digit reach IDs (including IDs that Excel has mangled into scientific notation), year and reach ranges, duplicate names, and unknown columns or settings. It reports every problem at once.

From Python or a notebook:

```python
from rivmapper_v2 import Project, run_pipeline, setup_logging

setup_logging()
project = Project.load("projects/oktedi/rivmapper.yaml")
results = run_pipeline(project, steps=["sword", "reaches"], rivers=["Beni"])
```

### Re-runs and provenance

Each step writes `{river}_{step}_run.yaml` next to its outputs. The file records the parameters used, the package version and git commit, input and output paths, and diagnostics. On later runs:

- **skipped**: outputs exist and were made with the current settings.
- **stale**: outputs exist but a relevant setting has changed. They are not rebuilt unless you pass `--overwrite`.
- **blocked**: an earlier step failed for that river, so later steps are not attempted.

### Migrating an old master CSV

```bash
rivmapper migrate E:/Dissertation/Data/Template_RivMapper_master.csv --out E:/Dissertation/Data
```

If the old working directory already holds a `SWORD/` folder, the new config keeps using it via `sword: dir:`. Migration also turns columns that are the same for every river into YAML defaults, keeps columns that vary as per-river overrides, renames old columns (`reach_length_modulator` → `length_modulator`, `DSWE_threshold_level` → `dswe_level`, `working_directory` → `data_root`, …), and drops pipeline-derived columns.

## Pipeline steps

| Step | Module | Output (under `data_root`) |
|---|---|---|
| `sword` | `rivmapper_v2/sword.py` | `SWORD/Extracted_Rivers/{river}/`: traced SWORD reaches and nodes |
| `reaches` | `rivmapper_v2/reaches.py` | `RiverMapping/Reaches/{river}/`: reach lines and polygons |
| masks | `notebooks/03_create_masks.ipynb` | Not yet ported; still reads the old master CSV |

## Repository structure

```
RivMapper_v2/
├── rivmapper_v2/      # Package: config, datasets, pipeline steps, CLI
├── notebooks/         # Notebooks that run the package and show QC plots (03 is still a prototype)
├── examples/          # Example project configs
├── tests/             # pytest suite
├── scripts/           # Standalone analysis and utility scripts
├── docs/              # Documentation
├── data/              # (ignored) shared datasets
├── projects/          # (ignored) your projects
├── pyproject.toml
└── environment.yml
```

## Testing

```bash
pytest                                       # synthetic-data tests
RIVMAPPER_TEST_SWORD_DIR=path/to/SWORD pytest   # also runs the real Beni end-to-end test
```

## License

MIT — see [LICENSE](LICENSE).
