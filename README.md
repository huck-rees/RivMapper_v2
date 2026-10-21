# RivMapper v2

**Next-generation channel mobility, sediment storage, and transit time analysis for alluvial rivers.**

RivMapper v2 is a ground-up rewrite of [RivMapper (v1)](https://github.com/huck-rees/RivMapper), designed around new input datasets (SWORD v17b in place of HydroRIVERS and GRWL) and a config-driven workflow that scales to long lists of rivers.

> **Note:** RivMapper v1 lives at <https://github.com/huck-rees/RivMapper> and is the version associated with the in-review manuscript. This repository (v2) is under active development and not yet ready for production use.

## Installation

```bash
conda env create -f environment.yml
conda activate rivmapper_v2
```

`environment.yml` installs the package in editable mode, which also provides the `rivmapper` command. In an existing environment, run `pip install -e ".[dev]"` from the repo root.

## How a project is set up

A project is two files, usually kept together in your data folder:

| File | What goes in it |
|---|---|
| `rivmapper.yaml` | Settings shared by every river: `data_root`, SWORD version and folder, GEE project, and default parameters for each step |
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
rivmapper init my_project/              # write starter rivmapper.yaml + rivers.csv
rivmapper validate my_project/rivmapper.yaml
rivmapper run my_project/rivmapper.yaml                       # all steps, all rivers
rivmapper run my_project/rivmapper.yaml --steps reaches --rivers Beni,Bermejo
rivmapper run my_project/rivmapper.yaml --overwrite           # rebuild existing outputs
```

`validate` checks every row before anything runs: continent codes, 11-digit reach IDs (including IDs that Excel has mangled into scientific notation), year and reach ranges, duplicate names, and unknown columns or settings. It reports every problem at once.

From Python or a notebook:

```python
from rivmapper_v2 import Project, run_pipeline, setup_logging

setup_logging()
project = Project.load("E:/Dissertation/Data/rivmapper.yaml")
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

This turns columns that are the same for every river into YAML defaults, keeps columns that vary as per-river overrides, renames old columns (`reach_length_modulator` → `length_modulator`, `DSWE_threshold_level` → `dswe_level`, `working_directory` → `data_root`, …), and drops pipeline-derived columns.

## Pipeline steps

| Step | Module | Output (under `data_root`) |
|---|---|---|
| `sword` | `rivmapper_v2/sword.py` | `SWORD/Extracted_Rivers/{river}/`: traced SWORD reaches and nodes |
| `reaches` | `rivmapper_v2/reaches.py` | `RiverMapping/Reaches/{river}/`: reach lines and polygons |
| masks | `notebooks/03_create_masks.ipynb` | Not yet ported; still reads the old master CSV |

## Project Structure

```
RivMapper_v2/
├── rivmapper_v2/      # Package: config, pipeline steps, CLI
├── notebooks/         # Notebooks that run the package and show QC plots (03 is still a prototype)
├── examples/          # Example project configs
├── tests/             # pytest suite
├── scripts/           # Standalone analysis and utility scripts
├── docs/              # Documentation
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
