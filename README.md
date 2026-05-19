# RivMapper v2

**Next-generation channel mobility, sediment storage, and transit time analysis for alluvial rivers.**

RivMapper v2 is a ground-up rewrite of [RivMapper (v1)](https://github.com/huck-rees/RivMapper), designed around new input datasets and a more streamlined, user-friendly workflow. While v1 remains the version tied to the currently in-review manuscript, v2 will eventually produce comparable outputs through a modernized pipeline.

> **Note:** RivMapper v1 lives at <https://github.com/huck-rees/RivMapper> and is the version associated with the in-review manuscript. This repository (v2) is under active development and not yet ready for production use.

## Installation

```bash
conda env create -f environment.yml
conda activate rivmapper_v2
pip install -e ".[dev]"
```

## Project Structure

```
RivMapper_v2/
├── rivmapper_v2/      # Main package — pipeline modules added here
├── scripts/           # Standalone analysis and utility scripts
├── tests/             # Test suite
├── docs/              # Documentation
├── shapefile_data/    # Input shapefiles and geospatial data
├── .github/workflows/ # CI/CD
├── pyproject.toml
├── environment.yml
└── ...
```

## License

MIT — see [LICENSE](LICENSE).
