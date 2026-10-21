# Notebooks

Each notebook runs one pipeline step from the `rivmapper_v2` package and adds
QC plots. Settings come from your project's `rivmapper.yaml` and rivers
table. Set `CONFIG_PATH` in the first code cell.

| Notebook | Step | Status |
|---|---|---|
| `01_sword_extraction.ipynb` | `sword` | Runs the package |
| `02_create_reaches.ipynb` | `reaches` | Runs the package |
| `03_create_masks.ipynb` | masks | Prototype; still reads the old master CSV until it is ported |

Code is prototyped in notebooks first. Once its logic is stable, it moves
into a module under `rivmapper_v2/`, and the notebook is cut down to running
that module and showing QC plots.
