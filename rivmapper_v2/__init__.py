"""RivMapper v2 — channel mobility, sediment storage, and transit time analysis.

Typical use from Python or a notebook::

    from rivmapper_v2 import Project, run_pipeline, setup_logging

    setup_logging()
    project = Project.load("projects/oktedi/rivmapper.yaml")
    run_pipeline(project, steps=["sword", "reaches"], rivers=["OkTedi"])

or from a terminal::

    rivmapper run projects/oktedi/rivmapper.yaml --rivers OkTedi
"""

__version__ = "0.3.0"

# Exports load on first use, so `import rivmapper_v2` (and the `rivmapper`
# command's --version/--help) stays fast and does not import geopandas.
_EXPORTS = {
    "ConfigError": ".config",
    "Project": ".config",
    "River": ".config",
    "STEPS": ".pipeline",
    "run_pipeline": ".pipeline",
    "setup_logging": ".cli",
}

__all__ = [*_EXPORTS, "__version__"]


def __getattr__(name: str):
    if name in _EXPORTS:
        from importlib import import_module

        value = getattr(import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
