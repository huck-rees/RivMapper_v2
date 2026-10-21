"""RivMapper v2 — channel mobility, sediment storage, and transit time analysis.

Typical use from Python or a notebook::

    from rivmapper_v2 import Project, run_pipeline, setup_logging

    setup_logging()
    project = Project.load("E:/Dissertation/Data/rivmapper.yaml")
    run_pipeline(project, steps=["sword", "reaches"], rivers=["Beni"])

or from a terminal::

    rivmapper run E:/Dissertation/Data/rivmapper.yaml --rivers Beni
"""

__version__ = "0.2.0"

from .cli import setup_logging  # noqa: E402
from .config import ConfigError, Project, River  # noqa: E402
from .pipeline import STEPS, run_pipeline  # noqa: E402

__all__ = [
    "ConfigError",
    "Project",
    "River",
    "STEPS",
    "run_pipeline",
    "setup_logging",
    "__version__",
]
