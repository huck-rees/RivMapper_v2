"""Command-line interface: ``rivmapper init | validate | run | migrate``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, Project
from .pipeline import STEPS, run_pipeline
from .templates import RIVERS_TEMPLATE, render_config


def setup_logging(verbose: bool = False) -> None:
    """Plain console logging; ``verbose`` adds debug messages."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        force=True,
    )
    for noisy in ("pyogrio", "fiona", "urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _split(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def cmd_init(args: argparse.Namespace) -> int:
    out = Path(args.directory)
    out.mkdir(parents=True, exist_ok=True)
    targets = {
        out / "rivmapper.yaml": render_config(),
        out / "rivers.csv": RIVERS_TEMPLATE,
    }
    for path, text in targets.items():
        if path.exists() and not args.force:
            print(f"{path} already exists (use --force to replace)", file=sys.stderr)
            return 1
    for path, text in targets.items():
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    print(f"Next: edit both files, then `rivmapper validate {out / 'rivmapper.yaml'}`")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    project = Project.load(args.config)
    s = project.settings
    print(f"Config:      {project.config_path}")
    print(f"Rivers:      {s.rivers_table} ({len(project.rivers)} rivers)")
    print(f"Data root:   {s.data_root}")
    print(f"SWORD:       {s.sword.version} in {s.sword.dir or s.data_root / 'SWORD'}")
    print()
    header = f"{'river':<24}{'cont':<6}{'us_reach_id':<14}{'ds_reach_id':<14}overrides"
    print(header)
    print("─" * len(header))
    for r in project.rivers:
        overrides = []
        for section in ("reaches", "masks"):
            base = getattr(s, section).model_dump()
            mine = getattr(r, section).model_dump()
            overrides += [f"{k}={mine[k]}" for k in mine if mine[k] != base[k]]
        if r.data_root != s.data_root:
            overrides.append(f"data_root={r.data_root}")
        print(
            f"{r.name:<24}{r.continent:<6}{r.us_reach_id:<14}{r.ds_reach_id:<14}"
            f"{', '.join(overrides) or '—'}"
        )

    # Settings a step will need but nobody has set.
    gaps = [r.name for r in project.rivers if r.reaches.buffer_width is None]
    if gaps:
        print(
            f"\nNote: reaches.buffer_width is unset for {len(gaps)} river(s) "
            f"({', '.join(gaps[:5])}{'…' if len(gaps) > 5 else ''}); "
            f"the reaches step will fail for them."
        )
    print("\nOK")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    project = Project.load(args.config)
    if args.overwrite:
        project.rivers = [
            r.model_copy(update={"overwrite": True}) for r in project.rivers
        ]
    table = run_pipeline(project, steps=_split(args.steps), rivers=_split(args.rivers))
    print()
    if table.empty:
        print("Nothing to do.")
        return 0
    print(table.to_string(index=False))
    counts = table["status"].value_counts().to_dict()
    print("\n" + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 1 if {"failed", "blocked", "stale"} & set(counts) else 0


def cmd_migrate(args: argparse.Namespace) -> int:
    from .migrate import migrate_master_csv

    report = migrate_master_csv(args.master_csv, args.out, force=args.force)
    print(report)
    print(
        f"\nNext: review {report.config_path.name} (set gee.project, sword.dir), "
        f"then `rivmapper validate {report.config_path}`"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rivmapper",
        description="RivMapper v2: map river reaches from SWORD and Landsat.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug output")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="write a starter rivmapper.yaml and rivers.csv")
    p.add_argument("directory", nargs="?", default=".")
    p.add_argument("--force", action="store_true", help="replace existing files")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser(
        "validate", help="check a config and rivers table without running"
    )
    p.add_argument("config", help="path to rivmapper.yaml")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("run", help="run pipeline steps")
    p.add_argument("config", help="path to rivmapper.yaml")
    p.add_argument(
        "--steps", help=f"comma-separated subset of: {','.join(STEPS)} (default: all)"
    )
    p.add_argument("--rivers", help="comma-separated river names (default: all)")
    p.add_argument(
        "--overwrite", action="store_true", help="rebuild outputs even if they exist"
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("migrate", help="convert an old master CSV to yaml + rivers.csv")
    p.add_argument("master_csv")
    p.add_argument("--out", default=".", help="folder to write the two files into")
    p.add_argument("--force", action="store_true", help="replace existing files")
    p.set_defaults(func=cmd_migrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        return args.func(args)
    except (ConfigError, FileNotFoundError, FileExistsError, KeyError, ValueError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"error: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
