"""External datasets: registry, download, verification and citation.

Every external dataset is described once in ``datasets.yaml`` (URL,
version, checksum, expected files, citation). This module downloads and
unpacks them into a *datasets directory*, records a manifest, and is
used both by ``rivmapper data ...`` and by pipeline steps, which fetch a
missing dataset automatically on first use.

Where datasets go (first match wins):

1. An explicit folder (e.g. ``sword: dir:`` in a project's YAML).
2. The ``RIVMAPPER_DATA_DIR`` environment variable — e.g. one shared copy
   on a lab server.
3. ``<repo>/data/`` when running from a git checkout of RivMapper_v2.
4. The user's cache folder (when installed with pip, no checkout).
"""

from __future__ import annotations

import datetime as dt
import fnmatch
import hashlib
import logging
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

ENV_VAR = "RIVMAPPER_DATA_DIR"
MANIFEST = ".rivmapper_dataset.yaml"


# ─────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Dataset:
    """One dataset at one version, as described in ``datasets.yaml``."""

    name: str
    version: str
    title: str
    subdir: str
    access: str
    license: str | None
    citation: str | None
    url: str | None = None
    hash: str | None = None
    size_gb: float | None = None
    doi: str | None = None
    unpack: str | None = None
    expect: list[str] = field(default_factory=list)

    @property
    def expected_files(self) -> list[str]:
        return [f.format(version=self.version) for f in self.expect]


@cache
def _raw_registry() -> dict[str, Any]:
    text = (
        resources.files("rivmapper_v2")
        .joinpath("datasets.yaml")
        .read_text(encoding="utf-8")
    )
    return yaml.safe_load(text)


def dataset_names() -> list[str]:
    """Names of all registered datasets."""
    return list(_raw_registry())


def get_dataset(name: str, version: str | None = None) -> Dataset:
    """Look up a dataset in the registry.

    Raises:
        KeyError: Unknown dataset name or version.
    """
    reg = _raw_registry()
    if name not in reg:
        raise KeyError(f"Unknown dataset {name!r}. Available: {list(reg)}")
    entry = reg[name]
    version = version or entry["default_version"]
    versions = entry.get("versions", {})
    if version not in versions:
        raise KeyError(
            f"{name} version {version!r} is not in the registry. "
            f"Known versions: {list(versions)}"
        )
    v = versions[version]
    return Dataset(
        name=name,
        version=version,
        title=entry.get("title", name),
        subdir=entry.get("subdir", name),
        access=entry.get("access", "download"),
        license=entry.get("license"),
        citation=" ".join(entry["citation"].split()) if entry.get("citation") else None,
        url=v.get("url"),
        hash=v.get("hash"),
        size_gb=v.get("size_gb"),
        doi=v.get("doi"),
        unpack=v.get("unpack"),
        expect=list(v.get("expect", [])),
    )


# ─────────────────────────────────────────────────────────────────────
# Locations
# ─────────────────────────────────────────────────────────────────────
def repo_root() -> Path | None:
    """Root of the RivMapper_v2 git checkout this package runs from, if any."""
    root = Path(__file__).resolve().parent.parent
    if (root / "pyproject.toml").exists() and (root / ".git").exists():
        return root
    return None


def datasets_dir() -> Path:
    """Folder holding all datasets (see module docstring for the order)."""
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    root = repo_root()
    if root is not None:
        return root / "data"
    import pooch

    return Path(pooch.os_cache("rivmapper"))


def dataset_dir(ds: Dataset, override: Path | None = None) -> Path:
    """Folder for one dataset: ``override`` if given, else datasets dir / subdir."""
    return Path(override) if override else datasets_dir() / ds.subdir


def missing_files(ds: Dataset, folder: Path) -> list[str]:
    """Expected files of ``ds`` that are not present in ``folder``."""
    return [f for f in ds.expected_files if not (folder / f).exists()]


def is_installed(ds: Dataset, folder: Path | None = None) -> bool:
    return not missing_files(ds, dataset_dir(ds, folder))


def status(ds: Dataset, folder: Path | None = None) -> str:
    """'installed', 'partial (n/N files)', 'not downloaded', or the access type."""
    where = dataset_dir(ds, folder)
    n_missing, n_total = len(missing_files(ds, where)), len(ds.expected_files)
    if n_missing == 0:
        return "installed"
    if n_missing < n_total:
        return f"partial ({n_total - n_missing}/{n_total} files)"
    return "not downloaded" if ds.access == "download" else ds.access


# ─────────────────────────────────────────────────────────────────────
# Fetch / verify
# ─────────────────────────────────────────────────────────────────────
def fetch(
    name: str,
    version: str | None = None,
    folder: Path | None = None,
    force: bool = False,
) -> Path:
    """Make sure a dataset is on disk; download and unpack it if not.

    Args:
        name: Registry name, e.g. ``"sword"``.
        version: Dataset version; registry default if None.
        folder: Where to put it; default from ``dataset_dir``.
        force: Re-download even if all expected files exist.

    Returns:
        The folder holding the dataset's files.

    Raises:
        RuntimeError: Dataset cannot be downloaded automatically, the
            checksum does not match, or expected files are missing after
            unpacking.
    """
    ds = get_dataset(name, version)
    dest = dataset_dir(ds, folder)
    if not force and not missing_files(ds, dest):
        return dest
    if ds.access != "download" or not ds.url:
        raise RuntimeError(
            f"{ds.title} ({ds.version}) cannot be downloaded automatically "
            f"(access: {ds.access}). Place its files in {dest}."
        )

    import pooch

    dest.mkdir(parents=True, exist_ok=True)
    size = f" (~{ds.size_gb} GB)" if ds.size_gb else ""
    log.info("Downloading %s %s%s to %s", ds.title, ds.version, size, dest)
    archive_name = Path(ds.url).name
    try:
        archive = Path(
            pooch.retrieve(
                ds.url,
                known_hash=ds.hash,
                fname=archive_name,
                path=dest,
                progressbar=True,
            )
        )
    except ValueError as e:  # pooch raises ValueError on hash mismatch
        raise RuntimeError(
            f"{ds.title} download failed its checksum: {e}. If the archive was "
            f"republished, update the hash in datasets.yaml."
        ) from None

    if ds.unpack and zipfile.is_zipfile(archive):
        log.info("Unpacking %s", archive.name)
        _unzip_flat(archive, dest, ds.unpack)
        archive.unlink()

    still_missing = missing_files(ds, dest)
    if still_missing:
        raise RuntimeError(
            f"{ds.title} {ds.version}: expected files missing after download: "
            f"{still_missing}"
        )
    _write_manifest(ds, dest)
    log.info("%s %s ready in %s", ds.title, ds.version, dest)
    return dest


def _unzip_flat(archive: Path, dest: Path, pattern: str) -> None:
    """Extract members matching ``pattern`` into ``dest``, dropping folders."""
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if not name or name.startswith("._") or not fnmatch.fnmatch(name, pattern):
                continue
            with zf.open(member) as src, open(dest / name, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _write_manifest(ds: Dataset, folder: Path) -> None:
    files = {
        f: {"bytes": (folder / f).stat().st_size, "sha256": _sha256(folder / f)}
        for f in ds.expected_files
    }
    record = {
        "dataset": ds.name,
        "version": ds.version,
        "source": ds.url,
        "source_hash": ds.hash,
        "fetched_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "files": files,
    }
    with open(folder / MANIFEST, "w", encoding="utf-8") as f:
        yaml.safe_dump(record, f, sort_keys=False)


def verify(
    name: str, version: str | None = None, folder: Path | None = None
) -> list[str]:
    """Check a dataset's files against its manifest.

    Returns:
        A list of problems; empty if everything matches.
    """
    ds = get_dataset(name, version)
    dest = dataset_dir(ds, folder)
    problems = [f"missing: {f}" for f in missing_files(ds, dest)]
    manifest_path = dest / MANIFEST
    if not manifest_path.exists():
        if not problems:
            problems.append(
                f"no manifest in {dest} (files were not fetched by rivmapper); "
                f"cannot check contents — run `rivmapper data fetch {name} --force` "
                f"to re-download, or ignore if you trust these files"
            )
        return problems
    recorded = (yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}).get(
        "files", {}
    )
    for fname, info in recorded.items():
        path = dest / fname
        if path.exists() and _sha256(path) != info.get("sha256"):
            problems.append(f"changed since download: {fname}")
    return problems
