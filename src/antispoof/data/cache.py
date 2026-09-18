"""Contract of the normalized image cache: what it holds, and how a run reads it.

The cache is written by :mod:`antispoof.data.cache_build` and read by training. Every cached image
is the source image resized to one square size and re-encoded at one JPEG quality and subsampling,
the same for every row regardless of class, so the header-level class signature that
``20260916-075616-probe_metadata`` found is gone from the pixels a cached run reads.

Layout mirrors the source ``image_path`` (``docs/SCHEMA.md`` §1.3), with the **manifest** split as
the first component::

    <cache-root>/<split>/<subject_id>/<live|spoof>/<filename>

``cached_path`` in ``cache_manifest_<split>.csv`` is that path, POSIX and relative to the cache
root, so a cache can be moved without rewriting its manifests.

:func:`load_cache` refuses a cache that does not belong to the run reading it: the wrong name, a
cache built from other manifests, or one that does not cover every subset row. A partial cache would
otherwise train on fewer images than the run record claims.

This module holds no encoding logic and imports nothing from :mod:`antispoof.eval` or
:mod:`antispoof.training.run`, so training can depend on it. The builder is
:mod:`antispoof.data.cache_build`.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from antispoof.data.dataset import ImageTransform, ManifestDataset, load_rgb_image
from antispoof.data.manifest import MANIFEST_COLUMNS, MANIFEST_DTYPES, parse_image_path
from antispoof.training.reproducibility import sha256_file

CACHE_MANIFEST_FILENAME = "cache_manifest_{split}.csv"
CACHE_SUMMARY_FILENAME = "cache_summary.json"

CACHED_PATH_COLUMN = "cached_path"
SOURCE_SHA256_COLUMN = "source_sha256"
CACHE_MANIFEST_COLUMNS = (*MANIFEST_COLUMNS, CACHED_PATH_COLUMN, SOURCE_SHA256_COLUMN)
"""Columns of ``cache_manifest_<split>.csv``: manifest v1, then the cache's two columns."""

CACHE_MANIFEST_DTYPES = {
    **MANIFEST_DTYPES,
    CACHED_PATH_COLUMN: str,
    SOURCE_SHA256_COLUMN: str,
}


class CacheError(ValueError):
    """Raised when a cache file is missing, malformed, or does not match the run that reads it."""


def cached_path(image_path: str, split: str) -> str:
    """Return the cache-relative path of one source row, mirroring the source layout.

    Args:
        image_path: The manifest's ``image_path``, relative to the dataset root.
        split: The manifest split (``train``, ``val`` or ``test``), which replaces the source
            path's official-split component.

    Returns:
        ``<split>/<subject_id>/<live|spoof>/<filename>``, POSIX and relative to the cache root.

    Raises:
        ManifestError: If ``image_path`` does not have the expected layout.
    """
    parsed = parse_image_path(image_path)
    return f"{split}/{parsed.subject_id}/{parsed.path_kind}/{image_path.rsplit('/', 1)[-1]}"


def cache_manifest_paths(cache_dir: Path, splits: Sequence[str]) -> dict[str, Path]:
    """Return the cache manifest of each split, in the given order.

    Args:
        cache_dir: The cache root.
        splits: Splits to include.

    Returns:
        ``{split: <cache_dir>/cache_manifest_<split>.csv}``.
    """
    return {split: cache_dir / CACHE_MANIFEST_FILENAME.format(split=split) for split in splits}


def read_cache_manifest(path: Path) -> pd.DataFrame:
    """Read a complete cache manifest, checking its header against the contract.

    Args:
        path: Path of a ``cache_manifest_<split>.csv``.

    Returns:
        The rows, with the manifest v1 dtypes plus two string columns.

    Raises:
        CacheError: If the file is missing or its header does not match
            :data:`CACHE_MANIFEST_COLUMNS`.
    """
    if not path.is_file():
        raise CacheError(f"Cache manifest not found: {path}")
    header = pd.read_csv(path, nrows=0).columns
    if tuple(header) != CACHE_MANIFEST_COLUMNS:
        raise CacheError(f"{path}: expected columns {CACHE_MANIFEST_COLUMNS}, got {tuple(header)}.")
    return pd.read_csv(path, dtype=CACHE_MANIFEST_DTYPES, keep_default_na=False)


def read_cached_rows(path: Path) -> dict[str, tuple[str, str]]:
    """Read what a previous build cached, tolerating a file an interrupted build left truncated.

    Every value is read as text and only the three columns the resume check needs are kept, so a
    half-written last line cannot break the read: a row with too many fields is skipped, a row with
    an empty cached path or digest is dropped here, and a row with a truncated digest simply fails
    the digest comparison and is rebuilt.

    Args:
        path: Path of a ``cache_manifest_<split>.csv``. A missing file means nothing was cached.

    Returns:
        ``{image_path: (cached_path, source_sha256)}``; empty when there is nothing usable.
    """
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="skip")
    wanted = ("image_path", CACHED_PATH_COLUMN, SOURCE_SHA256_COLUMN)
    if not set(wanted) <= set(frame.columns):
        return {}
    return {
        str(image_path): (str(cached), str(digest))
        for image_path, cached, digest in zip(
            frame["image_path"],
            frame[CACHED_PATH_COLUMN],
            frame[SOURCE_SHA256_COLUMN],
            strict=True,
        )
        if str(image_path) and str(cached) and str(digest)
    }


def read_cache_summary(cache_dir: Path) -> dict[str, Any]:
    """Read the ``cache_summary.json`` that describes how a cache was built.

    Args:
        cache_dir: The cache root.

    Returns:
        The parsed summary.

    Raises:
        CacheError: If the file is missing or is not a JSON object.
    """
    path = cache_dir / CACHE_SUMMARY_FILENAME
    if not path.is_file():
        raise CacheError(f"{cache_dir} is not a cache directory: {CACHE_SUMMARY_FILENAME} missing.")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise CacheError(f"{path}: expected a JSON object.")
    return summary


class CacheMismatchError(CacheError):
    """Raised when a cache was not built from the manifests a run reads, or misses its rows."""


@dataclass(frozen=True)
class CacheBinding:
    """The cache a run reads, and everything its record stores about it.

    Attributes:
        name: ``cache.name``, which matched the cache's own ``cache_summary.json``.
        cache_dir: The resolved cache root.
        settings: The ``settings`` block of ``cache_summary.json``: how the cache was built.
        manifest_sha256: SHA-256 hex of each cache manifest the run read, by file name.
        summary_sha256: SHA-256 hex of ``cache_summary.json``.
        cached_paths: Cache-relative path of every row of each subset, in subset row order.
    """

    name: str
    cache_dir: Path
    settings: dict[str, Any]
    manifest_sha256: dict[str, str]
    summary_sha256: str
    cached_paths: dict[str, list[str]]

    def record_entry(self) -> dict[str, Any]:
        """Return the run record's ``cache`` block (``docs/SCHEMA.md`` §3).

        Returns:
            ``{name, dir, settings, summary_sha256, cache_manifest_sha256}``.
        """
        return {
            "name": self.name,
            "dir": self.cache_dir.as_posix(),
            "settings": self.settings,
            "summary_sha256": self.summary_sha256,
            "cache_manifest_sha256": self.manifest_sha256,
        }


def load_cache(
    cache_dir: Path,
    name: str,
    subsets: Mapping[str, pd.DataFrame],
    manifest_sha256: Mapping[str, str],
) -> CacheBinding:
    """Bind a run to a cache, refusing one that does not match the run.

    A path alone does not identify a cache, and a partial cache would silently train on fewer
    images than the record claims, so three things are checked before anything is read: the cache's
    name, that it was built from exactly the manifests this run reads, and that it covers every row
    of every subset.

    Args:
        cache_dir: The resolved cache root.
        name: ``cache.name`` from the experiment config.
        subsets: ``{split: subset}``, the rows the run will use.
        manifest_sha256: The run's own ``manifest_sha256``, by manifest file name.

    Returns:
        The binding, ready for :class:`CachedManifestDataset` and the run record.

    Raises:
        CacheError: If ``cache_summary.json`` or a cache manifest is missing or malformed.
        CacheMismatchError: If the name, the source manifests or the coverage disagree.
    """
    summary = read_cache_summary(cache_dir)
    if summary.get("name") != name:
        raise CacheMismatchError(
            f"{cache_dir} holds cache {summary.get('name')!r}, but the config asks for {name!r}."
        )
    built_from = summary.get("source_manifest_sha256")
    if built_from != dict(manifest_sha256):
        raise CacheMismatchError(
            f"{cache_dir} was built from manifests {built_from}, but this run reads "
            f"{dict(manifest_sha256)}."
        )
    cached_paths: dict[str, list[str]] = {}
    hashes: dict[str, str] = {}
    for split, path in cache_manifest_paths(cache_dir, tuple(subsets)).items():
        frame = read_cache_manifest(path)
        cached_paths[split] = _resolve_cached_paths(frame, subsets[split], split, cache_dir)
        hashes[path.name] = sha256_file(path)
    return CacheBinding(
        name=name,
        cache_dir=cache_dir,
        settings=dict(summary.get("settings", {})),
        manifest_sha256=hashes,
        summary_sha256=sha256_file(cache_dir / CACHE_SUMMARY_FILENAME),
        cached_paths=cached_paths,
    )


def _resolve_cached_paths(
    cache_manifest: pd.DataFrame, subset: pd.DataFrame, split: str, cache_dir: Path
) -> list[str]:
    lookup = dict(
        zip(cache_manifest["image_path"], cache_manifest[CACHED_PATH_COLUMN], strict=True)
    )
    wanted = [str(image_path) for image_path in subset["image_path"]]
    missing = [image_path for image_path in wanted if image_path not in lookup]
    if missing:
        raise CacheMismatchError(
            f"{cache_dir} misses {len(missing)} of the {len(wanted)} {split} rows this run needs, "
            f"for example {missing[:3]}. A cache built with --limit, or interrupted, is partial."
        )
    return [str(lookup[image_path]) for image_path in wanted]


class CachedManifestDataset(ManifestDataset):
    """A :class:`ManifestDataset` that reads its pixels from the normalized cache.

    Labels, row indices and the transform are unchanged, and ``predictions.csv`` is built from the
    subset frame, so predictions still carry the source manifest's ``image_path``. Only where the
    pixels come from differs.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        cache_dir: Path,
        transform: ImageTransform,
        cached_paths: Sequence[str],
    ) -> None:
        """Store the rows and the cache-relative path of each one.

        Args:
            manifest: Frame with at least ``image_path`` and ``label`` columns.
            cache_dir: The cache root the ``cached_paths`` values are relative to.
            transform: Applied to every decoded RGB image.
            cached_paths: One cache-relative path per manifest row, in row order.

        Raises:
            ValueError: If a column or label is invalid, or ``cached_paths`` does not have exactly
                one entry per row.
        """
        super().__init__(manifest, cache_dir, transform)
        if len(cached_paths) != len(self):
            raise ValueError(
                f"cached_paths has {len(cached_paths)} entries for {len(self)} manifest rows."
            )
        self._cached_paths = list(cached_paths)

    def load_image(self, index: int) -> Image.Image:
        """Decode one cached image.

        Args:
            index: Positional row index in the manifest.

        Returns:
            The RGB image the transform receives.

        Raises:
            ImageLoadError: If the cached image is missing or cannot be decoded. The message names
                the cached path, not the source path.
        """
        return load_rgb_image(self._dataset_root / self._cached_paths[index])
