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

This module holds no encoding logic and imports nothing from :mod:`antispoof.eval`, so training can
depend on it.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from antispoof.data.manifest import MANIFEST_COLUMNS, MANIFEST_DTYPES, parse_image_path

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
