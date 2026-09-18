"""Build the normalized image cache: decode, resize to one square size, re-encode at one quality.

``20260916-075616-probe_metadata`` reached pooled val ACER 0.00% from image headers alone, and Part
1 of ``20260918-131447-counterfactual_jpeg`` traced that to one quantization table set per class.
This builder removes that signature from the data itself: every row, live or spoof, is written at
the same size, quality and subsampling, so a model trained on the cache cannot read the class off
the encoding.

Order of operations, per split:

1. Read the manifest and check that no two rows map to one cached path.
2. Read what a previous run already cached, so an interrupted build resumes.
3. For each row: hash the source, skip it when the cached file exists and the hash matches,
   otherwise decode, resize, re-encode and write. Unreadable rows are counted and listed, and do
   not stop the build.
4. Verify a seeded sample of cached files against the target quality, subsampling and size.

The cache manifest is written incrementally and flushed as the build proceeds, so an interruption
keeps its progress. Nothing under the dataset root is ever written.

No face crop in v1: the bounding-box format is still an open question (``docs/SCHEMA.md`` §1.4), and
``cache.face_crop`` names that explicitly rather than leaving it implied.
"""

import csv
import logging
import platform
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import PIL
import yaml
from PIL import Image

from antispoof.data.cache import (
    CACHE_MANIFEST_COLUMNS,
    CACHE_SUMMARY_FILENAME,
    cache_manifest_paths,
    cached_path,
    read_cached_rows,
)
from antispoof.data.config import DataConfig
from antispoof.data.dataset import ImageLoadError, load_rgb_image
from antispoof.data.manifest import MANIFEST_COLUMNS
from antispoof.eval.jpeg_tables import (
    RESAMPLE_FILTERS,
    encode_jpeg,
    read_jpeg_header,
    reference_tables,
)
from antispoof.eval.metadata_probe import SUBSAMPLING_NAMES
from antispoof.training import reproducibility
from antispoof.training.config import TrainConfigError, parse_section
from antispoof.training.run import load_manifests, manifest_paths, write_json

logger = logging.getLogger(__name__)

SUBSAMPLING_CODES = {name: code for code, name in SUBSAMPLING_NAMES.items()}
"""Subsampling code of each name accepted in the config, e.g. ``{"4:2:0": 2}``."""

JPEG_SUFFIXES = (".jpg", ".jpeg")
LAYOUT = "<split>/<subject_id>/<live|spoof>/<filename>, mirroring the source image_path"


class CacheConfigError(ValueError):
    """Raised when the cache config is missing keys or holds invalid values."""


class CacheCollisionError(ValueError):
    """Raised when two manifest rows of one split map to the same cached path."""


class CacheVerificationError(RuntimeError):
    """Raised when a cached file's header is not the target quality, subsampling or size."""


@dataclass(frozen=True)
class CacheSettings:
    """How every image is normalized (``cache:``). One setting for every row, whatever its class."""

    name: str
    target_size: int
    quality: int
    subsampling: str
    resample_filter: str
    face_crop: bool


@dataclass(frozen=True)
class VerifyConfig:
    """The seeded sample of cached files whose header is checked (``verify:``)."""

    sample_rows: int
    seed: int


@dataclass(frozen=True)
class ProgressConfig:
    """How often progress is logged and the cache manifest flushed (``progress:``)."""

    every: int


@dataclass(frozen=True)
class CacheConfig:
    """A resolved cache config. Fields are documented in the YAML file."""

    cache: CacheSettings
    verify: VerifyConfig
    progress: ProgressConfig

    def to_dict(self) -> dict[str, Any]:
        """Return the config as nested plain dicts, for hashing and the summary file."""
        return asdict(self)


@dataclass(frozen=True)
class ResolvedSettings:
    """The cache settings with the subsampling code and resampling filter resolved once."""

    target_size: int
    quality: int
    subsampling_code: int
    resample: Image.Resampling


@dataclass(frozen=True)
class CacheFailure:
    """One manifest row whose source image could not be read."""

    image_path: str
    split: str
    error: str


@dataclass(frozen=True)
class CacheInputs:
    """Everything a cache build needs. ``data_config`` already carries CLI path overrides.

    Attributes:
        config: The cache config.
        config_path: Path of the cache config, under ``configs/``.
        data_config: Manifest directory and dataset root.
        splits: Splits to cache, in build order.
        output_dir: The cache root to write.
        repo_root: The repository root, for the git state recorded in the summary.
        limit: Cache only the first ``limit`` rows of each split, for a quick check.
    """

    config: CacheConfig
    config_path: Path
    data_config: DataConfig
    splits: tuple[str, ...]
    output_dir: Path
    repo_root: Path
    limit: int | None = None


@dataclass(frozen=True)
class CacheReport:
    """Where the cache was written and what its summary says."""

    cache_dir: Path
    summary: dict[str, Any]


def load_cache_config(path: Path) -> CacheConfig:
    """Load a YAML such as ``configs/cache_v1.yaml`` into a validated config.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        CacheConfigError: If sections or keys are missing or unknown, a value has the wrong type,
            or a value is invalid.
    """
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    section_types = {field.name: field.type for field in fields(CacheConfig)}
    if not isinstance(document, dict) or set(document) != set(section_types):
        raise CacheConfigError(
            f"{path}: expected exactly the top-level sections {sorted(section_types)}."
        )
    try:
        config = CacheConfig(
            **{
                name: parse_section(name, document[name], kind)
                for name, kind in section_types.items()
            }
        )
    except TrainConfigError as error:
        raise CacheConfigError(f"{path}: {error}") from error
    validate_cache_config(config)
    return config


def validate_cache_config(config: CacheConfig) -> None:
    """Check the normalization settings, the verification sample and the progress interval.

    Args:
        config: The configuration.

    Raises:
        CacheConfigError: Listing every violated constraint.
    """
    cache = config.cache
    checks = [
        (bool(cache.name.strip()), "cache.name must not be empty"),
        (cache.target_size > 0, "cache.target_size must be > 0"),
        (1 <= cache.quality <= 100, "cache.quality must be in [1, 100]"),
        (
            cache.subsampling in SUBSAMPLING_CODES,
            f"cache.subsampling must be one of {sorted(SUBSAMPLING_CODES)}",
        ),
        (
            cache.resample_filter in RESAMPLE_FILTERS,
            f"cache.resample_filter must be one of {RESAMPLE_FILTERS}",
        ),
        (not cache.face_crop, "cache.face_crop must be false: the bbox format is still open"),
        (config.verify.sample_rows >= 0, "verify.sample_rows must be >= 0"),
        (config.verify.seed >= 0, "verify.seed must be >= 0"),
        (config.progress.every >= 1, "progress.every must be >= 1"),
    ]
    problems = [message for passed, message in checks if not passed]
    if problems:
        raise CacheConfigError("Invalid cache config: " + "; ".join(problems) + ".")


def resolve_settings(cache: CacheSettings) -> ResolvedSettings:
    """Resolve the subsampling name and filter name to the values Pillow takes.

    Args:
        cache: The validated ``cache:`` section.

    Returns:
        The resolved settings.
    """
    return ResolvedSettings(
        target_size=cache.target_size,
        quality=cache.quality,
        subsampling_code=SUBSAMPLING_CODES[cache.subsampling],
        resample=Image.Resampling[cache.resample_filter.upper()],
    )


def check_no_collisions(manifest: pd.DataFrame, split: str) -> None:
    """Check that no two rows of one split map to the same cached path.

    The layout mirrors the source path and ``image_path`` is unique across the manifests
    (``docs/SCHEMA.md`` §1.3), so this cannot happen; it is kept as a cheap invariant over the
    manifest alone, run before any image is read. Two rows with the same ``image_path`` collide
    too: either way one cached file would carry two cache-manifest rows.

    Args:
        manifest: The split's manifest.
        split: The split name.

    Raises:
        CacheCollisionError: Naming the colliding source paths.
    """
    seen: dict[str, str] = {}
    collisions: list[str] = []
    for image_path in manifest["image_path"]:
        target = cached_path(str(image_path), split)
        if target in seen:
            collisions.append(f"{seen[target]} and {image_path} both map to {target}")
        else:
            seen[target] = str(image_path)
    if collisions:
        raise CacheCollisionError(
            f"{split}: {len(collisions)} cached-path collision(s): " + "; ".join(collisions[:10])
        )


def cache_image(source: Path, target: Path, settings: ResolvedSettings) -> int:
    """Decode one image, resize it to the target square and write it at the target quality.

    Args:
        source: The source image.
        target: Where to write the cached JPEG. Parent directories are created.
        settings: The resolved normalization settings.

    Returns:
        The number of bytes written.

    Raises:
        ImageLoadError: If the source is missing or cannot be decoded.
    """
    image = load_rgb_image(source)
    side = settings.target_size
    resized = image.resize((side, side), resample=settings.resample)
    data = encode_jpeg(resized, settings.quality, settings.subsampling_code)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return len(data)


def verify_cache_sample(
    cached_paths: Sequence[str], cache_dir: Path, settings: ResolvedSettings, verify: VerifyConfig
) -> dict[str, Any]:
    """Check a seeded sample of cached files against the target tables and size, from headers only.

    The whole table set is compared with the one Pillow writes at the target quality, so the check
    covers the subsampling as well as the quality.

    Args:
        cached_paths: Cache-relative paths of the rows this build wrote or kept.
        cache_dir: The cache root.
        settings: The resolved normalization settings.
        verify: Sample size and seed.

    Returns:
        ``{"sample_rows", "checked", "seed"}``.

    Raises:
        CacheVerificationError: Naming the first cached file whose header is not the target.
    """
    checked = min(verify.sample_rows, len(cached_paths))
    rng = np.random.default_rng(verify.seed)
    expected = reference_tables(settings.quality, settings.subsampling_code)
    side = settings.target_size
    for index in rng.permutation(len(cached_paths))[:checked]:
        relative = cached_paths[int(index)]
        header = read_jpeg_header(cache_dir / relative)
        if (header.width, header.height) != (side, side):
            raise CacheVerificationError(
                f"{relative}: header reports {header.width}x{header.height}, not {side}x{side}."
            )
        if header.tables != expected:
            raise CacheVerificationError(
                f"{relative}: header reports luma mean {np.mean(header.tables.luma)}, not the "
                f"quality {settings.quality} table (luma mean {np.mean(expected.luma)})."
            )
    return {"sample_rows": verify.sample_rows, "checked": checked, "seed": verify.seed}


def build_cache(inputs: CacheInputs) -> CacheReport:
    """Cache every row of the configured splits and write the manifests and the summary.

    Writes ``<output_dir>/<split>/...`` for the images, one ``cache_manifest_<split>.csv`` per
    split, and ``cache_summary.json``. Nothing under the dataset root is written.

    Args:
        inputs: Config, paths, splits and an optional row limit.

    Returns:
        The cache root and the summary.

    Raises:
        CacheConfigError, CacheCollisionError, ManifestError, ValueError: Before any image is
            written.
        CacheVerificationError: If a cached file's header is not the target.
    """
    validate_cache_config(inputs.config)
    manifests = load_manifests(inputs.splits, inputs.data_config)
    for split, manifest in manifests.items():
        check_no_collisions(manifest, split)
    settings = resolve_settings(inputs.config.cache)
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    summary = _new_summary(inputs)
    failures: list[CacheFailure] = []
    for split, path in cache_manifest_paths(inputs.output_dir, inputs.splits).items():
        frame = manifests[split]
        frame = frame if inputs.limit is None else frame.head(inputs.limit)
        summary["splits"][split] = _cache_split(inputs, split, frame, path, settings, failures)
        summary["failures"] = [asdict(failure) for failure in failures]
        write_json(summary, inputs.output_dir / CACHE_SUMMARY_FILENAME)
    return CacheReport(cache_dir=inputs.output_dir, summary=summary)


def _new_summary(inputs: CacheInputs) -> dict[str, Any]:
    git = reproducibility.git_state(inputs.repo_root)
    resolved: dict[str, Any] = reproducibility.to_json_compatible(
        {"experiment": inputs.config.to_dict(), "data": asdict(inputs.data_config)}
    )
    return {
        "name": inputs.config.cache.name,
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": reproducibility.repo_relative_config_path(
            inputs.config_path, inputs.repo_root
        ),
        "config_hash": reproducibility.config_hash(resolved),
        "git_sha": git.sha,
        "git_dirty": git.dirty,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pillow": PIL.__version__,
        },
        "settings": asdict(inputs.config.cache),
        "layout": LAYOUT,
        "cached_path": "POSIX, relative to the cache root",
        "source_manifest_sha256": {
            path.name: reproducibility.sha256_file(path)
            for path in manifest_paths(inputs.data_config, inputs.splits).values()
        },
        "limit": inputs.limit,
        "splits": {},
        "failures": [],
    }


def _cache_split(
    inputs: CacheInputs,
    split: str,
    manifest: pd.DataFrame,
    manifest_path: Path,
    settings: ResolvedSettings,
    failures: list[CacheFailure],
) -> dict[str, Any]:
    previous = read_cached_rows(manifest_path)
    started = time.monotonic()
    counts = dict.fromkeys(("cached", "skipped", "failed", "bytes", "non_jpeg_suffix"), 0)
    cached_paths: list[str] = []
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CACHE_MANIFEST_COLUMNS)
        for position, row in enumerate(_source_rows(manifest), start=1):
            image_path = str(row[0])
            relative = cached_path(image_path, split)
            try:
                digest = _cache_one(inputs, settings, previous, (image_path, relative), counts)
            except ImageLoadError as error:
                counts["failed"] += 1
                failures.append(CacheFailure(image_path, split, str(error)))
            else:
                writer.writerow([*row, relative, digest])
                cached_paths.append(relative)
            if position % inputs.config.progress.every == 0:
                handle.flush()
                _log_progress(split, position, len(manifest), counts)
    _log_progress(split, len(manifest), len(manifest), counts)
    verification = verify_cache_sample(
        cached_paths, inputs.output_dir, settings, inputs.config.verify
    )
    return {
        "rows": int(len(manifest)),
        **{key: int(value) for key, value in counts.items()},
        "wall_time_s": round(time.monotonic() - started, 3),
        "cache_manifest": manifest_path.name,
        "cache_manifest_sha256": reproducibility.sha256_file(manifest_path),
        "verification": verification,
    }


def _source_rows(manifest: pd.DataFrame) -> Iterator[tuple[Any, ...]]:
    rows: Iterator[tuple[Any, ...]] = manifest[list(MANIFEST_COLUMNS)].itertuples(
        index=False, name=None
    )
    return rows


def _cache_one(
    inputs: CacheInputs,
    settings: ResolvedSettings,
    previous: Mapping[str, tuple[str, str]],
    paths: tuple[str, str],
    counts: dict[str, int],
) -> str:
    """Hash one source image and cache it unless an up-to-date cached file already exists.

    Args:
        inputs: The build inputs.
        settings: The resolved normalization settings.
        previous: What a previous build cached, from :func:`read_cached_rows`.
        paths: ``(image_path, cached_path)`` of the row.
        counts: Per-split counters, updated in place.

    Returns:
        The source image's SHA-256 hex.

    Raises:
        ImageLoadError: If the source cannot be hashed or decoded.
    """
    image_path, relative = paths
    source = inputs.data_config.dataset_root / image_path
    target = inputs.output_dir / relative
    try:
        digest = reproducibility.sha256_file(source)
    except OSError as error:
        raise ImageLoadError(f"Cannot read source image {source}: {error}") from error
    prior = previous.get(image_path)
    if prior is not None and prior[1] == digest and target.is_file():
        counts["skipped"] += 1
        return digest
    counts["bytes"] += cache_image(source, target, settings)
    counts["cached"] += 1
    if not image_path.lower().endswith(JPEG_SUFFIXES):
        counts["non_jpeg_suffix"] += 1
    return digest


def _log_progress(split: str, done: int, total: int, counts: Mapping[str, int]) -> None:
    logger.info(
        "%s: %d of %d rows (%d cached, %d skipped, %d failed, %d bytes).",
        split,
        done,
        total,
        counts["cached"],
        counts["skipped"],
        counts["failed"],
        counts["bytes"],
    )


def format_report(report: CacheReport) -> str:
    """Render the cache settings, the per-split counts and any failures as plain text.

    Args:
        report: From :func:`build_cache`.

    Returns:
        Multi-line text.
    """
    summary = report.summary
    settings = summary["settings"]
    lines = [
        f"Normalized image cache {summary['name']}: {report.cache_dir}",
        f"  config: {summary['config_path']}  config_hash: {summary['config_hash']}",
        f"  git_sha: {summary['git_sha']}  git_dirty: {summary['git_dirty']}",
        f"  pillow: {summary['environment']['pillow']}  limit: {summary['limit']}",
        f"  settings: {settings['target_size']}px square, quality {settings['quality']}, "
        f"subsampling {settings['subsampling']}, filter {settings['resample_filter']}, "
        f"face_crop {settings['face_crop']}",
        f"  layout: {LAYOUT}",
        f"  source manifests: {summary['source_manifest_sha256']}",
    ]
    for split, stats in summary["splits"].items():
        lines.append(
            f"  {split}: {stats['rows']} rows, {stats['cached']} cached, {stats['skipped']} "
            f"skipped, {stats['failed']} failed, {stats['bytes']} bytes, "
            f"{stats['wall_time_s']} s; verified {stats['verification']['checked']} headers"
        )
        lines.append(f"    {stats['cache_manifest']} sha256: {stats['cache_manifest_sha256']}")
        if stats["non_jpeg_suffix"]:
            lines.append(
                f"    warning: {stats['non_jpeg_suffix']} cached files hold JPEG bytes under a "
                "source filename whose suffix is not .jpg or .jpeg"
            )
    failures = summary["failures"]
    lines.append(f"Unreadable source images: {len(failures)} (they did not stop the build)")
    lines.extend(
        f"  {entry['split']} {entry['image_path']}: {entry['error']}" for entry in failures
    )
    if summary["git_dirty"]:
        lines.append("WARNING: git_dirty is true; a run citing this cache cannot be cited either.")
    return "\n".join(lines)
