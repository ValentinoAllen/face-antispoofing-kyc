"""Build the image manifest (manifest v1, ``docs/SCHEMA.md`` §1) from a CelebA-Spoof label file."""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import pandas as pd

from antispoof.data import labels
from antispoof.data.config import CONFLICT_EXCLUDE, CONFLICT_POLICIES, CONFLICT_TRUST_PATH

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = (
    "image_path",
    "subject_id",
    "split",
    "label",
    "spoof_type",
    "illumination",
    "environment",
    "path_kind",
    "conflict",
)

ATTACK_CODE_COLUMNS = ("spoof_type", "illumination", "environment")
"""Manifest columns holding the 1-indexed codes of ``labels.ATTACK_CODE_INDICES``."""

_DTYPES = {
    "image_path": str,
    "subject_id": str,
    "split": str,
    "label": "int8",
    "spoof_type": "int32",
    "illumination": "int32",
    "environment": "int32",
    "path_kind": str,
    "conflict": bool,
}

_PARTS_AFTER_SPLIT = 3
"""Path components after the split directory: subject id, path kind, file name."""


class ManifestError(ValueError):
    """Raised when a label file or image path does not match the expected structure."""


@dataclass(frozen=True)
class ParsedPath:
    """Fields read from a relative image path."""

    source_split: str
    subject_id: str
    path_kind: str


@dataclass(frozen=True)
class ManifestStats:
    """Row counts from one manifest build. ``*_out`` counts are after the conflict policy."""

    source_split: str
    conflict_policy: str
    rows_in: int
    subjects_in: int
    live_by_path: int
    spoof_by_path: int
    live_by_label: int
    spoof_by_label: int
    conflicts_found: int
    conflicts_dropped: int
    conflicts_relabelled: int
    rows_out: int
    subjects_out: int
    live_out: int
    spoof_out: int

    def to_dict(self) -> dict[str, int | str]:
        """Return the stats as a plain dict, for printing or logging."""
        return asdict(self)


def load_label_json(path: Path) -> dict[str, list[int]]:
    """Read a CelebA-Spoof label JSON file.

    Args:
        path: Path to e.g. ``metas/intra_test/train_label.json``.

    Returns:
        Mapping from relative image path to its annotation vector.

    Raises:
        ManifestError: If the file does not hold a JSON object.
    """
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ManifestError(f"{path}: expected a JSON object mapping image paths to vectors.")
    return document


def parse_image_path(image_path: str) -> ParsedPath:
    """Read split, subject id and path kind from a relative image path.

    The split is found by locating the first ``train``/``test`` component, so any prefix before it
    is allowed. It must be followed by exactly ``{subject_id}/{live|spoof}/{filename}``.

    Args:
        image_path: POSIX path such as ``Data/train/5028/live/000001.jpg``.

    Returns:
        The parsed fields. ``subject_id`` stays a string so leading zeros survive.

    Raises:
        ManifestError: If there is no split component, the tail has the wrong number of
            components, or the path kind is not ``live``/``spoof``.
    """
    parts = PurePosixPath(image_path).parts
    positions = [index for index, part in enumerate(parts) if part in labels.SOURCE_SPLITS]
    if not positions:
        raise ManifestError(f"{image_path!r}: no component in {labels.SOURCE_SPLITS}.")
    tail = parts[positions[0] + 1 :]
    if len(tail) != _PARTS_AFTER_SPLIT:
        raise ManifestError(
            f"{image_path!r}: expected '<split>/<subject_id>/<live|spoof>/<filename>'."
        )
    subject_id, path_kind = tail[0], tail[1]
    if path_kind not in labels.PATH_KIND_TO_LABEL:
        raise ManifestError(
            f"{image_path!r}: path kind {path_kind!r} is not one of "
            f"{tuple(labels.PATH_KIND_TO_LABEL)}."
        )
    return ParsedPath(parts[positions[0]], subject_id, path_kind)


def build_manifest(
    label_vectors: Mapping[str, Sequence[int]], expected_split: str, conflict_policy: str
) -> tuple[pd.DataFrame, ManifestStats]:
    """Build the manifest for one label file and apply the conflict policy.

    A row is a conflict when its ``live``/``spoof`` path segment disagrees with label index 43,
    in either direction. The number of conflicting rows is always logged.

    Args:
        label_vectors: Mapping from relative image path to annotation vector.
        expected_split: The source split every path must belong to (``train`` or ``test``).
        conflict_policy: One of ``CONFLICT_POLICIES``.

    Returns:
        The manifest with columns ``MANIFEST_COLUMNS``, and the build stats.

    Raises:
        ManifestError: If the policy is unknown, the mapping is empty, or a path is malformed or
            belongs to another split.
        LabelVectorError: If a vector has the wrong length or an unknown label.
    """
    if conflict_policy not in CONFLICT_POLICIES:
        raise ManifestError(
            f"conflict_policy {conflict_policy!r} is not one of {CONFLICT_POLICIES}."
        )
    raw = _raw_frame(label_vectors, expected_split)
    manifest = _apply_conflict_policy(raw, conflict_policy)
    stats = _manifest_stats(raw, manifest, expected_split, conflict_policy)
    _log_conflicts(stats)
    return manifest, stats


def _manifest_row(
    image_path: str, vector: Sequence[int], expected_split: str
) -> tuple[object, ...]:
    parsed = parse_image_path(image_path)
    if parsed.source_split != expected_split:
        raise ManifestError(
            f"{image_path!r} belongs to {parsed.source_split!r}, not {expected_split!r}."
        )
    try:
        label = labels.get_label(vector)
        spoof_type = labels.get_spoof_type(vector)
        illumination = labels.get_illumination(vector)
        environment = labels.get_environment(vector)
    except labels.LabelVectorError as error:
        raise labels.LabelVectorError(f"{image_path!r}: {error}") from error
    conflict = label != labels.PATH_KIND_TO_LABEL[parsed.path_kind]
    return (
        image_path,
        parsed.subject_id,
        parsed.source_split,
        label,
        spoof_type,
        illumination,
        environment,
        parsed.path_kind,
        conflict,
    )


def _raw_frame(label_vectors: Mapping[str, Sequence[int]], expected_split: str) -> pd.DataFrame:
    if not label_vectors:
        raise ManifestError(f"Label mapping for split {expected_split!r} is empty.")
    rows = [_manifest_row(path, vector, expected_split) for path, vector in label_vectors.items()]
    frame = pd.DataFrame.from_records(rows, columns=list(MANIFEST_COLUMNS))
    return frame.astype(_DTYPES)


def _apply_conflict_policy(raw: pd.DataFrame, conflict_policy: str) -> pd.DataFrame:
    if conflict_policy == CONFLICT_EXCLUDE:
        manifest = raw.loc[~raw["conflict"]].copy()
    elif conflict_policy == CONFLICT_TRUST_PATH:
        manifest = raw.copy()
        manifest["label"] = manifest["path_kind"].map(labels.PATH_KIND_TO_LABEL).astype("int8")
    else:
        manifest = raw.copy()
    return manifest.reset_index(drop=True)


def _count(series: pd.Series, value: object) -> int:
    return int((series == value).sum())


def _manifest_stats(
    raw: pd.DataFrame, manifest: pd.DataFrame, source_split: str, conflict_policy: str
) -> ManifestStats:
    conflicts = int(raw["conflict"].sum())
    return ManifestStats(
        source_split=source_split,
        conflict_policy=conflict_policy,
        rows_in=len(raw),
        subjects_in=int(raw["subject_id"].nunique()),
        live_by_path=_count(raw["path_kind"], labels.PATH_KIND_LIVE),
        spoof_by_path=_count(raw["path_kind"], labels.PATH_KIND_SPOOF),
        live_by_label=_count(raw["label"], labels.LABEL_LIVE),
        spoof_by_label=_count(raw["label"], labels.LABEL_SPOOF),
        conflicts_found=conflicts,
        conflicts_dropped=conflicts if conflict_policy == CONFLICT_EXCLUDE else 0,
        conflicts_relabelled=conflicts if conflict_policy == CONFLICT_TRUST_PATH else 0,
        rows_out=len(manifest),
        subjects_out=int(manifest["subject_id"].nunique()),
        live_out=_count(manifest["label"], labels.LABEL_LIVE),
        spoof_out=_count(manifest["label"], labels.LABEL_SPOOF),
    )


def _log_conflicts(stats: ManifestStats) -> None:
    level = logging.WARNING if stats.conflicts_found else logging.INFO
    logger.log(
        level,
        "%s manifest: %d conflicting rows (path kind disagrees with label); "
        "conflict_policy=%s dropped %d, relabelled %d; %d rows in, %d rows out.",
        stats.source_split,
        stats.conflicts_found,
        stats.conflict_policy,
        stats.conflicts_dropped,
        stats.conflicts_relabelled,
        stats.rows_in,
        stats.rows_out,
    )
