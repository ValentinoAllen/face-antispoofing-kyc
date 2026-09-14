"""Load and validate the data configuration (``configs/data.yaml``)."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from antispoof.data import labels

CONFLICT_EXCLUDE = "exclude"
CONFLICT_TRUST_LABEL = "trust_label"
CONFLICT_TRUST_PATH = "trust_path"
CONFLICT_POLICIES = (CONFLICT_EXCLUDE, CONFLICT_TRUST_LABEL, CONFLICT_TRUST_PATH)
"""How rows whose ``live``/``spoof`` path segment disagrees with label index 43 are handled.

- ``exclude``: drop the row.
- ``trust_label``: keep the row; the label comes from index 43.
- ``trust_path``: keep the row; the label comes from the path segment.
"""

_DATA_KEYS = frozenset(
    {
        "dataset_root",
        "metas_dir",
        "protocol",
        "train_label_file",
        "test_label_file",
        "conflict_policy",
        "excluded_subjects",
        "manifest_dir",
        "split_assignment_path",
    }
)
_SPLIT_KEYS = frozenset({"val_fraction", "seed", "stratify_bins"})

_ValueT = TypeVar("_ValueT", str, int, float)


class DataConfigError(ValueError):
    """Raised when the data configuration is missing keys or holds invalid values."""


@dataclass(frozen=True)
class DataConfig:
    """Resolved data settings. Each field is documented in ``configs/data.yaml``."""

    dataset_root: Path
    metas_dir: str
    protocol: str
    train_label_file: str
    test_label_file: str
    conflict_policy: str
    excluded_subjects: tuple[str, ...]
    manifest_dir: Path
    split_assignment_path: Path
    val_fraction: float
    seed: int
    stratify_bins: int

    def label_path(self, source_split: str) -> Path:
        """Return the label JSON path for one source split.

        Args:
            source_split: ``train`` or ``test``.

        Returns:
            ``dataset_root / metas_dir / protocol / <label file>``.

        Raises:
            DataConfigError: If ``source_split`` is not a source split.
        """
        label_files = {
            labels.SPLIT_TRAIN: self.train_label_file,
            labels.SPLIT_TEST: self.test_label_file,
        }
        if source_split not in label_files:
            raise DataConfigError(f"No label file for split {source_split!r}.")
        return self.dataset_root / self.metas_dir / self.protocol / label_files[source_split]


def load_data_config(path: Path) -> DataConfig:
    """Load ``configs/data.yaml`` into a validated :class:`DataConfig`.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        DataConfigError: If sections or keys are missing or unknown, or a value is invalid.
    """
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict) or set(document) != {"data", "split"}:
        raise DataConfigError(
            f"{path}: expected exactly the top-level sections 'data' and 'split'."
        )
    data = _section(document, "data", _DATA_KEYS)
    split = _section(document, "split", _SPLIT_KEYS)
    config = DataConfig(
        dataset_root=Path(_get(data, "dataset_root", str)),
        metas_dir=_get(data, "metas_dir", str),
        protocol=_get(data, "protocol", str),
        train_label_file=_get(data, "train_label_file", str),
        test_label_file=_get(data, "test_label_file", str),
        conflict_policy=_get(data, "conflict_policy", str),
        excluded_subjects=_get_subject_ids(data, "excluded_subjects"),
        manifest_dir=Path(_get(data, "manifest_dir", str)),
        split_assignment_path=Path(_get(data, "split_assignment_path", str)),
        val_fraction=_get(split, "val_fraction", float),
        seed=_get(split, "seed", int),
        stratify_bins=_get(split, "stratify_bins", int),
    )
    _validate_values(config)
    return config


def _section(document: Mapping[str, Any], name: str, keys: frozenset[str]) -> Mapping[str, Any]:
    section = document[name]
    if not isinstance(section, dict):
        raise DataConfigError(f"Section {name!r} must be a mapping.")
    missing, unknown = keys - set(section), set(section) - keys
    if missing or unknown:
        raise DataConfigError(
            f"Section {name!r}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}."
        )
    return section


def _get(section: Mapping[str, Any], key: str, expected: type[_ValueT]) -> _ValueT:
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, expected):
        raise DataConfigError(
            f"{key}: expected {expected.__name__}, got {type(value).__name__} ({value!r})."
        )
    return value


def _get_subject_ids(section: Mapping[str, Any], key: str) -> tuple[str, ...]:
    values = section[key]
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise DataConfigError(
            f"{key}: expected a list of quoted strings so leading zeros survive, got {values!r}."
        )
    return tuple(values)


def _validate_values(config: DataConfig) -> None:
    if config.conflict_policy not in CONFLICT_POLICIES:
        raise DataConfigError(
            f"conflict_policy {config.conflict_policy!r} is not one of {CONFLICT_POLICIES}."
        )
    if not 0 < config.val_fraction < 1:
        raise DataConfigError(f"val_fraction must be in (0, 1), got {config.val_fraction}.")
    if config.stratify_bins < 1:
        raise DataConfigError(f"stratify_bins must be >= 1, got {config.stratify_bins}.")
    if config.seed < 0:
        raise DataConfigError(f"seed must be >= 0, got {config.seed}.")
    if len(set(config.excluded_subjects)) != len(config.excluded_subjects):
        raise DataConfigError(f"excluded_subjects has duplicates: {config.excluded_subjects}.")
