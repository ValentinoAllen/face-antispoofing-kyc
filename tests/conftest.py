"""Shared synthetic fixtures. Nothing here reads from or writes to ``data/``."""

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from antispoof.data import labels
from antispoof.data.config import DataConfig, load_data_config

REPO_DATA_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"

SYNTHETIC_SPOOF_TYPE = 1
SYNTHETIC_ILLUMINATION = 1
SYNTHETIC_ENVIRONMENT = 1


def _vector(
    label: int, spoof_type: int = 0, illumination: int = 0, environment: int = 0
) -> list[int]:
    vector = [0] * labels.VECTOR_LENGTH
    vector[labels.INDEX_SPOOF_TYPE] = spoof_type
    vector[labels.INDEX_ILLUMINATION] = illumination
    vector[labels.INDEX_ENVIRONMENT] = environment
    vector[labels.INDEX_LABEL] = label
    return vector


def _image_path(source_split: str, subject_id: str, path_kind: str, index: int) -> str:
    return f"Data/{source_split}/{subject_id}/{path_kind}/{index:06d}.jpg"


def _labels(source_split: str, subjects: Mapping[str, tuple[int, int]]) -> dict[str, list[int]]:
    """Conflict-free label mapping built from ``{subject_id: (n_live, n_spoof)}``."""
    label_vectors: dict[str, list[int]] = {}
    for subject_id, (n_live, n_spoof) in subjects.items():
        for index in range(n_live):
            path = _image_path(source_split, subject_id, labels.PATH_KIND_LIVE, index)
            label_vectors[path] = _vector(labels.LABEL_LIVE)
        for index in range(n_live, n_live + n_spoof):
            path = _image_path(source_split, subject_id, labels.PATH_KIND_SPOOF, index)
            label_vectors[path] = _vector(
                labels.LABEL_SPOOF,
                spoof_type=SYNTHETIC_SPOOF_TYPE,
                illumination=SYNTHETIC_ILLUMINATION,
                environment=SYNTHETIC_ENVIRONMENT,
            )
    return label_vectors


@pytest.fixture
def make_vector() -> Callable[..., list[int]]:
    """Factory for a 44-element annotation vector."""
    return _vector


@pytest.fixture
def image_path() -> Callable[..., str]:
    """Factory for a relative image path in the dataset layout."""
    return _image_path


@pytest.fixture
def make_labels() -> Callable[..., dict[str, list[int]]]:
    """Factory for a conflict-free label mapping."""
    return _labels


@pytest.fixture
def repo_data_config() -> DataConfig:
    """The committed ``configs/data.yaml``."""
    return load_data_config(REPO_DATA_CONFIG)


@pytest.fixture
def tmp_data_config(tmp_path: Path, repo_data_config: DataConfig) -> DataConfig:
    """The committed config with every path redirected into ``tmp_path``."""
    return dataclasses.replace(
        repo_data_config,
        dataset_root=tmp_path / "dataset",
        manifest_dir=tmp_path / "manifests",
        split_assignment_path=tmp_path / "splits" / "split_assignment.csv",
    )
