"""Tests for ManifestDataset, make_subset, read_manifest and the baseline transform.

Synthetic images in ``tmp_path`` only; no network (the timm test model is built without weights).
"""

import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
import timm
import torch
from PIL import Image
from timm.data import resolve_model_data_config
from torch import nn

from antispoof.data import labels
from antispoof.data.config import CONFLICT_EXCLUDE
from antispoof.data.dataset import ImageLoadError, ManifestDataset, make_subset
from antispoof.data.manifest import ManifestError, build_manifest, read_manifest
from antispoof.data.transforms import build_baseline_transform

INPUT_SIZE = 24
SUBJECTS = {"0001": (2, 2), "0002": (2, 2), "0003": (2, 2), "0004": (2, 2)}
"""16 images: 8 live and 8 spoof."""


@pytest.fixture
def tiny_model() -> nn.Module:
    return timm.create_model("test_efficientnet", pretrained=False, num_classes=1)


@pytest.fixture
def manifest(make_labels: Callable[..., dict[str, list[int]]]) -> pd.DataFrame:
    frame, _ = build_manifest(
        make_labels(labels.SPLIT_TRAIN, SUBJECTS), labels.SPLIT_TRAIN, CONFLICT_EXCLUDE
    )
    return frame


@pytest.fixture
def dataset(
    manifest: pd.DataFrame,
    tmp_path: Path,
    write_images: Callable[[pd.DataFrame, Path], None],
    tiny_model: nn.Module,
) -> ManifestDataset:
    write_images(manifest, tmp_path)
    return ManifestDataset(manifest, tmp_path, build_baseline_transform(INPUT_SIZE, tiny_model))


def _label_frame(n_live: int, n_spoof: int) -> pd.DataFrame:
    n_rows = n_live + n_spoof
    return pd.DataFrame(
        {
            "image_path": [f"img_{index:04d}.jpg" for index in range(n_rows)],
            "label": [labels.LABEL_LIVE] * n_live + [labels.LABEL_SPOOF] * n_spoof,
        }
    )


def test_items_have_shape_dtype_label_and_row_index(
    dataset: ManifestDataset, manifest: pd.DataFrame
) -> None:
    assert len(dataset) == len(SUBJECTS) * 4
    seen_labels = set()
    for index in range(len(dataset)):
        image, label, row_index = dataset[index]
        assert image.shape == (3, INPUT_SIZE, INPUT_SIZE)
        assert image.dtype == torch.float32
        assert isinstance(label, float)
        assert label == float(manifest.loc[index, "label"])
        assert row_index == index
        seen_labels.add(label)
    assert seen_labels == {0.0, 1.0}


def test_transform_normalizes_with_backbone_pretrained_stats(
    dataset: ManifestDataset, tiny_model: nn.Module
) -> None:
    stats = resolve_model_data_config(tiny_model)
    mean = torch.tensor(stats["mean"]).view(3, 1, 1)
    std = torch.tensor(stats["std"]).view(3, 1, 1)
    for index in range(len(dataset)):
        image, label, _ = dataset[index]
        pixel = 1.0 if label == labels.LABEL_SPOOF else 0.0
        expected = ((pixel - mean) / std).expand_as(image)
        torch.testing.assert_close(image, expected, atol=1e-5, rtol=0)


def test_grayscale_image_is_converted_to_rgb(
    dataset: ManifestDataset, manifest: pd.DataFrame, tmp_path: Path
) -> None:
    Image.new("L", (10, 20), 128).save(tmp_path / manifest.loc[0, "image_path"])
    image, _, _ = dataset[0]
    assert image.shape == (3, INPUT_SIZE, INPUT_SIZE)


def test_missing_image_raises_naming_the_path(
    dataset: ManifestDataset, manifest: pd.DataFrame, tmp_path: Path
) -> None:
    missing = tmp_path / manifest.loc[3, "image_path"]
    missing.unlink()
    with pytest.raises(ImageLoadError, match=re.escape(str(missing))):
        dataset[3]


def test_unreadable_image_raises_naming_the_path(
    dataset: ManifestDataset, manifest: pd.DataFrame, tmp_path: Path
) -> None:
    broken = tmp_path / manifest.loc[5, "image_path"]
    broken.write_bytes(b"not an image")
    with pytest.raises(ImageLoadError, match=re.escape(str(broken))):
        dataset[5]


def test_rejects_labels_other_than_live_and_spoof(
    manifest: pd.DataFrame, tmp_path: Path, tiny_model: nn.Module
) -> None:
    manifest.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="Labels"):
        ManifestDataset(manifest, tmp_path, build_baseline_transform(INPUT_SIZE, tiny_model))


def test_make_subset_is_proportionally_stratified_by_label() -> None:
    frame = _label_frame(n_live=30, n_spoof=70)
    subset = make_subset(frame, 10, seed=0)
    assert len(subset) == 10
    assert (subset["label"] == labels.LABEL_LIVE).sum() == 3
    assert (subset["label"] == labels.LABEL_SPOOF).sum() == 7
    assert subset["image_path"].is_unique
    assert subset["image_path"].isin(frame["image_path"]).all()
    assert subset["image_path"].is_monotonic_increasing
    assert list(subset.index) == list(range(10))


def test_make_subset_rounds_by_largest_remainder() -> None:
    # Exact shares of 4 rows: live 3 * 4/7 = 1.71, spoof 4 * 4/7 = 2.29. Floors give 1 + 2; the
    # remaining row goes to live, which has the larger remainder.
    subset = make_subset(_label_frame(n_live=3, n_spoof=4), 4, seed=0)
    assert (subset["label"] == labels.LABEL_LIVE).sum() == 2
    assert (subset["label"] == labels.LABEL_SPOOF).sum() == 2


def test_make_subset_is_deterministic_for_a_fixed_seed() -> None:
    frame = _label_frame(n_live=30, n_spoof=70)
    pd.testing.assert_frame_equal(make_subset(frame, 10, seed=7), make_subset(frame, 10, seed=7))
    assert not make_subset(frame, 10, seed=7).equals(make_subset(frame, 10, seed=8))


def test_make_subset_of_full_size_returns_every_row() -> None:
    frame = _label_frame(n_live=3, n_spoof=4)
    pd.testing.assert_frame_equal(make_subset(frame, len(frame), seed=0), frame)


@pytest.mark.parametrize("size", [0, -1, 101])
def test_make_subset_rejects_out_of_range_sizes(size: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        make_subset(_label_frame(n_live=30, n_spoof=70), size, seed=0)


def test_read_manifest_round_trips_dtypes_and_leading_zeros(
    manifest: pd.DataFrame, tmp_path: Path
) -> None:
    manifest.loc[0, "conflict"] = True
    path = tmp_path / "manifest_train.csv"
    manifest.to_csv(path, index=False)
    loaded = read_manifest(path)
    pd.testing.assert_frame_equal(loaded, manifest)
    assert loaded.loc[0, "subject_id"] == "0001"
    assert loaded["conflict"].tolist().count(True) == 1


def test_read_manifest_rejects_unexpected_columns(manifest: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "manifest_train.csv"
    manifest.drop(columns="conflict").to_csv(path, index=False)
    with pytest.raises(ManifestError, match="expected columns"):
        read_manifest(path)
