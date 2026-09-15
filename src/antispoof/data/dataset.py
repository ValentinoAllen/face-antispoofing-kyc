"""PyTorch dataset over a manifest (``docs/SCHEMA.md`` §1) and seeded, label-stratified subsets."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from antispoof.data import labels
from antispoof.data.splits import _stratum_quotas

ImageTransform = Callable[[Image.Image], torch.Tensor]
"""Maps a decoded RGB image to a model input tensor."""

DatasetItem = tuple[torch.Tensor, float, int]
"""``(image, label, row_index)`` as returned by :class:`ManifestDataset`."""


class ImageLoadError(RuntimeError):
    """Raised when an image listed in a manifest is missing or cannot be decoded."""


def load_rgb_image(path: Path) -> Image.Image:
    """Decode an image file and convert it to RGB.

    Args:
        path: Image file path.

    Returns:
        The fully decoded image in RGB mode.

    Raises:
        ImageLoadError: If the file does not exist or cannot be decoded. The message names the path.
    """
    if not path.is_file():
        raise ImageLoadError(f"Image file not found: {path}")
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, ValueError) as error:
        raise ImageLoadError(f"Cannot decode image {path}: {error}") from error


class ManifestDataset(Dataset[DatasetItem]):
    """Images and live/spoof labels of the rows of a manifest.

    ``image_path`` holds the label-file key, a POSIX path relative to the dataset root
    (``docs/SCHEMA.md`` §1.3), so each file is read from ``dataset_root / image_path``.

    Each item is ``(image, label, row_index)``: the transformed RGB image, the label as a float
    (``1.0`` = spoof, label index 43), and the item's positional row in ``manifest``. The row index
    lets evaluation join predictions back to the manifest metadata.
    """

    def __init__(
        self, manifest: pd.DataFrame, dataset_root: Path, transform: ImageTransform
    ) -> None:
        """Store the image paths and labels of ``manifest``.

        Args:
            manifest: Frame with at least ``image_path`` and ``label`` columns.
            dataset_root: Directory the ``image_path`` values are relative to.
            transform: Applied to every decoded RGB image.

        Raises:
            ValueError: If a required column is missing or a label is not live or spoof.
        """
        missing = {"image_path", "label"} - set(manifest.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns {sorted(missing)}.")
        unknown = set(manifest["label"].unique()) - set(labels.LABEL_VALUES)
        if unknown:
            raise ValueError(f"Labels {sorted(unknown)} are not in {labels.LABEL_VALUES}.")
        self._image_paths: list[str] = manifest["image_path"].tolist()
        self._labels: list[float] = manifest["label"].astype(float).tolist()
        self._dataset_root = dataset_root
        self._transform = transform

    def __len__(self) -> int:
        """Return the number of manifest rows."""
        return len(self._image_paths)

    def __getitem__(self, index: int) -> DatasetItem:
        """Load, convert and transform one image.

        Args:
            index: Positional row index in the manifest.

        Returns:
            ``(image, label, index)``.

        Raises:
            ImageLoadError: If the image is missing or cannot be decoded.
        """
        image = load_rgb_image(self._dataset_root / self._image_paths[index])
        return self._transform(image), self._labels[index], index


def make_subset(manifest: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw a seeded sample of ``n`` manifest rows, stratified by ``label``.

    Each label receives a share of ``n`` proportional to its row count, rounded by largest
    remainder (the allocation used by the subject-level split). Its rows are drawn with
    ``numpy.random.default_rng(seed)``. Selected rows keep their original order.

    Subsetting a subject-disjoint manifest cannot create leakage; callers still run
    ``validate_splits`` on the subsets before training.

    Args:
        manifest: Frame with a ``label`` column.
        n: Number of rows to draw, in ``[1, len(manifest)]``.
        seed: Seed for the generator.

    Returns:
        The sampled rows with a fresh ``RangeIndex``.

    Raises:
        ValueError: If ``n`` is outside ``[1, len(manifest)]``.
    """
    if not 1 <= n <= len(manifest):
        raise ValueError(f"Subset size {n} is outside [1, {len(manifest)}].")
    label_of_row = manifest["label"].to_numpy()
    label_values = np.unique(label_of_row)
    sizes = np.array([np.count_nonzero(label_of_row == value) for value in label_values])
    quotas = _stratum_quotas(sizes, n / len(manifest), n)
    rng = np.random.default_rng(seed)
    chosen = [
        np.flatnonzero(label_of_row == value)[rng.permutation(size)[:quota]]
        for value, size, quota in zip(label_values, sizes, quotas, strict=True)
    ]
    positions = np.sort(np.concatenate(chosen))
    return manifest.iloc[positions].reset_index(drop=True)
