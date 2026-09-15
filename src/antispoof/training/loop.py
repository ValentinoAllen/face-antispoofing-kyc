"""Device selection, data loaders, the training epoch and the evaluation pass.

The model outputs one logit per image; ``sigmoid(logit)`` is the spoof score.
"""

import logging
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from antispoof.eval.pad_metrics import PadMetrics, pad_metrics
from antispoof.training.config import (
    DEVICE_AUTO,
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICES,
    OptimConfig,
    SubsetConfig,
    TrainConfig,
)
from antispoof.training.reproducibility import make_generator, seed_worker

logger = logging.getLogger(__name__)

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
"""A collated ``(images, labels, row_indices)`` batch from ``ManifestDataset``."""

PREDICTION_COLUMNS = (
    "image_path",
    "subject_id",
    "label",
    "spoof_type",
    "illumination",
    "environment",
)
"""Manifest columns copied into the predictions frame, followed by ``SCORE_COLUMN``."""

SCORE_COLUMN = "score"


@dataclass(frozen=True)
class EpochStats:
    """Loss and throughput of one training epoch. Wall time includes data loading."""

    steps: int
    images: int
    mean_loss: float
    wall_time_s: float
    images_per_s: float


def resolve_device(name: str) -> torch.device:
    """Map ``run.device`` to a torch device.

    Args:
        name: ``auto`` (cuda if available, else cpu), ``cpu`` or ``cuda``.

    Returns:
        The device.

    Raises:
        ValueError: If ``name`` is not one of ``DEVICES`` (MPS is not used).
        RuntimeError: If ``cuda`` is requested but not available.
    """
    if name not in DEVICES:
        raise ValueError(f"Unsupported device {name!r}; expected one of {DEVICES}.")
    if name == DEVICE_AUTO:
        return torch.device(DEVICE_CUDA if torch.cuda.is_available() else DEVICE_CPU)
    if name == DEVICE_CUDA and not torch.cuda.is_available():
        raise RuntimeError("run.device is 'cuda' but CUDA is not available.")
    return torch.device(name)


def make_loader(
    dataset: Dataset[Any], config: SubsetConfig, shuffle: bool, seed: int, device: torch.device
) -> DataLoader[Any]:
    """Build a seeded DataLoader.

    Args:
        dataset: The dataset to load.
        config: Batch size and worker count.
        shuffle: Whether to shuffle every epoch (train only).
        seed: Seeds the shuffling order and the workers.
        device: Pinned memory is used on CUDA only.

    Returns:
        The DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == DEVICE_CUDA,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )


def build_optimizer(model: nn.Module, config: OptimConfig) -> torch.optim.Optimizer:
    """Create AdamW over all model parameters with the configured lr and weight decay.

    Args:
        model: The model to optimize.
        config: Optimizer settings.

    Returns:
        The optimizer.
    """
    return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def make_grad_scaler(device: torch.device) -> torch.amp.GradScaler:
    """Return a gradient scaler that is enabled only on CUDA, where AMP is used.

    Args:
        device: The training device.

    Returns:
        The scaler; a pass-through on CPU.
    """
    return torch.amp.GradScaler(device.type, enabled=device.type == DEVICE_CUDA)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> EpochStats:
    """Run one pass over ``loader`` with ``BCEWithLogitsLoss``.

    Mixed precision (``torch.autocast`` with ``scaler``) is active only on CUDA.

    Args:
        model: Single-logit classifier.
        loader: Yields ``(images, labels, row_indices)`` batches.
        optimizer: Updated once per batch.
        scaler: From :func:`make_grad_scaler`.
        device: The training device.

    Returns:
        Step and image counts, the image-weighted mean loss, wall time and train images/sec.

    Raises:
        ValueError: If the loader yields no batches.
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    loss_sum = torch.zeros((), device=device)
    steps = images = 0
    start = time.perf_counter()
    for batch_images, batch_labels, _ in loader:
        targets = batch_labels.to(device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == DEVICE_CUDA):
            logits = model(batch_images.to(device, non_blocking=True)).squeeze(1)
        loss = criterion(logits.float(), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.detach() * targets.numel()
        steps += 1
        images += targets.numel()
    if steps == 0:
        raise ValueError("The training loader yielded no batches.")
    mean_loss = float(loss_sum.item()) / images
    wall_time_s = time.perf_counter() - start
    return EpochStats(steps, images, mean_loss, wall_time_s, images / wall_time_s)


def train_epochs(
    model: nn.Module, loader: Iterable[Batch], config: TrainConfig, device: torch.device
) -> tuple[EpochStats, ...]:
    """Train for ``optim.epochs`` epochs and log loss, wall time and throughput per epoch.

    Args:
        model: Single-logit classifier.
        loader: The training loader.
        config: The experiment config.
        device: The training device.

    Returns:
        Stats of every epoch.

    Raises:
        FloatingPointError: If an epoch's mean loss is not finite.
    """
    optimizer = build_optimizer(model, config.optim)
    scaler = make_grad_scaler(device)
    history: list[EpochStats] = []
    for epoch in range(1, config.optim.epochs + 1):
        stats = train_one_epoch(model, loader, optimizer, scaler, device)
        logger.info(
            "epoch %d/%d: mean train loss %.4f, wall time %.1f s, %.1f train images/s",
            epoch,
            config.optim.epochs,
            stats.mean_loss,
            stats.wall_time_s,
            stats.images_per_s,
        )
        if not math.isfinite(stats.mean_loss):
            raise FloatingPointError(f"Mean train loss is {stats.mean_loss} in epoch {epoch}.")
        history.append(stats)
    return tuple(history)


def predict_scores(
    model: nn.Module, loader: Iterable[Batch], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Score every image with ``sigmoid(logit)``.

    Args:
        model: Single-logit classifier.
        loader: Yields ``(images, labels, row_indices)`` batches.
        device: The evaluation device. Autocast is active only on CUDA.

    Returns:
        ``(row_indices, scores)`` in loader order.

    Raises:
        ValueError: If the loader yields no batches.
    """
    model.eval()
    row_batches: list[torch.Tensor] = []
    score_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_images, _, batch_rows in loader:
            with torch.autocast(device_type=device.type, enabled=device.type == DEVICE_CUDA):
                logits = model(batch_images.to(device, non_blocking=True)).squeeze(1)
            score_batches.append(torch.sigmoid(logits.float()).cpu())
            row_batches.append(batch_rows)
    if not score_batches:
        raise ValueError("The evaluation loader yielded no batches.")
    scores = torch.cat(score_batches).numpy().astype(np.float64)
    return torch.cat(row_batches).numpy(), scores


def predictions_frame(
    manifest: pd.DataFrame, row_indices: np.ndarray, scores: np.ndarray
) -> pd.DataFrame:
    """Join scores back to the manifest metadata by positional row index.

    Args:
        manifest: The frame the evaluated dataset was built from.
        row_indices: Positional row index of each score.
        scores: Spoof score per image.

    Returns:
        ``PREDICTION_COLUMNS`` plus ``score``, one row per manifest row, in manifest order.

    Raises:
        ValueError: If the row indices do not cover every manifest row exactly once.
    """
    order = np.argsort(row_indices, kind="stable")
    if not np.array_equal(row_indices[order], np.arange(len(manifest))):
        raise ValueError("Predictions must cover every manifest row exactly once.")
    frame = manifest.iloc[row_indices[order]][list(PREDICTION_COLUMNS)].reset_index(drop=True)
    frame[SCORE_COLUMN] = scores[order]
    return frame


def evaluate(
    model: nn.Module,
    loader: Iterable[Batch],
    manifest: pd.DataFrame,
    device: torch.device,
    threshold: float,
) -> tuple[PadMetrics, pd.DataFrame]:
    """Score every image and compute pooled PAD metrics at ``threshold``.

    Args:
        model: Single-logit classifier.
        loader: Loader over a ``ManifestDataset`` built from ``manifest``.
        manifest: The evaluated manifest rows.
        device: The evaluation device.
        threshold: A score ``>= threshold`` is classified as spoof.

    Returns:
        The metrics and the predictions frame (:func:`predictions_frame`).

    Raises:
        PadMetricsError: If the metrics are undefined, e.g. one class is absent.
    """
    row_indices, scores = predict_scores(model, loader, device)
    predictions = predictions_frame(manifest, row_indices, scores)
    metrics = pad_metrics(
        predictions["label"].to_numpy(), predictions[SCORE_COLUMN].to_numpy(), threshold
    )
    return metrics, predictions
