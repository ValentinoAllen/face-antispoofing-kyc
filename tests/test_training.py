"""End-to-end smoke, run-directory and overfit tests on synthetic images (CPU, no network)."""

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
import torch

from antispoof.data import labels
from antispoof.data.build import MANIFEST_FILENAME, SplitOutputs, write_outputs
from antispoof.data.config import CONFLICT_EXCLUDE, DataConfig
from antispoof.data.dataset import ImageLoadError, ManifestDataset
from antispoof.data.manifest import build_manifest
from antispoof.data.splits import SplitLeakageError
from antispoof.data.transforms import build_baseline_transform
from antispoof.models.factory import build_model
from antispoof.training import reproducibility
from antispoof.training.config import TrainConfig, load_train_config
from antispoof.training.loop import (
    PREDICTION_COLUMNS,
    SCORE_COLUMN,
    evaluate,
    make_grad_scaler,
    make_loader,
    train_one_epoch,
)
from antispoof.training.run import (
    CHECKPOINT_FILENAME,
    PREDICTIONS_FILENAME,
    RECORD_FILENAME,
    RESOLVED_CONFIG_FILENAME,
    RunInputs,
    format_ledger_row,
    load_subsets,
    run_baseline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = REPO_ROOT / "configs" / "baseline.yaml"
CPU = torch.device("cpu")

TRAIN_SUBJECTS = {"0001": (2, 2), "0002": (2, 2)}
VAL_SUBJECTS = {"0003": (2, 2), "0004": (2, 2)}
TEST_SUBJECT = "0005"
IMAGES_PER_SPLIT = 8
TINY_BATCH = 4

CONTRACT_FIELDS = {
    "run_id",
    "created_at",
    "hypothesis",
    "config_path",
    "config_hash",
    "git_sha",
    "git_dirty",
    "seed",
    "split_name",
    "split_sha256",
    "manifest_sha256",
    "environment",
    "status",
    "eval_split",
    "threshold",
    "threshold_rule",
    "metrics",
    "artifacts",
    "notes",
}
CONTRACT_METRICS = {
    "apcer_max",
    "apcer_per_species",
    "bpcer",
    "acer",
    "bpcer_at_apcer_1pct",
    "n_bona_fide",
    "n_attack",
}

OVERFIT_STEPS = 30
OVERFIT_LR = 1e-3
OVERFIT_MAX_FINAL_LOSS_FRACTION = 0.2
"""The last epoch's loss must be below this fraction of the first epoch's loss."""
OVERFIT_MAX_FINAL_LOSS = 0.05
"""Absolute bound on the last loss, because a random init can already start with a small loss."""


@pytest.fixture
def synthetic_data(
    tmp_data_config: DataConfig,
    make_labels: Callable[..., dict[str, list[int]]],
    write_images: Callable[[pd.DataFrame, Path], None],
) -> DataConfig:
    """Train and val manifests, their images, and a split assignment, all under ``tmp_path``."""
    manifests: dict[str, pd.DataFrame] = {}
    for split, subjects in ((labels.SPLIT_TRAIN, TRAIN_SUBJECTS), (labels.SPLIT_VAL, VAL_SUBJECTS)):
        vectors = make_labels(labels.SPLIT_TRAIN, subjects)
        manifest, _ = build_manifest(vectors, labels.SPLIT_TRAIN, CONFLICT_EXCLUDE)
        manifest["split"] = split
        manifests[split] = manifest
        write_images(manifest, tmp_data_config.dataset_root)
    assignment = pd.DataFrame(
        [(subject, labels.SPLIT_TRAIN) for subject in TRAIN_SUBJECTS]
        + [(subject, labels.SPLIT_VAL) for subject in VAL_SUBJECTS]
        + [(TEST_SUBJECT, labels.SPLIT_TEST)],
        columns=["subject_id", "split"],
    )
    outputs = SplitOutputs(manifests=manifests, assignment=assignment)
    write_outputs(outputs, tmp_data_config.manifest_dir, tmp_data_config.split_assignment_path)
    return tmp_data_config


def _tiny_config() -> TrainConfig:
    """The committed baseline with a tiny untrained model, small inputs, CPU and no workers."""
    config = load_train_config(BASELINE_CONFIG)
    return dataclasses.replace(
        config,
        run=dataclasses.replace(config.run, device="cpu"),
        model=dataclasses.replace(
            config.model, backbone="test_efficientnet", pretrained=False, input_size=32
        ),
        data=dataclasses.replace(
            config.data,
            train_subset=IMAGES_PER_SPLIT,
            val_subset=IMAGES_PER_SPLIT,
            batch_size=TINY_BATCH,
            num_workers=0,
        ),
    )


def _run_inputs(data_config: DataConfig, output_dir: Path) -> RunInputs:
    return RunInputs(
        train_config=_tiny_config(),
        config_path=BASELINE_CONFIG,
        data_config=data_config,
        output_dir=output_dir,
        repo_root=REPO_ROOT,
    )


def test_two_training_steps_then_evaluate(synthetic_data: DataConfig) -> None:
    config = _tiny_config()
    subsets = load_subsets(config, synthetic_data)
    reproducibility.seed_everything(config.run.seed)
    model = build_model(config.model)
    transform = build_baseline_transform(config.model.input_size, model)
    loaders = {
        split: make_loader(
            ManifestDataset(frame, synthetic_data.dataset_root, transform),
            config.data,
            shuffle=split == labels.SPLIT_TRAIN,
            seed=config.run.seed,
            device=CPU,
        )
        for split, frame in subsets.items()
    }
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optim.lr)
    stats = train_one_epoch(model, loaders["train"], optimizer, make_grad_scaler(CPU), CPU)
    assert (stats.steps, stats.images) == (2, IMAGES_PER_SPLIT)
    assert math.isfinite(stats.mean_loss)
    assert stats.images_per_s > 0
    after = list(model.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))

    metrics, predictions = evaluate(
        model, loaders["val"], subsets["val"], CPU, config.eval.threshold
    )
    assert list(predictions.columns) == [*PREDICTION_COLUMNS, SCORE_COLUMN]
    assert len(predictions) == IMAGES_PER_SPLIT
    assert predictions[SCORE_COLUMN].between(0.0, 1.0).all()
    assert set(predictions["subject_id"]) == set(VAL_SUBJECTS)
    assert (metrics.n_attack, metrics.n_bona_fide) == (4, 4)


def test_run_baseline_writes_record_predictions_and_checkpoint(
    synthetic_data: DataConfig, tmp_path: Path
) -> None:
    result = run_baseline(_run_inputs(synthetic_data, tmp_path / "runs"))
    run_dir = result.run_dir
    assert run_dir.parent == tmp_path / "runs"
    for name in (RECORD_FILENAME, PREDICTIONS_FILENAME, CHECKPOINT_FILENAME):
        assert (run_dir / name).is_file()

    record = json.loads((run_dir / RECORD_FILENAME).read_text(encoding="utf-8"))
    assert record == result.record
    assert set(record) >= CONTRACT_FIELDS
    assert set(record["metrics"]) >= CONTRACT_METRICS
    assert (
        re.fullmatch(r"\d{8}-\d{6}-baseline", record["run_id"]) and record["run_id"] == run_dir.name
    )
    assert (record["status"], record["eval_split"]) == ("completed", "val")
    assert (record["threshold"], record["threshold_rule"]) == (0.5, "fixed_config")
    assert record["config_path"] == "configs/baseline.yaml"
    assert re.fullmatch(r"[0-9a-f]{40}", record["git_sha"])
    split_bytes = synthetic_data.split_assignment_path.read_bytes()
    assert record["split_sha256"] == hashlib.sha256(split_bytes).hexdigest()
    manifest_names = [MANIFEST_FILENAME.format(split=split) for split in ("train", "val")]
    assert record["manifest_sha256"] == {
        name: hashlib.sha256((synthetic_data.manifest_dir / name).read_bytes()).hexdigest()
        for name in manifest_names
    }
    resolved = json.loads((run_dir / RESOLVED_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert record["config_hash"] == reproducibility.config_hash(resolved)
    assert resolved["experiment"]["model"]["backbone"] == "test_efficientnet"

    metrics = record["metrics"]
    assert metrics["apcer_max"] is None and metrics["acer"] is None
    assert metrics["apcer_per_species"] is None and metrics["bpcer_at_apcer_1pct"] is None
    assert (metrics["n_attack"], metrics["n_bona_fide"]) == (4, 4)
    assert metrics["apcer_pooled"] == metrics["n_attack_accepted"] / 4
    assert metrics["bpcer"] == metrics["n_bona_fide_rejected"] / 4
    assert math.isclose(metrics["acer_pooled"], (metrics["apcer_pooled"] + metrics["bpcer"]) / 2)
    assert record["artifacts"]["checkpoint"] == (run_dir / CHECKPOINT_FILENAME).as_posix()
    assert len(record["training_epochs"]) == 1

    predictions = pd.read_csv(run_dir / PREDICTIONS_FILENAME, dtype={"subject_id": str})
    assert list(predictions.columns) == [*PREDICTION_COLUMNS, SCORE_COLUMN]
    assert len(predictions) == IMAGES_PER_SPLIT
    cells = format_ledger_row(record).strip("|").split(" | ")
    assert len(cells) == 11
    assert cells[9] == "—"


def test_run_baseline_marks_the_record_failed_when_an_image_is_missing(
    synthetic_data: DataConfig, tmp_path: Path
) -> None:
    val_manifest = pd.read_csv(
        synthetic_data.manifest_dir / MANIFEST_FILENAME.format(split=labels.SPLIT_VAL)
    )
    (synthetic_data.dataset_root / val_manifest.loc[0, "image_path"]).unlink()
    output_dir = tmp_path / "runs"
    with pytest.raises(ImageLoadError, match="Image file not found"):
        run_baseline(_run_inputs(synthetic_data, output_dir))
    (record_path,) = output_dir.glob(f"*/{RECORD_FILENAME}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error"].startswith("ImageLoadError")


def test_load_subsets_rejects_manifests_that_disagree_with_the_assignment(
    synthetic_data: DataConfig,
) -> None:
    assignment_path = synthetic_data.split_assignment_path
    assignment = pd.read_csv(assignment_path, dtype=str)
    assignment.loc[assignment["subject_id"] == "0003", "split"] = labels.SPLIT_TRAIN
    assignment.to_csv(assignment_path, index=False)
    with pytest.raises(SplitLeakageError, match="0003"):
        load_subsets(_tiny_config(), synthetic_data)


def test_loss_drops_when_overfitting_black_versus_white(synthetic_data: DataConfig) -> None:
    config = _tiny_config()
    train = load_subsets(config, synthetic_data)[labels.SPLIT_TRAIN]
    reproducibility.seed_everything(config.run.seed)
    model = build_model(config.model)
    dataset = ManifestDataset(
        train, synthetic_data.dataset_root, build_baseline_transform(config.model.input_size, model)
    )
    full_batch = dataclasses.replace(config.data, batch_size=IMAGES_PER_SPLIT)
    loader = make_loader(dataset, full_batch, shuffle=True, seed=config.run.seed, device=CPU)
    optimizer = torch.optim.AdamW(model.parameters(), lr=OVERFIT_LR)
    scaler = make_grad_scaler(CPU)
    losses = [
        train_one_epoch(model, loader, optimizer, scaler, CPU).mean_loss
        for _ in range(OVERFIT_STEPS)
    ]
    assert losses[-1] < OVERFIT_MAX_FINAL_LOSS_FRACTION * losses[0], losses
    assert losses[-1] < OVERFIT_MAX_FINAL_LOSS, losses
