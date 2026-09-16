"""Tests for the counterfactual re-encoding evaluation on synthetic images (CPU, no network).

Each end-to-end test first trains a tiny baseline with ``run_baseline`` to get a real run folder.
"""

import dataclasses
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from antispoof.data import labels
from antispoof.data.build import SplitOutputs, write_outputs
from antispoof.data.config import CONFLICT_EXCLUDE, DataConfig
from antispoof.data.dataset import ImageTransform, ManifestDataset, load_rgb_image
from antispoof.data.manifest import build_manifest
from antispoof.data.transforms import build_baseline_transform
from antispoof.eval import counterfactual as cf
from antispoof.eval import jpeg_tables
from antispoof.eval.pad_metrics import PadMetrics
from antispoof.models.factory import build_model
from antispoof.training import reproducibility
from antispoof.training.config import TrainConfig, load_train_config
from antispoof.training.run import (
    CHECKPOINT_FILENAME,
    RECORD_FILENAME,
    RESOLVED_CONFIG_FILENAME,
    RunInputs,
    format_ledger_row,
    load_subsets,
    run_baseline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = REPO_ROOT / "configs" / "baseline.yaml"
COUNTERFACTUAL_CONFIG = REPO_ROOT / "configs" / "counterfactual_jpeg.yaml"

HYPOTHESIS = (
    "The baseline CNN relies on the JPEG-encoding trace of attack images. Val subset, threshold "
    "0.5, live images unchanged. Validity: reencode_same pooled APCER must be <= 5%, otherwise the "
    "test is inconclusive. Then on reencode_live_quality pooled APCER: >= 20% strong reliance; "
    "<= 5% no evidence of reliance; otherwise partial. reencode_live_quality_and_size is "
    "secondary, same bands."
)
WHAT_CHANGED = "Counterfactual eval: attack images re-encoded at evaluation time, no retraining"
NOTES = (
    "same val subset and threshold as 20260915-153606-baseline; checkpoint from a rerun of "
    "configs/baseline.yaml"
)

TRAIN_SUBJECTS = {"0001": (2, 2), "0002": (2, 2)}
VAL_SUBJECTS = {"0003": (2, 2), "0004": (2, 2)}
TEST_SUBJECT = "0005"
ROWS_PER_SPLIT = 8
N_SPOOF_VAL = 4
LIVE_QUALITY = 75
SPOOF_QUALITY = 95
SPOOF_SIZE = (30, 40)
LIVE_WIDTHS = (20, 24, 28)
LIVE_HEIGHT = 36


def _live_size(position: int) -> tuple[int, int]:
    return LIVE_WIDTHS[position % len(LIVE_WIDTHS)], LIVE_HEIGHT


def _write_noise_images(manifest: pd.DataFrame, root: Path) -> None:
    """Seeded noise images: live at LIVE_QUALITY with varying sizes, spoof at SPOOF_QUALITY."""
    for position, (image_path, label) in enumerate(
        zip(manifest["image_path"], manifest["label"], strict=True)
    ):
        is_spoof = label == labels.LABEL_SPOOF
        width, height = SPOOF_SIZE if is_spoof else _live_size(position)
        rng = np.random.default_rng(position)
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        path = root / image_path
        path.parent.mkdir(parents=True, exist_ok=True)
        quality = SPOOF_QUALITY if is_spoof else LIVE_QUALITY
        Image.fromarray(pixels, mode="RGB").save(path, "JPEG", quality=quality, subsampling=2)


@pytest.fixture
def cf_data(
    tmp_data_config: DataConfig, make_labels: Callable[..., dict[str, list[int]]]
) -> DataConfig:
    """Train and val manifests, their noise JPEGs, and a split assignment, under ``tmp_path``."""
    manifests: dict[str, pd.DataFrame] = {}
    for split, subjects in ((labels.SPLIT_TRAIN, TRAIN_SUBJECTS), (labels.SPLIT_VAL, VAL_SUBJECTS)):
        vectors = make_labels(labels.SPLIT_TRAIN, subjects)
        manifest, _ = build_manifest(vectors, labels.SPLIT_TRAIN, CONFLICT_EXCLUDE)
        manifest["split"] = split
        manifests[split] = manifest
        _write_noise_images(manifest, tmp_data_config.dataset_root)
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
            train_subset=ROWS_PER_SPLIT,
            val_subset=ROWS_PER_SPLIT,
            batch_size=4,
            num_workers=0,
        ),
    )


@pytest.fixture
def source_run_dir(cf_data: DataConfig, tmp_path: Path) -> Path:
    """A completed tiny baseline run folder trained on ``cf_data``."""
    inputs = RunInputs(_tiny_config(), BASELINE_CONFIG, cf_data, tmp_path / "baseline", REPO_ROOT)
    return run_baseline(inputs).run_dir


def _inputs(
    data_config: DataConfig,
    run_dir: Path,
    output_dir: Path,
    baseline_config: TrainConfig | None = None,
) -> cf.CounterfactualInputs:
    return cf.CounterfactualInputs(
        config=cf.load_counterfactual_config(COUNTERFACTUAL_CONFIG),
        config_path=COUNTERFACTUAL_CONFIG,
        baseline_config=_tiny_config() if baseline_config is None else baseline_config,
        baseline_config_path=BASELINE_CONFIG,
        data_config=data_config,
        source_run_dir=run_dir,
        output_dir=output_dir,
        repo_root=REPO_ROOT,
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _edit_record(run_dir: Path, name: str, change: Callable[[dict[str, Any]], None]) -> None:
    document = _read_json(run_dir / name)
    change(document)
    (run_dir / name).write_text(json.dumps(document), encoding="utf-8")


def test_committed_config_has_the_agreed_fields() -> None:
    config = cf.load_counterfactual_config(COUNTERFACTUAL_CONFIG)
    assert (config.run.hypothesis, config.run.what_changed, config.run.notes) == (
        HYPOTHESIS,
        WHAT_CHANGED,
        NOTES,
    )
    assert config.run.seed == 42
    assert config.counterfactual.arms == cf.KNOWN_ARMS
    assert config.counterfactual.resize_filter == "bicubic"
    assert config.baseline.config == "configs/baseline.yaml"
    baseline = load_train_config(REPO_ROOT / config.baseline.config)
    assert config.eval == baseline.eval


@pytest.mark.parametrize(
    ("arms", "resize_filter", "message"),
    [
        (["identity", "reencode_same", "reencode_live_quality", "crop"], "bicubic", "must be in"),
        (["identity", "reencode_live_quality"], "bicubic", "must include"),
        (["reencode_same", "identity", "reencode_live_quality"], "bicubic", "start with identity"),
        (
            ["identity", "reencode_same", "reencode_same", "reencode_live_quality"],
            "bicubic",
            "must not repeat",
        ),
        (["identity", "reencode_same", "reencode_live_quality"], "sharpest", "resize_filter"),
    ],
)
def test_invalid_arms_are_rejected(
    tmp_path: Path, arms: list[str], resize_filter: str, message: str
) -> None:
    document = yaml.safe_load(COUNTERFACTUAL_CONFIG.read_text(encoding="utf-8"))
    document["counterfactual"] = {"arms": arms, "resize_filter": resize_filter}
    path = tmp_path / "counterfactual.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(cf.CounterfactualConfigError, match=message):
        cf.load_counterfactual_config(path)


def _arm_datasets(
    data_config: DataConfig,
) -> tuple[pd.DataFrame, ImageTransform, dict[str, cf.CounterfactualDataset]]:
    config = _tiny_config()
    subsets = load_subsets(config, data_config)
    headers = jpeg_tables.read_subset_headers(subsets, data_config.dataset_root)
    encodings = jpeg_tables.class_encodings(jpeg_tables.audit_quantization(headers))
    sizes = cf.draw_live_sizes(headers, N_SPOOF_VAL, seed=0)
    transform = build_baseline_transform(config.model.input_size, build_model(config.model))
    val = subsets[labels.SPLIT_VAL]
    datasets = {
        arm: cf.CounterfactualDataset(
            val,
            data_config.dataset_root,
            transform,
            cf.arm_edits(arm, val, encodings, sizes, "bicubic"),
        )
        for arm in cf.KNOWN_ARMS
    }
    return val, transform, datasets


def test_live_rows_are_unchanged_and_spoof_rows_are_reencoded(cf_data: DataConfig) -> None:
    val, transform, datasets = _arm_datasets(cf_data)
    root = cf_data.dataset_root
    before = {path: hashlib.sha256((root / path).read_bytes()).digest() for path in val.image_path}
    identity = datasets[cf.ARM_IDENTITY]
    reference = ManifestDataset(val, root, transform)
    for index, (path, label) in enumerate(zip(val["image_path"], val["label"], strict=True)):
        original = load_rgb_image(root / path)
        assert torch.equal(identity[index][0], reference[index][0])
        for arm in cf.KNOWN_ARMS:
            image = datasets[arm].load_image(index)
            if label == labels.LABEL_LIVE or arm == cf.ARM_IDENTITY:
                assert image.tobytes() == original.tobytes(), (arm, path)
            elif arm == cf.ARM_REENCODE_LIVE_QUALITY:
                assert image.size == original.size
                assert image.tobytes() != original.tobytes()
    assert before == {
        path: hashlib.sha256((root / path).read_bytes()).digest() for path in val.image_path
    }
    size_arm = datasets[cf.ARM_REENCODE_LIVE_QUALITY_AND_SIZE]
    spoof_rows = np.flatnonzero(val["label"] == labels.LABEL_SPOOF)
    assert {size_arm.load_image(int(row)).size for row in spoof_rows} <= set(
        _live_size(position) for position in range(ROWS_PER_SPLIT)
    )


def test_edits_on_live_rows_are_rejected(cf_data: DataConfig) -> None:
    val = load_subsets(_tiny_config(), cf_data)[labels.SPLIT_VAL]
    live_row = int(np.flatnonzero(val["label"] == labels.LABEL_LIVE)[0])
    edit = cf.SpoofEdit(jpeg_tables.Reencoding(LIVE_QUALITY, 2))
    with pytest.raises(ValueError, match="spoof rows only"):
        cf.CounterfactualDataset(val, cf_data.dataset_root, torch.as_tensor, {live_row: edit})


def test_live_sizes_are_drawn_with_a_seed_from_live_train_rows(cf_data: DataConfig) -> None:
    subsets = load_subsets(_tiny_config(), cf_data)
    headers = jpeg_tables.read_subset_headers(subsets, cf_data.dataset_root)
    first = cf.draw_live_sizes(headers, 50, seed=7)
    assert first == cf.draw_live_sizes(headers, 50, seed=7)
    train = subsets[labels.SPLIT_TRAIN]
    live_paths = set(train.loc[train["label"] == labels.LABEL_LIVE, "image_path"])
    assert {size.image_path for size in first} == live_paths
    for size in first:
        assert (size.width, size.height) == Image.open(cf_data.dataset_root / size.image_path).size


def test_run_counterfactual_writes_the_run_directory(
    cf_data: DataConfig, source_run_dir: Path, tmp_path: Path
) -> None:
    result = cf.run_counterfactual(_inputs(cf_data, source_run_dir, tmp_path / "runs"))
    run_dir, summary = result.run_dir, result.summary
    record = _read_json(run_dir / RECORD_FILENAME)
    source = _read_json(source_run_dir / RECORD_FILENAME)
    assert record == result.record
    assert (record["status"], record["eval_split"], record["threshold"]) == (
        "completed",
        "val",
        0.5,
    )
    assert record["run_id"].endswith("-counterfactual_jpeg")
    assert record["config_path"] == "configs/counterfactual_jpeg.yaml"
    assert record["training_epochs"] is None and record["artifacts"]["checkpoint"] is None
    assert record["metrics_arm"] == cf.ARM_REENCODE_LIVE_QUALITY
    checkpoint = source_run_dir / CHECKPOINT_FILENAME
    assert record["source_run"] == {
        "run_id": source["run_id"],
        "config_path": "configs/baseline.yaml",
        "config_hash": source["config_hash"],
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": reproducibility.sha256_file(checkpoint),
    }
    resolved = _read_json(run_dir / RESOLVED_CONFIG_FILENAME)
    assert record["config_hash"] == reproducibility.config_hash(resolved)
    assert resolved["baseline"]["config_hash"] == source["config_hash"]

    assert _read_json(run_dir / cf.SUMMARY_FILENAME) == summary
    assert summary["class_encodings"] == {
        "live": {"quality": LIVE_QUALITY, "subsampling": "4:2:0"},
        "spoof": {"quality": SPOOF_QUALITY, "subsampling": "4:2:0"},
    }
    assert list(summary["arms"]) == list(cf.KNOWN_ARMS)
    arms = {arm: entry["pad_metrics"] for arm, entry in summary["arms"].items()}
    for key in cf.COUNT_KEYS:
        assert arms[cf.ARM_IDENTITY][key] == source["metrics"][key]
    assert {metrics["bpcer"] for metrics in arms.values()} == {source["metrics"]["bpcer"]}
    live_metrics = arms[cf.ARM_REENCODE_LIVE_QUALITY]
    assert record["metrics"]["apcer_pooled"] == live_metrics["apcer"]
    assert record["metrics"]["n_attack_accepted"] == live_metrics["n_attack_accepted"]

    for arm in cf.KNOWN_ARMS:
        predictions = pd.read_csv(run_dir / f"predictions_{arm}.csv")
        assert len(predictions) == ROWS_PER_SPLIT
        spoof_rows = predictions["label"] == labels.LABEL_SPOOF
        expected = spoof_rows & (arm != cf.ARM_IDENTITY)
        assert predictions["reencode_quality"].notna().tolist() == expected.tolist(), arm
    size_predictions = pd.read_csv(run_dir / "predictions_reencode_live_quality_and_size.csv")
    spoof = size_predictions[size_predictions["label"] == labels.LABEL_SPOOF]
    assert set(spoof["reencode_quality"]) == {LIVE_QUALITY}
    assert spoof["size_source_image_path"].notna().all()

    cells = format_ledger_row(record).strip("|").split(" | ")
    assert len(cells) == 11
    assert cells[6].endswith("(pooled)") and cells[8].endswith("(pooled)")
    report = cf.format_counterfactual_report(result)
    assert "Quantization-table audit (headers only, no pixels decoded):" in report
    assert "q=75: luma mean 29.03125" in report and "q=95: luma mean 5.765625" in report
    assert all(arm in report for arm in cf.KNOWN_ARMS)


def test_run_dir_config_hash_must_match_the_baseline_config(
    cf_data: DataConfig, source_run_dir: Path, tmp_path: Path
) -> None:
    source = cf.load_source_run(source_run_dir)
    tiny = _tiny_config()
    expected = cf.check_source_config_hash(source.record, source.resolved, tiny, cf_data)
    assert expected == source.record["config_hash"]
    committed = load_train_config(BASELINE_CONFIG)
    with pytest.raises(cf.CounterfactualInputError, match="must match the committed"):
        cf.check_source_config_hash(source.record, source.resolved, committed, cf_data)
    tampered = json.loads(json.dumps(source.resolved))
    tampered["experiment"]["optim"]["lr"] = 1.0
    with pytest.raises(cf.CounterfactualInputError, match="must match the committed"):
        cf.check_source_config_hash(source.record, tampered, tiny, cf_data)

    output_dir = tmp_path / "runs"
    with pytest.raises(cf.CounterfactualInputError, match="must match the committed"):
        cf.run_counterfactual(_inputs(cf_data, source_run_dir, output_dir, committed))
    _edit_record(
        source_run_dir,
        RESOLVED_CONFIG_FILENAME,
        lambda document: document["experiment"]["optim"].update(lr=1.0),
    )
    with pytest.raises(cf.CounterfactualInputError, match="must match the committed"):
        cf.run_counterfactual(_inputs(cf_data, source_run_dir, output_dir))
    assert not output_dir.exists()


def test_incomplete_run_dir_and_foreign_checkpoint_are_rejected(
    cf_data: DataConfig, source_run_dir: Path, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(cf.CounterfactualInputError, match="missing"):
        cf.load_source_run(empty)
    checkpoint_path = source_run_dir / CHECKPOINT_FILENAME
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    torch.save({**checkpoint, "run_id": "20260101-000000-baseline"}, checkpoint_path)
    output_dir = tmp_path / "runs"
    with pytest.raises(cf.CounterfactualInputError, match="belongs to run"):
        cf.run_counterfactual(_inputs(cf_data, source_run_dir, output_dir))
    assert not output_dir.exists()


def _single_failed_record(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    (run_dir,) = output_dir.iterdir()
    return _read_json(run_dir / RECORD_FILENAME), _read_json(run_dir / cf.SUMMARY_FILENAME)


def test_identity_mismatch_fails_the_run(
    cf_data: DataConfig, source_run_dir: Path, tmp_path: Path
) -> None:
    _edit_record(
        source_run_dir,
        RECORD_FILENAME,
        lambda record: record["metrics"].update(
            n_attack_accepted=record["metrics"]["n_attack_accepted"] + 1
        ),
    )
    output_dir = tmp_path / "runs"
    with pytest.raises(cf.CounterfactualCheckError, match="identity arm must reproduce"):
        cf.run_counterfactual(_inputs(cf_data, source_run_dir, output_dir))
    record, summary = _single_failed_record(output_dir)
    assert record["status"] == "failed"
    assert record["error"].startswith("CounterfactualCheckError")
    assert summary["quantization_audit"] is not None and summary["arms"] == {}


def test_changed_bpcer_fails_the_run(
    cf_data: DataConfig,
    source_run_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = PadMetrics(0.5, 0.25, 0.25, 0.25, 4, 4, 1, 1)
    shifted = dataclasses.replace(identity, bpcer=0.5, n_bona_fide_rejected=2)
    cf.check_bpcer_unchanged("reencode_same", dataclasses.replace(identity, apcer=1.0), identity)
    with pytest.raises(cf.CounterfactualCheckError, match="changed BPCER: 2/4"):
        cf.check_bpcer_unchanged("reencode_same", shifted, identity)

    original = cf.evaluate
    arms_evaluated: list[PadMetrics] = []

    def shift_after_identity(*args: Any, **kwargs: Any) -> tuple[PadMetrics, pd.DataFrame]:
        metrics, predictions = original(*args, **kwargs)
        arms_evaluated.append(metrics)
        if len(arms_evaluated) > 1:
            metrics = dataclasses.replace(
                metrics, n_bona_fide_rejected=metrics.n_bona_fide_rejected + 1
            )
        return metrics, predictions

    monkeypatch.setattr(cf, "evaluate", shift_after_identity)
    output_dir = tmp_path / "runs"
    with pytest.raises(cf.CounterfactualCheckError, match="'reencode_same' changed BPCER"):
        cf.run_counterfactual(_inputs(cf_data, source_run_dir, output_dir))
    record, summary = _single_failed_record(output_dir)
    assert record["status"] == "failed"
    assert list(summary["arms"]) == [cf.ARM_IDENTITY]


def test_more_than_one_quality_in_a_class_fails_the_run(
    cf_data: DataConfig, source_run_dir: Path, tmp_path: Path
) -> None:
    val = load_subsets(_tiny_config(), cf_data)[labels.SPLIT_VAL]
    spoof_path = val.loc[val["label"] == labels.LABEL_SPOOF, "image_path"].iloc[0]
    image = load_rgb_image(cf_data.dataset_root / spoof_path)
    image.save(cf_data.dataset_root / spoof_path, "JPEG", quality=90, subsampling=2)
    output_dir = tmp_path / "runs"
    with pytest.raises(jpeg_tables.QualityAuditError, match="spoof has 2 distinct table sets"):
        cf.run_counterfactual(_inputs(cf_data, source_run_dir, output_dir))
    record, summary = _single_failed_record(output_dir)
    assert record["status"] == "failed"
    assert summary["quantization_audit"]["spoof"]["n_distinct_table_sets"] == 2
