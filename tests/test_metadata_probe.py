"""Tests for the header-metadata probe on synthetic JPEG and PNG files. No pixel data is decoded."""

import dataclasses
import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image, ImageFile

from antispoof.data import labels
from antispoof.data.build import MANIFEST_FILENAME, SplitOutputs, write_outputs
from antispoof.data.config import CONFLICT_EXCLUDE, DataConfig
from antispoof.data.dataset import ImageLoadError
from antispoof.data.manifest import build_manifest
from antispoof.eval import metadata_probe as probe
from antispoof.training import reproducibility
from antispoof.training.config import load_train_config
from antispoof.training.run import RECORD_FILENAME, RESOLVED_CONFIG_FILENAME, format_ledger_row

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_CONFIG = REPO_ROOT / "configs" / "probe_metadata.yaml"
BASELINE_CONFIG = REPO_ROOT / "configs" / "baseline.yaml"

HYPOTHESIS = (
    "Header-level metadata alone (format, dimensions, file size, JPEG tables, EXIF/ICC presence; "
    "no pixels) separates live from spoof on the baseline's 4k/2k subsets. Decision rule fixed "
    "before the run, on the primary classifier's pooled val ACER: <= 5% strong header-level "
    "shortcut; >= 25% weak header-level shortcut, pixel-level cues untested; otherwise partial."
)
WHAT_CHANGED = "Metadata probe: no pixels, HistGradientBoosting on header features"
NOTES = "probe, not a model; same subsets and threshold as 20260915-153606-baseline"

TRAIN_SUBJECTS = {"0001": (2, 2), "0002": (2, 2)}
VAL_SUBJECTS = {"0003": (2, 2), "0004": (2, 2)}
TEST_SUBJECT = "0005"
ROWS_PER_SPLIT = 8
THRESHOLD = 0.5

JPEG_SIZE = (40, 20)
PNG_SIZE = (30, 10)
LUMA_QUANT = 7
ICC_BYTES = b"synthetic icc profile"


@pytest.fixture
def spy_on_load(monkeypatch: pytest.MonkeyPatch) -> Callable[[], list[str]]:
    """Return a function that starts recording every pixel-load call and returns the record."""
    calls: list[str] = []

    def install() -> list[str]:
        for owner in (Image.Image, ImageFile.ImageFile):
            monkeypatch.setattr(owner, "load", _recording(owner.load, calls))
        return calls

    return install


def _recording(original: Callable[..., Any], calls: list[str]) -> Callable[..., Any]:
    def spy(self: Image.Image, *args: Any, **kwargs: Any) -> Any:
        calls.append(type(self).__name__)
        return original(self, *args, **kwargs)

    return spy


@pytest.fixture
def probe_data(
    tmp_data_config: DataConfig,
    make_labels: Callable[..., dict[str, list[int]]],
    write_images: Callable[[pd.DataFrame, Path], None],
) -> DataConfig:
    """Train and val manifests, their JPEG images, and a split assignment, under ``tmp_path``."""
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


def _write_jpeg(path: Path, *, exif: bool, icc: bool, progressive: bool) -> None:
    options: dict[str, Any] = {
        "qtables": [[LUMA_QUANT] * 64] * 2,
        "subsampling": 2,
        "progressive": progressive,
    }
    if exif:
        tags = Image.Exif()
        tags[0x010F] = "synthetic camera"
        options["exif"] = tags.tobytes()
    if icc:
        options["icc_profile"] = ICC_BYTES
    Image.new("RGB", JPEG_SIZE, (120, 60, 30)).save(path, "JPEG", **options)


def _comparison_inputs() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Four live then four spoof val rows, metadata scores, and CNN predictions in reverse order."""
    val = pd.DataFrame(
        {
            "image_path": [f"img{index}.jpg" for index in range(8)],
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    metadata_scores = np.array([0.7, 0.1, 0.6, 0.3, 0.4, 0.9, 0.9, 0.9])
    cnn_scores = [0.9, 0.8, 0.1, 0.2, 0.9, 0.8, 0.7, 0.6]
    predictions = val.assign(score=cnn_scores).iloc[::-1].reset_index(drop=True)
    return val, metadata_scores, predictions


def _write_predictions(data_config: DataConfig, path: Path, *, drop_last: bool = False) -> Path:
    val_manifest = data_config.manifest_dir / MANIFEST_FILENAME.format(split=labels.SPLIT_VAL)
    val = pd.read_csv(val_manifest, dtype={"subject_id": str})
    rows = val.iloc[:-1] if drop_last else val
    scored = rows.assign(score=0.1 + 0.8 * rows["label"])
    scored[["image_path", "subject_id", "label", "score"]].to_csv(path, index=False)
    return path


def _probe_inputs(
    data_config: DataConfig, output_dir: Path, predictions_path: Path | None
) -> probe.ProbeInputs:
    baseline = load_train_config(BASELINE_CONFIG)
    subset_config = dataclasses.replace(
        baseline,
        data=dataclasses.replace(
            baseline.data, train_subset=ROWS_PER_SPLIT, val_subset=ROWS_PER_SPLIT
        ),
    )
    return probe.ProbeInputs(
        probe_config=probe.load_probe_config(PROBE_CONFIG),
        config_path=PROBE_CONFIG,
        subset_config=subset_config,
        subset_config_path=BASELINE_CONFIG,
        data_config=data_config,
        output_dir=output_dir,
        repo_root=REPO_ROOT,
        predictions_path=predictions_path,
    )


def test_jpeg_header_features_match_what_was_written(
    tmp_path: Path, spy_on_load: Callable[[], list[str]]
) -> None:
    path = tmp_path / "photo.JPG"
    _write_jpeg(path, exif=True, icc=True, progressive=True)
    calls = spy_on_load()
    values = probe.read_header_features(path)
    assert calls == []
    width, height = JPEG_SIZE
    size = path.stat().st_size
    assert values == {
        "extension": ".jpg",
        "format": "JPEG",
        "mode": "RGB",
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "pixel_count": width * height,
        "file_size_bytes": size,
        "bytes_per_pixel": size / (width * height),
        "jpeg_subsampling": "4:2:0",
        "jpeg_progressive": True,
        "jpeg_luma_quant_mean": float(LUMA_QUANT),
        "has_exif": True,
        "has_icc_profile": True,
    }


def test_jpeg_without_exif_icc_or_progressive_scan(tmp_path: Path) -> None:
    path = tmp_path / "plain.jpg"
    _write_jpeg(path, exif=False, icc=False, progressive=False)
    values = probe.read_header_features(path)
    assert (values["has_exif"], values["has_icc_profile"], values["jpeg_progressive"]) == (
        False,
        False,
        False,
    )


def test_png_has_null_jpeg_features_with_missing_indicators(
    tmp_path: Path, spy_on_load: Callable[[], list[str]]
) -> None:
    relative = "Data/train/0001/spoof/000001.png"
    (tmp_path / relative).parent.mkdir(parents=True)
    Image.new("RGBA", PNG_SIZE, (0, 0, 0, 0)).save(tmp_path / relative, icc_profile=ICC_BYTES)
    subset = pd.DataFrame(
        {"image_path": [relative], "split": [labels.SPLIT_TRAIN], "label": [labels.LABEL_SPOOF]}
    )
    calls = spy_on_load()
    features = probe.extract_features(subset, tmp_path)
    assert calls == []
    assert list(features.columns) == [*probe.IDENTIFIER_COLUMNS, *probe.MODEL_FEATURES]
    row = features.iloc[0]
    assert (row["format"], row["mode"], row["extension"]) == ("PNG", "RGBA", ".png")
    assert (row["width"], row["height"]) == PNG_SIZE
    assert not row["has_exif"]
    assert row["has_icc_profile"]
    for name in probe.JPEG_ONLY_FEATURES:
        assert pd.isna(row[name])
        assert row[f"{name}_missing"]


def test_unreadable_headers_raise_naming_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jpg"
    with pytest.raises(ImageLoadError, match=re.escape(str(missing))):
        probe.read_header_features(missing)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    with pytest.raises(ImageLoadError, match=re.escape(str(broken))):
        probe.read_header_features(broken)


def test_comparison_counts_errors_and_expected_overlap() -> None:
    val, metadata_scores, predictions = _comparison_inputs()
    comparison = probe.compare_with_predictions(val, metadata_scores, predictions, THRESHOLD)
    live, spoof = comparison["live"], comparison["spoof"]

    # Live errors score >= 0.5. CNN: rows 0 and 1. Metadata: rows 0 and 2. Shared: row 0.
    assert (live["n"], live["cnn_errors"], live["metadata_errors"], live["both_errors"]) == (
        4,
        2,
        2,
        1,
    )
    assert live["both_errors_expected_if_independent"] == {"value": 1.0, "na_reason": None}
    assert live["cnn_errors_also_metadata_errors"] == {"value": 0.5, "na_reason": None}
    # Ranks: CNN (4, 3, 1, 2), metadata (4, 1, 3, 2); rho = 1 / 5.
    assert math.isclose(live["spearman_rho"]["value"], 0.2)

    # Spoof errors score < 0.5. CNN: none. Metadata: row 4.
    assert (spoof["cnn_errors"], spoof["metadata_errors"], spoof["both_errors"]) == (0, 1, 0)
    assert spoof["both_errors_expected_if_independent"] == {"value": 0.0, "na_reason": None}
    assert spoof["cnn_errors_also_metadata_errors"]["value"] is None
    assert "no CNN errors" in spoof["cnn_errors_also_metadata_errors"]["na_reason"]
    # Ranks: CNN (4, 3, 2, 1), metadata (1, 3, 3, 3); rho = -3 / sqrt(15).
    assert math.isclose(spoof["spearman_rho"]["value"], -3 / math.sqrt(15))


def test_comparison_is_n_a_for_constant_scores_and_empty_classes() -> None:
    val, _, predictions = _comparison_inputs()
    live_val = val[val["label"] == labels.LABEL_LIVE]
    live_predictions = predictions[predictions["label"] == labels.LABEL_LIVE]
    constant = np.full(len(live_val), 0.5)
    comparison = probe.compare_with_predictions(live_val, constant, live_predictions, THRESHOLD)
    rho = comparison["live"]["spearman_rho"]
    assert rho["value"] is None
    assert "constant" in rho["na_reason"]
    spoof = comparison["spoof"]
    assert spoof["n"] == 0
    assert spoof["both_errors_expected_if_independent"]["value"] is None
    assert spoof["both_errors_expected_if_independent"]["na_reason"] == "no rows in this class"
    assert spoof["spearman_rho"]["na_reason"] == "fewer than two rows"


@pytest.mark.parametrize(
    "change",
    [
        lambda frame: frame.iloc[:-1],
        lambda frame: pd.concat([frame, frame.iloc[:1].assign(image_path="extra.jpg")]),
        lambda frame: pd.concat([frame, frame.iloc[:1]]),
    ],
    ids=["missing", "extra", "duplicated"],
)
def test_predictions_must_match_the_val_subset_exactly(
    change: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    val, metadata_scores, predictions = _comparison_inputs()
    with pytest.raises(probe.ProbeInputError, match="exactly"):
        probe.compare_with_predictions(val, metadata_scores, change(predictions), THRESHOLD)


def test_predictions_with_a_different_label_are_rejected() -> None:
    val, metadata_scores, predictions = _comparison_inputs()
    predictions.loc[0, "label"] = 1 - predictions.loc[0, "label"]
    with pytest.raises(probe.ProbeInputError, match="label"):
        probe.compare_with_predictions(val, metadata_scores, predictions, THRESHOLD)


def test_measured_requires_a_reason_for_missing_values() -> None:
    assert probe.measured(0.25) == {"value": 0.25, "na_reason": None}
    with pytest.raises(ValueError, match="reason"):
        probe.measured(None)


def test_committed_probe_config_has_the_agreed_fields() -> None:
    config = probe.load_probe_config(PROBE_CONFIG)
    assert (config.run.hypothesis, config.run.what_changed, config.run.notes) == (
        HYPOTHESIS,
        WHAT_CHANGED,
        NOTES,
    )
    assert config.subsets.config == "configs/baseline.yaml"
    baseline = load_train_config(REPO_ROOT / config.subsets.config)
    assert config.eval.threshold == baseline.eval.threshold == THRESHOLD
    assert config.primary.early_stopping is False
    probe.check_probe_inputs(config, baseline)


def test_probe_threshold_must_equal_the_baseline_threshold() -> None:
    config = probe.load_probe_config(PROBE_CONFIG)
    shifted = dataclasses.replace(config, eval=dataclasses.replace(config.eval, threshold=0.4))
    with pytest.raises(probe.ProbeConfigError, match="baseline's threshold"):
        probe.check_probe_inputs(shifted, load_train_config(BASELINE_CONFIG))


def test_log1p_must_name_numeric_features(tmp_path: Path) -> None:
    document = yaml.safe_load(PROBE_CONFIG.read_text(encoding="utf-8"))
    document["features"]["log1p"] = ["extension"]
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(probe.ProbeConfigError, match="features.log1p"):
        probe.load_probe_config(path)


def test_run_probe_writes_the_run_directory_without_decoding_pixels(
    probe_data: DataConfig, tmp_path: Path, spy_on_load: Callable[[], list[str]]
) -> None:
    predictions_path = _write_predictions(probe_data, tmp_path / "predictions.csv")
    calls = spy_on_load()
    result = probe.run_probe(_probe_inputs(probe_data, tmp_path / "runs", predictions_path))
    assert calls == []

    run_dir = result.run_dir
    for name in (
        RECORD_FILENAME,
        RESOLVED_CONFIG_FILENAME,
        probe.PROBE_SUMMARY_FILENAME,
        probe.FEATURES_FILENAME,
    ):
        assert (run_dir / name).is_file()
    record = json.loads((run_dir / RECORD_FILENAME).read_text(encoding="utf-8"))
    assert record == result.record
    assert (record["status"], record["eval_split"], record["threshold"]) == (
        "completed",
        "val",
        THRESHOLD,
    )
    assert record["training_epochs"] is None
    assert record["run_id"].endswith("-probe_metadata")
    assert record["config_path"] == "configs/probe_metadata.yaml"
    assert set(record["manifest_sha256"]) == {"manifest_train.csv", "manifest_val.csv"}
    assert record["data_subsets"]["val"]["rows"] == ROWS_PER_SPLIT
    assert all(value is None for value in record["artifacts"].values())
    assert (record["metrics"]["n_attack"], record["metrics"]["n_bona_fide"]) == (4, 4)
    assert {"pillow", "sklearn"} <= set(record["environment"])

    resolved = json.loads((run_dir / RESOLVED_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert record["config_hash"] == reproducibility.config_hash(resolved)
    assert resolved["subsets"]["config_path"] == "configs/baseline.yaml"
    assert resolved["subsets"]["train_subset"] == ROWS_PER_SPLIT

    features = pd.read_csv(run_dir / probe.FEATURES_FILENAME)
    assert list(features.columns) == [*probe.IDENTIFIER_COLUMNS, *probe.MODEL_FEATURES]
    assert len(features) == 2 * ROWS_PER_SPLIT
    summary = json.loads((run_dir / probe.PROBE_SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary == result.summary
    assert set(summary["predictions_comparison"]) == {"live", "spoof"}
    assert len(summary["single_feature_aucs"]["features"]) == len(probe.HEADER_FEATURES)
    assert summary["inputs"]["predictions_sha256"] == reproducibility.sha256_file(predictions_path)

    cells = format_ledger_row(record).strip("|").split(" | ")
    assert len(cells) == 11
    assert cells[6].endswith("(pooled)") and cells[8].endswith("(pooled)")
    assert "not a PAD metric" in probe.format_probe_report(result)


def test_run_probe_rejects_mismatched_predictions_before_writing(
    probe_data: DataConfig, tmp_path: Path
) -> None:
    predictions_path = _write_predictions(probe_data, tmp_path / "short.csv", drop_last=True)
    output_dir = tmp_path / "runs"
    with pytest.raises(probe.ProbeInputError, match="1 missing"):
        probe.run_probe(_probe_inputs(probe_data, output_dir, predictions_path))
    assert not output_dir.exists()
