"""Tests for the full-split JPEG quantization-table audit.

Synthetic images under ``tmp_path`` only; no pixel data is decoded and nothing reads ``data/``.
"""

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from antispoof.data import labels
from antispoof.data.build import SplitOutputs, write_outputs
from antispoof.data.config import DataConfig
from antispoof.data.manifest import build_manifest
from antispoof.eval import jpeg_audit
from antispoof.eval.jpeg_tables import reference_tables
from antispoof.training.run import (
    RECORD_FILENAME,
    RESOLVED_CONFIG_FILENAME,
    load_manifests,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_CONFIG = REPO_ROOT / "configs" / "audit_jpeg.yaml"

CONFLICT_EXCLUDE = "exclude"
LIVE_QUALITY = 75
MAJOR_SPOOF_QUALITY = 95
MINOR_SPOOF_QUALITY = 85
SUBSAMPLING_420 = 2
MAJOR_SPOOF_TYPE = 1
MINOR_SPOOF_TYPE = 2

SUBJECTS = {
    labels.SPLIT_TRAIN: {"0001": (1, 4), "0002": (1, 4)},
    labels.SPLIT_VAL: {"0003": (2, 4)},
    labels.SPLIT_TEST: {"0004": (2, 4)},
}
ROWS = {labels.SPLIT_TRAIN: 10, labels.SPLIT_VAL: 6, labels.SPLIT_TEST: 6}
"""Rows per split. One spoof row per split carries MINOR_SPOOF_TYPE; the rest the major one."""

MAJOR_SPOOF_ROWS = {labels.SPLIT_TRAIN: 7, labels.SPLIT_VAL: 3, labels.SPLIT_TEST: 3}
MINOR_SPOOF_ROWS = dict.fromkeys(labels.SPLITS, 1)
LIVE_ROWS = {labels.SPLIT_TRAIN: 2, labels.SPLIT_VAL: 2, labels.SPLIT_TEST: 2}


def _mark_minor_spoof_row(manifest: pd.DataFrame) -> None:
    """Give the last spoof row a second spoof_type, so the class has two encodings."""
    spoof = manifest.index[manifest["label"] == labels.LABEL_SPOOF]
    manifest.loc[spoof[-1], "spoof_type"] = MINOR_SPOOF_TYPE


def _quality(label: int, spoof_type: int) -> int:
    if label == labels.LABEL_LIVE:
        return LIVE_QUALITY
    return MINOR_SPOOF_QUALITY if spoof_type == MINOR_SPOOF_TYPE else MAJOR_SPOOF_QUALITY


def _write_noise_images(manifest: pd.DataFrame, root: Path) -> None:
    """Seeded noise JPEGs whose quality is decided by the row's label and spoof_type."""
    for position, (image_path, label, spoof_type) in enumerate(
        zip(manifest["image_path"], manifest["label"], manifest["spoof_type"], strict=True)
    ):
        pixels = np.random.default_rng(position).integers(0, 256, size=(24, 20, 3), dtype=np.uint8)
        path = root / image_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels, mode="RGB").save(
            path, "JPEG", quality=_quality(int(label), int(spoof_type)), subsampling=SUBSAMPLING_420
        )


@pytest.fixture
def audit_data(
    tmp_data_config: DataConfig, make_labels: Callable[..., dict[str, list[int]]]
) -> DataConfig:
    """Three manifests, their noise JPEGs and a split assignment, under ``tmp_path``."""
    manifests: dict[str, pd.DataFrame] = {}
    assignment_rows: list[tuple[str, str]] = []
    for split, subjects in SUBJECTS.items():
        source = labels.SPLIT_TEST if split == labels.SPLIT_TEST else labels.SPLIT_TRAIN
        manifest, _ = build_manifest(make_labels(source, subjects), source, CONFLICT_EXCLUDE)
        manifest["split"] = split
        _mark_minor_spoof_row(manifest)
        manifests[split] = manifest
        _write_noise_images(manifest, tmp_data_config.dataset_root)
        assignment_rows.extend((subject, split) for subject in subjects)
    assignment = pd.DataFrame(assignment_rows, columns=["subject_id", "split"])
    outputs = SplitOutputs(manifests=manifests, assignment=assignment)
    write_outputs(outputs, tmp_data_config.manifest_dir, tmp_data_config.split_assignment_path)
    return tmp_data_config


def _config(**scan: Any) -> jpeg_audit.AuditConfig:
    config = jpeg_audit.load_audit_config(AUDIT_CONFIG)
    return dataclasses.replace(config, audit=dataclasses.replace(config.audit, **scan))


def _inputs(data_config: DataConfig, output_dir: Path, **scan: Any) -> jpeg_audit.AuditInputs:
    return jpeg_audit.AuditInputs(
        config=_config(**scan),
        config_path=AUDIT_CONFIG,
        data_config=data_config,
        output_dir=output_dir,
        repo_root=REPO_ROOT,
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_audit_writes_the_run_directory(audit_data: DataConfig, tmp_path: Path) -> None:
    result = jpeg_audit.run_audit(_inputs(audit_data, tmp_path / "runs"))
    record, summary = result.record, result.summary
    assert record == _read_json(result.run_dir / RECORD_FILENAME)
    assert summary == _read_json(result.run_dir / jpeg_audit.AUDIT_SUMMARY_FILENAME)
    assert record["status"] == "completed"
    assert record["run_id"].endswith("-audit_jpeg")
    assert record["config_path"] == "configs/audit_jpeg.yaml"
    assert set(record["manifest_sha256"]) == {
        "manifest_train.csv",
        "manifest_val.csv",
        "manifest_test.csv",
    }
    assert set(record["metrics"].values()) == {None}
    assert record["training_epochs"] is None
    assert (record["eval_split"], record["threshold"], record["threshold_rule"]) == (
        None,
        None,
        None,
    )
    assert set(record["data_subsets"]) == set(labels.SPLITS)
    resolved = _read_json(result.run_dir / RESOLVED_CONFIG_FILENAME)
    assert resolved["experiment"]["audit"]["splits"] == list(labels.SPLITS)

    assert (summary["rows_scanned"], summary["headers_read"]) == (sum(ROWS.values()),) * 2
    assert summary["failures"]["count"] == 0
    audit = summary["quantization_audit"]
    assert audit["live"]["n_distinct_table_sets"] == 1
    assert audit["spoof"]["n_distinct_table_sets"] == 2
    (live_set,) = audit["live"]["table_sets"]
    assert (live_set["match"], live_set["rows"]) == (f"q={LIVE_QUALITY}", LIVE_ROWS)
    assert live_set["luma_mean"] == float(
        np.mean(reference_tables(LIVE_QUALITY, SUBSAMPLING_420).luma)
    )
    assert [entry["match"] for entry in audit["spoof"]["table_sets"]] == [
        f"q={MAJOR_SPOOF_QUALITY}",
        f"q={MINOR_SPOOF_QUALITY}",
    ]
    assert audit["spoof"]["table_sets"][0]["rows"] == MAJOR_SPOOF_ROWS
    assert audit["spoof"]["table_sets"][1]["rows"] == MINOR_SPOOF_ROWS


def test_shares_are_the_table_set_fraction_of_its_class_in_each_split(
    audit_data: DataConfig, tmp_path: Path
) -> None:
    summary = jpeg_audit.run_audit(_inputs(audit_data, tmp_path / "runs")).summary
    audit = summary["quantization_audit"]
    for class_name, entry in audit.items():
        for split in labels.SPLITS:
            total = sum(table_set["share"][split] for table_set in entry["table_sets"])
            assert total == pytest.approx(1.0), (class_name, split)
    major, minor = audit["spoof"]["table_sets"]
    train = labels.SPLIT_TRAIN
    assert major["share"][train] == pytest.approx(
        MAJOR_SPOOF_ROWS[train] / (MAJOR_SPOOF_ROWS[train] + MINOR_SPOOF_ROWS[train])
    )
    assert minor["share"][train] == pytest.approx(
        MINOR_SPOOF_ROWS[train] / (MAJOR_SPOOF_ROWS[train] + MINOR_SPOOF_ROWS[train])
    )


def test_header_value_counts_cover_every_split_and_class(
    audit_data: DataConfig, tmp_path: Path
) -> None:
    summary = jpeg_audit.run_audit(_inputs(audit_data, tmp_path / "runs")).summary
    counts = summary["header_value_counts"]
    assert set(counts) == set(labels.SPLITS)
    for split, per_class in counts.items():
        assert set(per_class) == {"live", "spoof"}
        assert per_class["live"]["rows"] == LIVE_ROWS[split]
        assert per_class["live"]["format"] == {"JPEG": LIVE_ROWS[split]}
        assert per_class["spoof"]["mode"] == {"RGB": ROWS[split] - LIVE_ROWS[split]}
        assert per_class["spoof"]["subsampling"] == {"4:2:0": ROWS[split] - LIVE_ROWS[split]}


def test_spoof_type_crosstab_separates_the_two_encodings(
    audit_data: DataConfig, tmp_path: Path
) -> None:
    summary = jpeg_audit.run_audit(_inputs(audit_data, tmp_path / "runs")).summary
    crosstab = summary["spoof_type_crosstab"]
    assert crosstab["spoof_type_codes"] == [MAJOR_SPOOF_TYPE, MINOR_SPOOF_TYPE]
    major, minor = crosstab["table_sets"]
    assert (major["rank"], major["match"]) == (1, f"q={MAJOR_SPOOF_QUALITY}")
    assert (minor["rank"], minor["match"]) == (2, f"q={MINOR_SPOOF_QUALITY}")
    for split in labels.SPLITS:
        assert major["counts"][split] == {
            str(MAJOR_SPOOF_TYPE): MAJOR_SPOOF_ROWS[split],
            str(MINOR_SPOOF_TYPE): 0,
        }
        assert minor["counts"][split] == {
            str(MAJOR_SPOOF_TYPE): 0,
            str(MINOR_SPOOF_TYPE): MINOR_SPOOF_ROWS[split],
        }


def test_audit_reads_headers_only(
    audit_data: DataConfig, tmp_path: Path, spy_on_file_loads: Callable[[], list[str]]
) -> None:
    calls = spy_on_file_loads()
    result = jpeg_audit.run_audit(_inputs(audit_data, tmp_path / "runs"))
    assert calls == []
    assert result.summary["headers_read"] == sum(ROWS.values())
    manifest = pd.read_csv(audit_data.manifest_dir / "manifest_train.csv")
    with Image.open(audit_data.dataset_root / manifest["image_path"][0]) as image:
        image.load()
    assert calls and set(calls) == {"JpegImageFile"}, "the spy must record a real pixel load"


def test_unreadable_headers_are_counted_and_listed_without_stopping_the_run(
    audit_data: DataConfig, tmp_path: Path
) -> None:
    manifest = pd.read_csv(audit_data.manifest_dir / "manifest_val.csv")
    missing, broken, png = (audit_data.dataset_root / manifest["image_path"][i] for i in (0, 1, 2))
    missing.unlink()
    broken.write_bytes(b"not an image")
    Image.new("RGB", (8, 8)).save(png, "PNG")

    result = jpeg_audit.run_audit(_inputs(audit_data, tmp_path / "runs"))
    failures = result.summary["failures"]
    assert result.record["status"] == "completed"
    assert failures["count"] == 3
    assert result.summary["headers_read"] == sum(ROWS.values()) - 3
    assert result.summary["rows_scanned"] == sum(ROWS.values())
    listed = {entry["image_path"]: entry["error"] for entry in failures["listed"]}
    assert set(listed) == {manifest["image_path"][i] for i in (0, 1, 2)}
    assert all(entry["split"] == labels.SPLIT_VAL for entry in failures["listed"])
    assert "Image file not found" in listed[manifest["image_path"][0]]
    assert "is not a JPEG" in listed[manifest["image_path"][2]]
    report = jpeg_audit.format_audit_report(result)
    assert "Unreadable headers: 3" in report


def test_failure_list_is_capped_while_the_count_is_not(
    audit_data: DataConfig, tmp_path: Path
) -> None:
    manifest = pd.read_csv(audit_data.manifest_dir / "manifest_test.csv")
    for image_path in manifest["image_path"]:
        (audit_data.dataset_root / image_path).unlink()

    result = jpeg_audit.run_audit(_inputs(audit_data, tmp_path / "runs", max_listed_failures=2))
    failures = result.summary["failures"]
    assert failures["count"] == ROWS[labels.SPLIT_TEST]
    assert len(failures["listed"]) == 2
    assert failures["max_listed"] == 2


def test_equal_table_sets_are_interned_to_one_object(
    audit_data: DataConfig, tmp_path: Path
) -> None:
    manifests = load_manifests(labels.SPLITS, audit_data)
    headers, failures = jpeg_audit.read_audit_headers(manifests, audit_data.dataset_root, 5)
    assert not failures
    distinct = {id(tables) for tables in headers["tables"]}
    assert len(distinct) == 3, "one object per distinct table set, not one per row"
    assert len({tables for tables in headers["tables"]}) == 3


def test_a_single_split_can_be_audited(audit_data: DataConfig, tmp_path: Path) -> None:
    result = jpeg_audit.run_audit(
        _inputs(audit_data, tmp_path / "runs", splits=(labels.SPLIT_VAL,))
    )
    assert set(result.record["manifest_sha256"]) == {"manifest_val.csv"}
    assert result.summary["rows_scanned"] == ROWS[labels.SPLIT_VAL]
    assert set(result.summary["quantization_audit"]["live"]["rows"]) == {labels.SPLIT_VAL}


def test_report_states_that_the_run_gets_no_ledger_row(
    audit_data: DataConfig, tmp_path: Path
) -> None:
    result = jpeg_audit.run_audit(_inputs(audit_data, tmp_path / "runs"))
    report = jpeg_audit.format_audit_report(result)
    assert report.splitlines()[-1] == jpeg_audit.NO_LEDGER_ROW
    assert "no PAD metrics" in jpeg_audit.NO_LEDGER_ROW
    assert "Quantization-table audit (headers only, no pixels decoded):" in report
    assert f"q={MAJOR_SPOOF_QUALITY}" in report and f"q={LIVE_QUALITY}" in report
    assert "Spoof table set against spoof_type" in report


def test_a_manifest_of_another_split_is_rejected(audit_data: DataConfig, tmp_path: Path) -> None:
    path = audit_data.manifest_dir / "manifest_val.csv"
    manifest = pd.read_csv(path)
    manifest["split"] = labels.SPLIT_TRAIN
    manifest.to_csv(path, index=False)
    with pytest.raises(ValueError, match="rows are not in split 'val'"):
        jpeg_audit.run_audit(_inputs(audit_data, tmp_path / "runs"))
    assert not (tmp_path / "runs").exists()


def test_committed_audit_config_has_the_agreed_fields() -> None:
    config = jpeg_audit.load_audit_config(AUDIT_CONFIG)
    assert config.audit == jpeg_audit.AuditScanConfig(
        splits=labels.SPLITS, progress_every=50000, max_listed_failures=50
    )
    assert config.run.seed == 42
    assert config.run.what_changed == "Full-split header-only audit of JPEG quantization tables"
    assert "spoof_type" in config.run.hypothesis
    assert "headers only" in config.run.notes


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("audit", "splits", ["train", "train"], "must not repeat a split"),
        ("audit", "splits", ["train", "dev"], "audit.splits must be in"),
        ("audit", "splits", [], "audit.splits must not be empty"),
        ("audit", "splits", "train", "expected a list of split names"),
        ("audit", "progress_every", 0, "audit.progress_every must be >= 1"),
        ("audit", "progress_every", "many", "audit.progress_every: expected int"),
        ("audit", "max_listed_failures", -1, "audit.max_listed_failures must be >= 0"),
        ("run", "seed", -1, "run.seed must be >= 0"),
        ("run", "hypothesis", "  ", "run.hypothesis must not be empty"),
    ],
)
def test_invalid_audit_configs_are_rejected(
    tmp_path: Path, section: str, key: str, value: object, message: str
) -> None:
    document = yaml.safe_load(AUDIT_CONFIG.read_text(encoding="utf-8"))
    document[section][key] = value
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(jpeg_audit.AuditConfigError, match=message):
        jpeg_audit.load_audit_config(path)


def test_unknown_audit_section_or_key_is_rejected(tmp_path: Path) -> None:
    document = yaml.safe_load(AUDIT_CONFIG.read_text(encoding="utf-8"))
    document["audit"]["stride"] = 2
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(jpeg_audit.AuditConfigError, match="must hold exactly the keys"):
        jpeg_audit.load_audit_config(path)
