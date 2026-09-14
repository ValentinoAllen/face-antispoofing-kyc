"""Tests for the annotation-vector layout, path parsing, conflict policies and the data config."""

import json
import logging
from pathlib import Path

import pytest
import yaml

from antispoof.data import labels
from antispoof.data.config import (
    CONFLICT_EXCLUDE,
    CONFLICT_POLICIES,
    CONFLICT_TRUST_LABEL,
    CONFLICT_TRUST_PATH,
    DataConfigError,
    load_data_config,
)
from antispoof.data.manifest import (
    MANIFEST_COLUMNS,
    ManifestError,
    ParsedPath,
    build_manifest,
    load_label_json,
    parse_image_path,
)

REPO_DATA_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"
TRAIN, TEST = labels.SPLIT_TRAIN, labels.SPLIT_TEST
ACCESSORS = [labels.get_label, labels.get_spoof_type, labels.get_attr_41, labels.get_attr_42]


# ---------------------------------------------------------------- annotation vector


@pytest.mark.parametrize("accessor", ACCESSORS)
@pytest.mark.parametrize("length", [0, labels.VECTOR_LENGTH - 1, labels.VECTOR_LENGTH + 1])
def test_every_accessor_validates_vector_length(accessor, length) -> None:
    with pytest.raises(labels.LabelVectorError, match=f"has {length} entries"):
        accessor([0] * length)


def test_accessors_read_the_documented_indices(make_vector) -> None:
    vector = make_vector(labels.LABEL_SPOOF, spoof_type=3, attr_41=5, attr_42=7)
    assert labels.get_label(vector) == labels.LABEL_SPOOF
    assert labels.get_spoof_type(vector) == 3
    assert labels.get_attr_41(vector) == 5
    assert labels.get_attr_42(vector) == 7


def test_unknown_label_value_is_rejected(make_vector) -> None:
    vector = make_vector(labels.LABEL_SPOOF)
    vector[labels.INDEX_LABEL] = 2
    with pytest.raises(labels.LabelVectorError, match="not one of"):
        labels.get_label(vector)


# ---------------------------------------------------------------- path parsing


def test_parse_image_path_reads_split_subject_and_kind() -> None:
    parsed = parse_image_path("Data/train/1234/live/000001.jpg")
    assert parsed == ParsedPath(TRAIN, "1234", labels.PATH_KIND_LIVE)


def test_parse_image_path_locates_split_after_any_prefix_and_keeps_leading_zeros() -> None:
    parsed = parse_image_path("CelebA_Spoof/Data/test/0042/spoof/000123.png")
    assert parsed == ParsedPath(TEST, "0042", labels.PATH_KIND_SPOOF)


@pytest.mark.parametrize(
    "image_path",
    [
        "Data/val/1234/live/000001.jpg",  # no train/test component
        "Data/train/1234/replay/000001.jpg",  # unknown path kind
        "Data/train/1234/live",  # too short
        "Data/train/1234/live/extra/000001.jpg",  # too long
    ],
)
def test_parse_image_path_rejects_malformed_paths(image_path) -> None:
    with pytest.raises(ManifestError):
        parse_image_path(image_path)


# ---------------------------------------------------------------- manifest and conflict policies


@pytest.fixture
def conflicting_train_labels(make_labels, make_vector, image_path) -> dict[str, list[int]]:
    """Ten rows with three conflicts, one in the reverse direction.

    - Subject 0001: 2 live, 3 spoof, consistent.
    - Subject 0002: 1 live, 1 spoof consistent, plus 2 images under ``live/`` labelled spoof.
    - Subject 0003: a single image under ``spoof/`` labelled live.
    """
    label_vectors = make_labels(TRAIN, {"0001": (2, 3), "0002": (1, 1)})
    for index in (90, 91):
        path = image_path(TRAIN, "0002", labels.PATH_KIND_LIVE, index)
        label_vectors[path] = make_vector(labels.LABEL_SPOOF, spoof_type=1)
    label_vectors[image_path(TRAIN, "0003", labels.PATH_KIND_SPOOF, 92)] = make_vector(
        labels.LABEL_LIVE
    )
    return label_vectors


def test_manifest_columns_and_types(conflicting_train_labels) -> None:
    manifest, _ = build_manifest(conflicting_train_labels, TRAIN, CONFLICT_TRUST_LABEL)
    assert list(manifest.columns) == list(MANIFEST_COLUMNS)
    assert set(manifest["subject_id"]) == {"0001", "0002", "0003"}
    assert set(manifest["split"]) == {TRAIN}
    assert str(manifest["label"].dtype) == "int8"
    assert manifest["conflict"].dtype == bool


@pytest.mark.parametrize("policy", CONFLICT_POLICIES)
def test_stats_count_by_path_and_by_label(conflicting_train_labels, policy) -> None:
    _, stats = build_manifest(conflicting_train_labels, TRAIN, policy)
    assert (stats.rows_in, stats.subjects_in) == (10, 3)
    assert (stats.live_by_path, stats.spoof_by_path) == (5, 5)
    assert (stats.live_by_label, stats.spoof_by_label) == (4, 6)
    assert stats.conflicts_found == 3


def test_exclude_policy_drops_conflicting_rows(conflicting_train_labels) -> None:
    manifest, stats = build_manifest(conflicting_train_labels, TRAIN, CONFLICT_EXCLUDE)
    assert not manifest["conflict"].any()
    assert (stats.conflicts_dropped, stats.conflicts_relabelled) == (3, 0)
    assert (stats.rows_out, stats.subjects_out, stats.live_out, stats.spoof_out) == (7, 2, 3, 4)
    assert len(manifest) == stats.rows_out
    assert "0003" not in set(manifest["subject_id"])


def test_trust_label_policy_keeps_rows_and_index_43_labels(conflicting_train_labels) -> None:
    manifest, stats = build_manifest(conflicting_train_labels, TRAIN, CONFLICT_TRUST_LABEL)
    conflicting = manifest.loc[manifest["conflict"]]
    assert len(manifest) == 10 and len(conflicting) == 3
    assert (conflicting["label"] != conflicting["path_kind"].map(labels.PATH_KIND_TO_LABEL)).all()
    assert (stats.conflicts_dropped, stats.conflicts_relabelled) == (0, 0)
    assert (stats.live_out, stats.spoof_out) == (4, 6)


def test_trust_path_policy_relabels_from_path(conflicting_train_labels) -> None:
    manifest, stats = build_manifest(conflicting_train_labels, TRAIN, CONFLICT_TRUST_PATH)
    assert len(manifest) == 10
    assert (manifest["label"] == manifest["path_kind"].map(labels.PATH_KIND_TO_LABEL)).all()
    assert manifest["conflict"].sum() == 3
    assert (stats.conflicts_dropped, stats.conflicts_relabelled) == (0, 3)
    assert (stats.live_out, stats.spoof_out) == (5, 5)


@pytest.mark.parametrize("policy", CONFLICT_POLICIES)
def test_conflict_count_is_logged(conflicting_train_labels, policy, caplog) -> None:
    caplog.set_level(logging.INFO, logger="antispoof.data.manifest")
    build_manifest(conflicting_train_labels, TRAIN, policy)
    messages = [r for r in caplog.records if "3 conflicting rows" in r.getMessage()]
    assert len(messages) == 1
    assert messages[0].levelno == logging.WARNING
    assert f"conflict_policy={policy}" in messages[0].getMessage()


def test_zero_conflicts_are_still_logged(make_labels, caplog) -> None:
    caplog.set_level(logging.INFO, logger="antispoof.data.manifest")
    build_manifest(make_labels(TEST, {"0001": (1, 1)}), TEST, CONFLICT_EXCLUDE)
    assert any("0 conflicting rows" in r.getMessage() for r in caplog.records)


def test_rows_from_another_split_are_rejected(make_labels) -> None:
    with pytest.raises(ManifestError, match="belongs to 'test'"):
        build_manifest(make_labels(TEST, {"0001": (1, 1)}), TRAIN, CONFLICT_EXCLUDE)


def test_unknown_policy_is_rejected(make_labels) -> None:
    with pytest.raises(ManifestError, match="conflict_policy"):
        build_manifest(make_labels(TRAIN, {"0001": (1, 1)}), TRAIN, "majority_vote")


def test_empty_label_mapping_is_rejected() -> None:
    with pytest.raises(ManifestError, match="empty"):
        build_manifest({}, TRAIN, CONFLICT_EXCLUDE)


def test_bad_vector_error_names_the_image(image_path) -> None:
    path = image_path(TRAIN, "0001", labels.PATH_KIND_LIVE, 0)
    with pytest.raises(labels.LabelVectorError, match="0001/live/000000.jpg"):
        build_manifest({path: [0] * (labels.VECTOR_LENGTH - 1)}, TRAIN, CONFLICT_EXCLUDE)


def test_load_label_json_round_trip(tmp_path, make_labels) -> None:
    label_vectors = make_labels(TRAIN, {"0001": (1, 2)})
    path = tmp_path / "train_label.json"
    path.write_text(json.dumps(label_vectors), encoding="utf-8")
    assert load_label_json(path) == label_vectors


def test_load_label_json_rejects_non_object(tmp_path) -> None:
    path = tmp_path / "train_label.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ManifestError, match="JSON object"):
        load_label_json(path)


# ---------------------------------------------------------------- data config


def test_repo_config_defaults_to_exclude(repo_data_config) -> None:
    assert repo_data_config.conflict_policy == CONFLICT_EXCLUDE


def test_repo_config_label_paths(repo_data_config) -> None:
    config = repo_data_config
    base = config.dataset_root / config.metas_dir / config.protocol
    assert config.label_path(TRAIN) == base / config.train_label_file
    assert config.label_path(TEST) == base / config.test_label_file
    with pytest.raises(DataConfigError):
        config.label_path(labels.SPLIT_VAL)


def _write_modified_config(tmp_path: Path, section: str, key: str, value: object) -> Path:
    document = yaml.safe_load(REPO_DATA_CONFIG.read_text(encoding="utf-8"))
    document[section][key] = value
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("data", "conflict_policy", "majority_vote"),
        ("data", "excluded_subjects", [1]),  # unquoted ids lose leading zeros
        ("data", "excluded_subjects", ["0001", "0001"]),
        ("split", "val_fraction", 0.0),
        ("split", "val_fraction", 1.0),
        ("split", "stratify_bins", 0),
        ("split", "seed", True),
        ("split", "unknown_key", 1),
    ],
)
def test_invalid_config_is_rejected(tmp_path, section, key, value) -> None:
    path = _write_modified_config(tmp_path, section, key, value)
    with pytest.raises(DataConfigError):
        load_data_config(path)


def test_config_with_missing_key_is_rejected(tmp_path) -> None:
    document = yaml.safe_load(REPO_DATA_CONFIG.read_text(encoding="utf-8"))
    del document["split"]["seed"]
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(DataConfigError, match="missing keys \\['seed'\\]"):
        load_data_config(path)
