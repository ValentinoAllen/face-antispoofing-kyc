"""Tests for the subject-level split, the disjointness invariant and the manifest build."""

import dataclasses
import json

import pandas as pd
import pytest

from antispoof.data import labels
from antispoof.data.build import format_report, run
from antispoof.data.config import CONFLICT_EXCLUDE, CONFLICT_TRUST_LABEL, CONFLICT_TRUST_PATH
from antispoof.data.manifest import build_manifest
from antispoof.data.splits import (
    ProtectedTestSplitError,
    SplitCoverageError,
    SplitLeakageError,
    assign_validation,
    ensure_test_untouched,
    load_split_assignment,
    validate_coverage,
    validate_splits,
)

TRAIN, VAL, TEST = labels.SPLIT_TRAIN, labels.SPLIT_VAL, labels.SPLIT_TEST
VAL_FRACTION = 0.1
STRATIFY_BINS = 10
SPOOF_RATIO_TOLERANCE = 0.03
"""Maximum allowed |spoof ratio(val) - spoof ratio(train)| on the synthetic stratification set."""


def _assignment(split_by_subject: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"subject_id": list(split_by_subject), "split": list(split_by_subject.values())}
    )


def _manifest(make_labels, subjects: dict[str, tuple[int, int]]) -> pd.DataFrame:
    manifest, _ = build_manifest(make_labels(TRAIN, subjects), TRAIN, CONFLICT_EXCLUDE)
    return manifest


@pytest.fixture
def varied_train_manifest(make_labels) -> pd.DataFrame:
    """300 subjects whose live/spoof counts, and so spoof fractions, vary widely."""
    subjects = {f"{i:04d}": (1 + i % 4, (i * 7) % 11) for i in range(300)}
    return _manifest(make_labels, subjects)


# ---------------------------------------------------------------- validate_splits


def test_disjoint_assignment_passes() -> None:
    validate_splits(_assignment({"0001": TRAIN, "0002": VAL, "0003": TEST}))


def test_image_level_manifest_with_repeated_subjects_passes() -> None:
    frame = pd.DataFrame({"subject_id": ["0001", "0001", "0002"], "split": [TRAIN, TRAIN, TEST]})
    validate_splits(frame)


@pytest.mark.parametrize(("first", "second"), [(TRAIN, VAL), (TRAIN, TEST), (VAL, TEST)])
def test_shared_subject_fails_and_is_named(first, second) -> None:
    frame = pd.DataFrame({"subject_id": ["0007", "0007", "0001"], "split": [first, second, first]})
    with pytest.raises(SplitLeakageError, match=f"{first} and {second} share \\['0007'\\]"):
        validate_splits(frame)


def test_every_overlap_is_listed() -> None:
    frame = pd.DataFrame(
        {"subject_id": ["0007", "0007", "0009", "0009"], "split": [TRAIN, TEST, VAL, TEST]}
    )
    with pytest.raises(SplitLeakageError) as error:
        validate_splits(frame)
    assert "0007" in str(error.value) and "0009" in str(error.value)


@pytest.mark.parametrize("split", [TRAIN, VAL])
def test_excluded_subject_in_train_or_val_fails(split) -> None:
    with pytest.raises(SplitLeakageError, match="excluded subjects in train/val: \\['0005'\\]"):
        validate_splits(_assignment({"0005": split, "0001": TEST}), excluded_subjects=["0005"])


def test_unknown_split_name_fails() -> None:
    with pytest.raises(ValueError, match="Unknown split names"):
        validate_splits(_assignment({"0001": "holdout"}))


def test_coverage_detects_missing_test_subject() -> None:
    assignment = _assignment({"0001": TRAIN, "0002": VAL})
    with pytest.raises(SplitCoverageError, match="test: missing \\['0003'\\]"):
        validate_coverage(assignment, ["0001", "0002"], ["0003"], excluded_subjects=[])


def test_coverage_detects_excluded_subject_kept_in_train() -> None:
    assignment = _assignment({"0001": TRAIN, "0002": VAL, "0005": TRAIN, "0003": TEST})
    with pytest.raises(SplitCoverageError, match="unexpected \\['0005'\\]"):
        validate_coverage(assignment, ["0001", "0002", "0005"], ["0003"], ["0005"])


# ---------------------------------------------------------------- assign_validation


def test_excluded_subjects_are_removed_and_the_rest_covered(varied_train_manifest) -> None:
    excluded = ["0010", "0020", "0030"]
    assignment = assign_validation(varied_train_manifest, VAL_FRACTION, 0, STRATIFY_BINS, excluded)
    assert not set(excluded) & set(assignment["subject_id"])
    assert set(assignment["subject_id"]) == set(varied_train_manifest["subject_id"]) - set(excluded)
    assert assignment["subject_id"].is_unique
    assert set(assignment["split"]) == {TRAIN, VAL}


def test_val_size_follows_the_fraction(varied_train_manifest) -> None:
    assignment = assign_validation(varied_train_manifest, VAL_FRACTION, 0, STRATIFY_BINS, [])
    assert (assignment["split"] == VAL).sum() == round(300 * VAL_FRACTION)


def test_same_seed_gives_same_assignment(varied_train_manifest) -> None:
    first = assign_validation(varied_train_manifest, VAL_FRACTION, 7, STRATIFY_BINS, [])
    second = assign_validation(varied_train_manifest, VAL_FRACTION, 7, STRATIFY_BINS, [])
    pd.testing.assert_frame_equal(first, second)


def test_different_seed_gives_different_assignment(varied_train_manifest) -> None:
    first = assign_validation(varied_train_manifest, VAL_FRACTION, 0, STRATIFY_BINS, [])
    second = assign_validation(varied_train_manifest, VAL_FRACTION, 1, STRATIFY_BINS, [])
    assert not first.equals(second)


def test_assignment_does_not_depend_on_row_order(varied_train_manifest) -> None:
    shuffled = varied_train_manifest.sample(frac=1, random_state=0)
    first = assign_validation(varied_train_manifest, VAL_FRACTION, 3, STRATIFY_BINS, [])
    second = assign_validation(shuffled, VAL_FRACTION, 3, STRATIFY_BINS, [])
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("seed", range(5))
def test_stratification_keeps_spoof_ratio_close(varied_train_manifest, seed) -> None:
    assignment = assign_validation(varied_train_manifest, VAL_FRACTION, seed, STRATIFY_BINS, [])
    split_of = assignment.set_index("subject_id")["split"]
    rows = varied_train_manifest.assign(split=varied_train_manifest["subject_id"].map(split_of))
    spoof_ratio = (rows["label"] == labels.LABEL_SPOOF).groupby(rows["split"]).mean()
    assert abs(spoof_ratio[VAL] - spoof_ratio[TRAIN]) <= SPOOF_RATIO_TOLERANCE


def test_single_image_subjects(make_labels) -> None:
    subjects = {f"{i:04d}": (i % 2, 1 - i % 2) for i in range(20)}
    assignment = assign_validation(_manifest(make_labels, subjects), 0.2, 0, STRATIFY_BINS, [])
    assert (assignment["split"] == VAL).sum() == 4
    assert set(assignment["subject_id"]) == set(subjects)
    validate_splits(assignment)


def test_tiny_dataset_with_two_subjects(make_labels) -> None:
    manifest = _manifest(make_labels, {"0001": (1, 1), "0002": (1, 0)})
    assignment = assign_validation(manifest, 0.5, 0, STRATIFY_BINS, [])
    assert sorted(assignment["split"]) == [TRAIN, VAL]


def test_fraction_that_empties_val_fails(make_labels) -> None:
    manifest = _manifest(make_labels, {"0001": (1, 1), "0002": (1, 0), "0003": (0, 1)})
    with pytest.raises(ValueError, match="leaves train or val empty"):
        assign_validation(manifest, VAL_FRACTION, 0, STRATIFY_BINS, [])


def test_no_subjects_left_after_exclusion_fails(make_labels) -> None:
    manifest = _manifest(make_labels, {"0001": (1, 1)})
    with pytest.raises(ValueError, match="No train subjects"):
        assign_validation(manifest, VAL_FRACTION, 0, STRATIFY_BINS, ["0001"])


# ---------------------------------------------------------------- test-split tripwire


@pytest.fixture
def conflicting_test_labels(make_labels, make_vector, image_path) -> dict[str, list[int]]:
    """Test-split labels with one image under ``live/`` labelled spoof."""
    label_vectors = make_labels(TEST, {"0101": (1, 1)})
    path = image_path(TEST, "0101", labels.PATH_KIND_LIVE, 50)
    label_vectors[path] = make_vector(labels.LABEL_SPOOF, spoof_type=1)
    return label_vectors


@pytest.mark.parametrize("policy", [CONFLICT_EXCLUDE, CONFLICT_TRUST_PATH])
def test_policy_that_modifies_test_rows_is_refused(conflicting_test_labels, policy) -> None:
    _, stats = build_manifest(conflicting_test_labels, TEST, policy)
    with pytest.raises(ProtectedTestSplitError):
        ensure_test_untouched(stats)


def test_trust_label_leaves_test_rows_untouched(conflicting_test_labels) -> None:
    _, stats = build_manifest(conflicting_test_labels, TEST, CONFLICT_TRUST_LABEL)
    ensure_test_untouched(stats)


def test_tripwire_rejects_train_stats(make_labels) -> None:
    _, stats = build_manifest(make_labels(TRAIN, {"0001": (1, 1)}), TRAIN, CONFLICT_EXCLUDE)
    with pytest.raises(ValueError, match="test-split stats"):
        ensure_test_untouched(stats)


# ---------------------------------------------------------------- end-to-end build


def _write_label_files(config, train_labels, test_labels) -> None:
    for source_split, label_vectors in ((TRAIN, train_labels), (TEST, test_labels)):
        path = config.label_path(source_split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(label_vectors), encoding="utf-8")


@pytest.fixture
def leaking_dataset(tmp_data_config, make_labels):
    """40 train subjects and 10 test subjects, plus one configured excluded subject in both."""
    leak = tmp_data_config.excluded_subjects[0]
    train_subjects = {f"{i:04d}": (2, 3) for i in range(40)} | {leak: (1, 1)}
    test_subjects = {f"{i:04d}": (1, 2) for i in range(100, 110)} | {leak: (2, 2)}
    train_labels = make_labels(TRAIN, train_subjects)
    test_labels = make_labels(TEST, test_subjects)
    _write_label_files(tmp_data_config, train_labels, test_labels)
    return leak, train_labels, test_labels


def test_build_excludes_leaking_subject_from_train_only(tmp_data_config, leaking_dataset) -> None:
    leak, train_labels, test_labels = leaking_dataset
    config = tmp_data_config
    report = run(config)

    assignment = load_split_assignment(config.split_assignment_path, config.excluded_subjects)
    assert list(assignment.columns) == ["subject_id", "split"]
    assert assignment.loc[assignment["subject_id"] == leak, "split"].tolist() == [TEST]
    assert report.source_overlap_subjects == (leak,)

    test_manifest = pd.read_csv(report.manifest_paths[TEST], dtype={"subject_id": str})
    assert set(test_manifest["image_path"]) == set(test_labels)
    assert report.summaries[TEST].rows == len(test_labels)

    stacked = pd.concat(
        pd.read_csv(path, dtype={"subject_id": str}) for path in report.manifest_paths.values()
    )
    validate_splits(stacked, config.excluded_subjects)
    trainval = stacked.loc[stacked["split"] != TEST]
    assert leak not in set(trainval["subject_id"])
    assert len(trainval) == len(train_labels) - 2  # the leaking subject's two train images
    assert "Final splits" in format_report(report)


def test_build_without_exclusion_fails_before_writing(tmp_data_config, leaking_dataset) -> None:
    leak, _, _ = leaking_dataset
    config = dataclasses.replace(tmp_data_config, excluded_subjects=())
    with pytest.raises(SplitLeakageError, match=leak):
        run(config)
    assert not config.manifest_dir.exists()
    assert not config.split_assignment_path.exists()


def test_build_refuses_to_modify_test_split(
    tmp_data_config, make_labels, conflicting_test_labels
) -> None:
    _write_label_files(
        tmp_data_config, make_labels(TRAIN, {"0001": (1, 1)}), conflicting_test_labels
    )
    with pytest.raises(ProtectedTestSplitError):
        run(tmp_data_config)
    assert not tmp_data_config.manifest_dir.exists()


def test_loading_assignment_with_shared_subject_fails(tmp_path) -> None:
    path = tmp_path / "split_assignment.csv"
    _assignment({"0001": TRAIN, "0002": TEST}).pipe(
        lambda frame: pd.concat([frame, _assignment({"0001": TEST})])
    ).to_csv(path, index=False)
    with pytest.raises(SplitLeakageError, match="0001"):
        load_split_assignment(path)


def test_loading_assignment_keeps_leading_zeros_and_checks_columns(tmp_path) -> None:
    path = tmp_path / "split_assignment.csv"
    _assignment({"0001": TRAIN, "0002": VAL, "0003": TEST}).to_csv(path, index=False)
    assert load_split_assignment(path)["subject_id"].tolist() == ["0001", "0002", "0003"]
    path.write_text("subject,split\n0001,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected columns"):
        load_split_assignment(path)
