"""Subject-level validation split and the subject-disjointness invariant.

The official CelebA-Spoof test split is used unchanged. Validation subjects are carved out of the
official train split after removing ``excluded_subjects`` (subjects that also appear in test).
:func:`validate_splits` is the single source of truth for the disjointness invariant in
``docs/SCHEMA.md`` §2: call it wherever a split is produced or loaded.
"""

import logging
from collections.abc import Collection, Iterable
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from antispoof.data import labels
from antispoof.data.manifest import ManifestStats

logger = logging.getLogger(__name__)

SPLIT_ASSIGNMENT_COLUMNS = ("subject_id", "split")


class SplitLeakageError(ValueError):
    """Raised when a subject is in more than one split, or an excluded subject is in train/val."""


class SplitCoverageError(ValueError):
    """Raised when a split assignment does not cover exactly the expected subjects."""


class ProtectedTestSplitError(ValueError):
    """Raised when a build would drop or relabel rows of the official test split."""


def validate_splits(assignment: pd.DataFrame, excluded_subjects: Collection[str] = ()) -> None:
    """Assert that train, val and test are pairwise subject-disjoint.

    Accepts a split assignment (one row per subject) or a stacked manifest (one row per image).
    Only the ``subject_id`` and ``split`` columns are read.

    Args:
        assignment: Frame with ``subject_id`` and ``split`` columns.
        excluded_subjects: Subjects that must not appear in train or val.

    Raises:
        ValueError: If a column is missing or a split name is unknown.
        SplitLeakageError: If any pairwise subject intersection is non-empty, or an excluded
            subject is in train or val. The message lists the offending subject ids.
    """
    subjects = _subjects_by_split(assignment)
    problems = [
        f"{first} and {second} share {sorted(subjects[first] & subjects[second])}"
        for first, second in combinations(labels.SPLITS, 2)
        if subjects[first] & subjects[second]
    ]
    leaked = set(excluded_subjects) & (subjects[labels.SPLIT_TRAIN] | subjects[labels.SPLIT_VAL])
    if leaked:
        problems.append(f"excluded subjects in train/val: {sorted(leaked)}")
    if problems:
        raise SplitLeakageError("Subject disjointness violated: " + "; ".join(problems))


def validate_coverage(
    assignment: pd.DataFrame,
    source_train_subjects: Iterable[str],
    source_test_subjects: Iterable[str],
    excluded_subjects: Collection[str],
) -> None:
    """Assert ``S_test = T`` and ``S_train ∪ S_val = R \\ E`` (``docs/SCHEMA.md`` §2).

    Args:
        assignment: Frame with ``subject_id`` and ``split`` columns.
        source_train_subjects: R, the subjects of the official train manifest.
        source_test_subjects: T, the subjects of the official test manifest.
        excluded_subjects: E, the subjects removed from train/val.

    Raises:
        ValueError: If a column is missing or a split name is unknown.
        SplitCoverageError: If either equation fails. The message lists missing and unexpected ids.
    """
    subjects = _subjects_by_split(assignment)
    expected = {
        "test": set(source_test_subjects),
        "train+val": set(source_train_subjects) - set(excluded_subjects),
    }
    actual = {
        "test": subjects[labels.SPLIT_TEST],
        "train+val": subjects[labels.SPLIT_TRAIN] | subjects[labels.SPLIT_VAL],
    }
    problems = [
        f"{name}: missing {sorted(expected[name] - actual[name])}, "
        f"unexpected {sorted(actual[name] - expected[name])}"
        for name in expected
        if expected[name] != actual[name]
    ]
    if problems:
        raise SplitCoverageError("Split coverage violated: " + "; ".join(problems))


def ensure_test_untouched(test_stats: ManifestStats) -> None:
    """Refuse a build whose conflict policy would drop or relabel official test rows.

    Args:
        test_stats: Stats from building the test manifest.

    Raises:
        ValueError: If the stats are not for the test split.
        ProtectedTestSplitError: If any test row was dropped or relabelled.
    """
    if test_stats.source_split != labels.SPLIT_TEST:
        raise ValueError(f"Expected test-split stats, got {test_stats.source_split!r}.")
    if test_stats.conflicts_dropped or test_stats.conflicts_relabelled:
        raise ProtectedTestSplitError(
            f"conflict_policy={test_stats.conflict_policy} would drop "
            f"{test_stats.conflicts_dropped} and relabel {test_stats.conflicts_relabelled} test "
            "rows. The official test split is never modified (ADR-009)."
        )


def assign_validation(
    train_manifest: pd.DataFrame,
    val_fraction: float,
    seed: int,
    stratify_bins: int,
    excluded_subjects: Collection[str],
) -> pd.DataFrame:
    """Assign official-train subjects to train or val, stratified by spoof-image fraction.

    Excluded subjects are removed first. The remaining subjects are sorted by (spoof fraction,
    subject id) and cut into ``stratify_bins`` equal-count strata. ``round(val_fraction * n)`` val
    subjects are allocated across strata in proportion to their size (largest remainder), then
    drawn from each stratum with a seeded generator. The result does not depend on row order.

    Args:
        train_manifest: Official-train manifest with ``subject_id`` and ``label`` columns.
        val_fraction: Fraction of subjects assigned to val, in (0, 1).
        seed: Seed for ``numpy.random.default_rng``.
        stratify_bins: Maximum number of strata.
        excluded_subjects: Subjects removed before assignment.

    Returns:
        Frame with columns ``subject_id`` and ``split`` (``train``/``val``), sorted by subject id.

    Raises:
        ValueError: If no subjects remain, or the fraction leaves train or val empty.
    """
    kept = train_manifest.loc[~train_manifest["subject_id"].isin(list(excluded_subjects))]
    strata = _stratify_subjects(kept, stratify_bins)
    n_val = _validation_count(len(strata), val_fraction)
    sizes = strata["stratum"].value_counts().sort_index().to_numpy()
    rng = np.random.default_rng(seed)
    val_ids: list[str] = []
    for stratum, quota in enumerate(_stratum_quotas(sizes, val_fraction, n_val)):
        members = strata.loc[strata["stratum"] == stratum, "subject_id"].to_numpy()
        val_ids.extend(members[rng.permutation(len(members))[:quota]].tolist())
    is_val = strata["subject_id"].isin(val_ids).to_numpy()
    assignment = pd.DataFrame(
        {
            "subject_id": strata["subject_id"],
            "split": np.where(is_val, labels.SPLIT_VAL, labels.SPLIT_TRAIN),
        }
    )
    logger.info("Assigned %d subjects to val and %d to train.", n_val, len(strata) - n_val)
    return assignment.sort_values("subject_id").reset_index(drop=True)


def load_split_assignment(path: Path, excluded_subjects: Collection[str] = ()) -> pd.DataFrame:
    """Load ``split_assignment.csv`` and validate it.

    Args:
        path: Path to the CSV with columns ``subject_id,split``.
        excluded_subjects: Subjects that must not appear in train or val.

    Returns:
        The assignment, with ``subject_id`` kept as strings.

    Raises:
        ValueError: If the columns are wrong or a subject id is listed twice.
        SplitLeakageError: If the invariant fails (see :func:`validate_splits`).
    """
    assignment = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(assignment.columns) != SPLIT_ASSIGNMENT_COLUMNS:
        raise ValueError(f"{path}: expected columns {SPLIT_ASSIGNMENT_COLUMNS}.")
    validate_splits(assignment, excluded_subjects)
    duplicated = assignment.loc[assignment["subject_id"].duplicated(), "subject_id"]
    if len(duplicated):
        raise ValueError(f"{path}: subject ids listed more than once: {sorted(duplicated)}.")
    return assignment


def _subjects_by_split(assignment: pd.DataFrame) -> dict[str, set[str]]:
    missing = set(SPLIT_ASSIGNMENT_COLUMNS) - set(assignment.columns)
    if missing:
        raise ValueError(f"Split frame is missing columns {sorted(missing)}.")
    unknown = set(assignment["split"].unique()) - set(labels.SPLITS)
    if unknown:
        raise ValueError(
            f"Unknown split names {sorted(map(str, unknown))}; expected {labels.SPLITS}."
        )
    return {
        name: set(assignment.loc[assignment["split"] == name, "subject_id"].unique())
        for name in labels.SPLITS
    }


def _stratify_subjects(manifest: pd.DataFrame, stratify_bins: int) -> pd.DataFrame:
    if manifest.empty:
        raise ValueError("No train subjects left to assign.")
    is_spoof = manifest["label"] == labels.LABEL_SPOOF
    fractions = is_spoof.groupby(manifest["subject_id"]).mean()
    strata = pd.DataFrame(
        {"subject_id": fractions.index.astype(str), "spoof_fraction": fractions.to_numpy()}
    )
    strata = strata.sort_values(["spoof_fraction", "subject_id"], kind="stable")
    strata = strata.reset_index(drop=True)
    n_strata = min(stratify_bins, len(strata))
    strata["stratum"] = (np.arange(len(strata)) * n_strata) // len(strata)
    return strata


def _validation_count(n_subjects: int, val_fraction: float) -> int:
    n_val = round(n_subjects * val_fraction)
    if n_val == 0 or n_val == n_subjects:
        raise ValueError(
            f"val_fraction={val_fraction} with {n_subjects} subjects leaves train or val empty."
        )
    return n_val


def _stratum_quotas(sizes: np.ndarray, val_fraction: float, n_val: int) -> np.ndarray:
    exact = sizes * val_fraction
    quotas = np.floor(exact).astype(np.int64)
    shortfall = n_val - int(quotas.sum())
    order = np.argsort(-(exact - quotas), kind="stable")
    quotas[order[:shortfall]] += 1
    return quotas
