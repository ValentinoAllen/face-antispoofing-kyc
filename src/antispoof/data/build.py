"""Build the image manifests and the subject-level split assignment from a data config.

This is the logic behind ``scripts/build_manifest.py``. Order of operations:

1. Build the test manifest and refuse to continue if the conflict policy would drop or relabel any
   test row. The official test split is never modified.
2. Build the train manifest, applying the conflict policy.
3. Carve validation subjects out of train after removing ``excluded_subjects``.
4. Validate subject disjointness and coverage (``antispoof.data.splits``).
5. Write ``manifest_{train,val,test}.csv`` and ``split_assignment.csv``.

Nothing is written unless every validation passes.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from antispoof.data import labels
from antispoof.data.config import DataConfig
from antispoof.data.manifest import (
    ATTACK_CODE_COLUMNS,
    ManifestStats,
    build_manifest,
    load_label_json,
)
from antispoof.data.splits import (
    SPLIT_ASSIGNMENT_COLUMNS,
    assign_validation,
    ensure_test_untouched,
    validate_coverage,
    validate_splits,
)

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest_{split}.csv"


@dataclass(frozen=True)
class SplitSummary:
    """Row, subject and class counts of one final split."""

    rows: int
    subjects: int
    live: int
    spoof: int


@dataclass(frozen=True)
class CodeCounts:
    """Value counts of each attack-code column over the rows stored under ``spoof/``."""

    spoof_rows: int
    counts: dict[str, dict[int, int]]


@dataclass(frozen=True)
class SplitOutputs:
    """Final per-split manifests and the subject-level split assignment."""

    manifests: dict[str, pd.DataFrame]
    assignment: pd.DataFrame


@dataclass(frozen=True)
class BuildReport:
    """Everything the build script prints, for reconciliation against documented counts."""

    config: DataConfig
    train_stats: ManifestStats
    test_stats: ManifestStats
    source_overlap_subjects: tuple[str, ...]
    source_unique_subjects: int
    code_counts: dict[str, CodeCounts]
    summaries: dict[str, SplitSummary]
    manifest_paths: dict[str, Path]


def build_source_manifest(
    config: DataConfig, source_split: str
) -> tuple[pd.DataFrame, ManifestStats]:
    """Load one label file and build its manifest with the configured conflict policy.

    Args:
        config: The data configuration.
        source_split: ``train`` or ``test``.

    Returns:
        The manifest and its build stats.
    """
    label_vectors = load_label_json(config.label_path(source_split))
    return build_manifest(label_vectors, source_split, config.conflict_policy)


def build_splits(
    train_manifest: pd.DataFrame, test_manifest: pd.DataFrame, config: DataConfig
) -> SplitOutputs:
    """Assign train/val subjects, keep test unchanged, and validate the result.

    Args:
        train_manifest: Official-train manifest after the conflict policy.
        test_manifest: Official-test manifest, used as is.
        config: The data configuration.

    Returns:
        The train, val and test manifests plus the subject-level assignment.

    Raises:
        SplitLeakageError: If the splits share a subject (e.g. an overlap not in
            ``excluded_subjects``).
        SplitCoverageError: If the assignment does not cover exactly the expected subjects.
    """
    trainval = assign_validation(
        train_manifest,
        config.val_fraction,
        config.seed,
        config.stratify_bins,
        config.excluded_subjects,
    )
    test_subjects = sorted(test_manifest["subject_id"].unique())
    test_assignment = pd.DataFrame({"subject_id": test_subjects, "split": labels.SPLIT_TEST})
    assignment = pd.concat([trainval, test_assignment], ignore_index=True)
    validate_splits(assignment, config.excluded_subjects)
    validate_coverage(
        assignment,
        train_manifest["subject_id"],
        test_manifest["subject_id"],
        config.excluded_subjects,
    )
    manifests = _split_train_manifest(train_manifest, trainval)
    manifests[labels.SPLIT_TEST] = test_manifest
    stacked = pd.concat([frame[list(SPLIT_ASSIGNMENT_COLUMNS)] for frame in manifests.values()])
    validate_splits(stacked, config.excluded_subjects)
    return SplitOutputs(manifests=manifests, assignment=assignment)


def write_outputs(
    outputs: SplitOutputs, manifest_dir: Path, split_assignment_path: Path
) -> dict[str, Path]:
    """Write the three manifests and the split assignment as CSV.

    Args:
        outputs: Result of :func:`build_splits`.
        manifest_dir: Directory for ``manifest_{split}.csv``.
        split_assignment_path: Path for ``split_assignment.csv``.

    Returns:
        Mapping from split name to the manifest path written.
    """
    manifest_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, manifest in outputs.manifests.items():
        paths[name] = manifest_dir / MANIFEST_FILENAME.format(split=name)
        manifest.sort_values("image_path").to_csv(paths[name], index=False)
    split_assignment_path.parent.mkdir(parents=True, exist_ok=True)
    outputs.assignment.sort_values("subject_id").to_csv(split_assignment_path, index=False)
    return paths


def summarize_split(manifest: pd.DataFrame) -> SplitSummary:
    """Count rows, subjects, live and spoof images in one manifest.

    Args:
        manifest: A manifest with ``subject_id`` and ``label`` columns.

    Returns:
        The counts.
    """
    return SplitSummary(
        rows=len(manifest),
        subjects=int(manifest["subject_id"].nunique()),
        live=int((manifest["label"] == labels.LABEL_LIVE).sum()),
        spoof=int((manifest["label"] == labels.LABEL_SPOOF).sum()),
    )


def count_attack_codes(manifest: pd.DataFrame) -> CodeCounts:
    """Count the values of each attack-code column on rows stored under ``spoof/``.

    These are the index 40–42 distributions recorded in ``docs/SCHEMA.md`` §1.1. Codes are
    1-indexed; a ``labels.CODE_NOT_APPLICABLE`` value here would break the documented convention.

    Args:
        manifest: A manifest with ``path_kind`` and the ``ATTACK_CODE_COLUMNS``.

    Returns:
        The number of ``spoof/`` rows and, per column, a mapping from code to row count, sorted by
        code.
    """
    spoof_rows = manifest.loc[manifest["path_kind"] == labels.PATH_KIND_SPOOF]
    counts = {
        column: {
            int(code): int(count)
            for code, count in spoof_rows[column].value_counts().sort_index().items()
        }
        for column in ATTACK_CODE_COLUMNS
    }
    return CodeCounts(spoof_rows=len(spoof_rows), counts=counts)


def run(config: DataConfig) -> BuildReport:
    """Build, validate and write the manifests and the split assignment.

    Args:
        config: The data configuration.

    Returns:
        A report of every count needed for reconciliation.

    Raises:
        ProtectedTestSplitError: If the conflict policy would modify the test split.
        SplitLeakageError: If the final splits share a subject.
        SplitCoverageError: If the assignment does not cover exactly the expected subjects.
    """
    test_manifest, test_stats = build_source_manifest(config, labels.SPLIT_TEST)
    ensure_test_untouched(test_stats)
    train_manifest, train_stats = build_source_manifest(config, labels.SPLIT_TRAIN)
    train_subjects = set(train_manifest["subject_id"])
    test_subjects = set(test_manifest["subject_id"])
    overlap = tuple(sorted(train_subjects & test_subjects))
    logger.info("Subjects in both source splits: %s", ", ".join(overlap) or "none")
    outputs = build_splits(train_manifest, test_manifest, config)
    manifest_paths = write_outputs(outputs, config.manifest_dir, config.split_assignment_path)
    return BuildReport(
        config=config,
        train_stats=train_stats,
        test_stats=test_stats,
        source_overlap_subjects=overlap,
        source_unique_subjects=len(train_subjects | test_subjects),
        code_counts={
            labels.SPLIT_TRAIN: count_attack_codes(train_manifest),
            labels.SPLIT_TEST: count_attack_codes(test_manifest),
        },
        summaries={name: summarize_split(frame) for name, frame in outputs.manifests.items()},
        manifest_paths=manifest_paths,
    )


def format_report(report: BuildReport) -> str:
    """Render a build report as plain text for the console.

    Args:
        report: Result of :func:`run`.

    Returns:
        Multi-line text.
    """
    config = report.config
    lines = [
        "Resolved data config",
        f"  dataset_root: {config.dataset_root}",
        f"  protocol: {config.protocol}",
        f"  conflict_policy: {config.conflict_policy}",
        f"  excluded_subjects: {', '.join(config.excluded_subjects) or 'none'}",
        f"  val_fraction: {config.val_fraction}, seed: {config.seed}, "
        f"stratify_bins: {config.stratify_bins}",
        "",
        "Source label files",
        *_format_stats(report.train_stats),
        *_format_stats(report.test_stats),
        f"  total rows in: {report.train_stats.rows_in + report.test_stats.rows_in:,}",
        "  after conflict policy: unique subjects "
        f"{report.source_unique_subjects:,}; in both train and test: "
        f"{', '.join(report.source_overlap_subjects) or 'none'}",
        "",
        *_format_code_counts(report.code_counts),
        "",
        "Final splits",
        *(
            f"  {name}: rows={s.rows:,} subjects={s.subjects:,} live={s.live:,} spoof={s.spoof:,}"
            for name, s in report.summaries.items()
        ),
        "",
        "Written",
        *(f"  {path}" for path in report.manifest_paths.values()),
        f"  {config.split_assignment_path}",
    ]
    return "\n".join(lines)


def _split_train_manifest(
    train_manifest: pd.DataFrame, trainval: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    split_of = trainval.set_index("subject_id")["split"]
    kept = train_manifest.loc[train_manifest["subject_id"].isin(split_of.index)].copy()
    kept["split"] = kept["subject_id"].map(split_of)
    return {
        name: kept.loc[kept["split"] == name].reset_index(drop=True)
        for name in (labels.SPLIT_TRAIN, labels.SPLIT_VAL)
    }


def _format_code_counts(code_counts: dict[str, CodeCounts]) -> list[str]:
    lines = ["Attack codes on rows under spoof/, after conflict policy (1-indexed; 0 = n/a)"]
    for source_split, counts in code_counts.items():
        lines.append(f"  {source_split}: spoof/ rows={counts.spoof_rows:,}")
        lines.extend(
            f"    {column}: "
            + " ".join(f"{code}={count:,}" for code, count in by_code.items())
            + f" ({len(by_code)} distinct)"
            for column, by_code in counts.counts.items()
        )
    return lines


def _format_stats(stats: ManifestStats) -> list[str]:
    return [
        f"  {stats.source_split}: rows={stats.rows_in:,} subjects={stats.subjects_in:,}",
        f"    by path segment: live={stats.live_by_path:,} spoof={stats.spoof_by_path:,}",
        f"    by index {labels.INDEX_LABEL}: live={stats.live_by_label:,} "
        f"spoof={stats.spoof_by_label:,}",
        f"    conflicts: found={stats.conflicts_found:,} dropped={stats.conflicts_dropped:,} "
        f"relabelled={stats.conflicts_relabelled:,}",
        f"    after {stats.conflict_policy}: rows={stats.rows_out:,} "
        f"subjects={stats.subjects_out:,} live={stats.live_out:,} spoof={stats.spoof_out:,}",
    ]
