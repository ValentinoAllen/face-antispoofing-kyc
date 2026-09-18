"""Full-split audit of JPEG quantization tables, read from image headers only.

Part 1 of ``20260918-131447-counterfactual_jpeg`` found one table set per class, matching Pillow
quality 75 (live) and 95 (spoof), on the baseline's 4,000-row train and 2,000-row val subsets. This
module repeats that measurement over **every row** of ``manifest_train.csv``, ``manifest_val.csv``
and ``manifest_test.csv``, reports the distinct table sets per split and per class with their row
counts and shares, counts the formats, modes and subsamplings, and cross-tabulates table set against
``spoof_type`` for the attack rows, because per-species APCER later depends on whether attack types
differ in encoding.

The matcher is the one in :mod:`antispoof.eval.jpeg_tables`; nothing is matched twice.
:func:`antispoof.eval.jpeg_tables.class_encodings` is deliberately not called: it raises unless
there is exactly one table set per class, which a full-split audit must not assume.

Rows whose header cannot be read are counted and listed by path, and do not stop the run. Every
other figure covers the rows whose header was read.

Reading ``manifest_test.csv`` does not spend the single test evaluation of ``docs/RULES.md`` §3:
this run reads test **headers only**, decodes no pixel, builds no model and produces no
model-selection signal. It computes no PAD metrics, so its record's ``metrics`` stay null and it
gets no ``docs/EXPERIMENTS.md`` row.
"""

import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from PIL import Image

from antispoof.data import labels
from antispoof.data.build import summarize_split
from antispoof.data.config import DataConfig
from antispoof.data.dataset import ImageLoadError
from antispoof.data.manifest import read_manifest
from antispoof.eval.jpeg_tables import CLASS_NAME, JpegTables, audit_quantization, jpeg_tables
from antispoof.eval.metadata_probe import CLASS_NAMES, SUBSAMPLING_NAMES, SUBSAMPLING_OTHER
from antispoof.training import reproducibility
from antispoof.training.config import DEVICE_CPU, TrainConfigError, parse_section
from antispoof.training.run import (
    RECORD_FILENAME,
    RESOLVED_CONFIG_FILENAME,
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    RecordHeader,
    build_record,
    close_failed_record,
    manifest_paths,
    write_json,
)

logger = logging.getLogger(__name__)

AUDIT_SUMMARY_FILENAME = "audit_summary.json"

AUDIT_COLUMNS = ("split", "label", "spoof_type", "format", "tables")
"""Columns of :func:`read_audit_headers`. ``tables`` holds interned :class:`JpegTables` objects."""

SPOOF_CLASS = CLASS_NAME[labels.LABEL_SPOOF]

NO_LEDGER_ROW = (
    "This run measures the data, not a model: it computes no PAD metrics, "
    "so it gets no docs/EXPERIMENTS.md row."
)


class AuditConfigError(ValueError):
    """Raised when the audit config is missing keys or holds invalid values."""


@dataclass(frozen=True)
class HeaderFailure:
    """One manifest row whose image header could not be read."""

    image_path: str
    split: str
    error: str


@dataclass(frozen=True)
class AuditRunConfig:
    """Run description and seed (``run:``)."""

    hypothesis: str
    what_changed: str
    notes: str
    seed: int


@dataclass(frozen=True)
class AuditScanConfig:
    """Which manifests to scan and how often to report progress (``audit:``)."""

    splits: tuple[str, ...]
    progress_every: int
    max_listed_failures: int


@dataclass(frozen=True)
class AuditConfig:
    """A resolved audit config. Fields are documented in the YAML file."""

    run: AuditRunConfig
    audit: AuditScanConfig

    def to_dict(self) -> dict[str, Any]:
        """Return the config as nested plain dicts, for hashing and the resolved-config file."""
        return asdict(self)


_SECTION_TYPES = {"run": AuditRunConfig}


@dataclass(frozen=True)
class AuditInputs:
    """Everything an audit run needs. ``data_config`` already carries CLI path overrides.

    Attributes:
        config: The audit config.
        config_path: Path of the audit config, under ``configs/``.
        data_config: Manifest directory, dataset root and split assignment.
        output_dir: Parent of the new run directory.
        repo_root: The repository root.
    """

    config: AuditConfig
    config_path: Path
    data_config: DataConfig
    output_dir: Path
    repo_root: Path


@dataclass(frozen=True)
class AuditResult:
    """The final run record, its directory and the summary."""

    record: dict[str, Any]
    run_dir: Path
    summary: dict[str, Any]


def load_audit_config(path: Path) -> AuditConfig:
    """Load a YAML such as ``configs/audit_jpeg.yaml`` into a validated config.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        AuditConfigError: If sections or keys are missing or unknown, a value has the wrong type,
            or a value is invalid.
    """
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    names = {field.name for field in fields(AuditConfig)}
    if not isinstance(document, dict) or set(document) != names:
        raise AuditConfigError(f"{path}: expected exactly the top-level sections {sorted(names)}.")
    try:
        sections = {
            name: parse_section(name, document[name], kind) for name, kind in _SECTION_TYPES.items()
        }
    except TrainConfigError as error:
        raise AuditConfigError(f"{path}: {error}") from error
    config = AuditConfig(**sections, audit=_parse_scan(document["audit"]))
    validate_audit_config(config)
    return config


def _parse_scan(section: object) -> AuditScanConfig:
    expected = {"splits", "progress_every", "max_listed_failures"}
    if not isinstance(section, dict) or set(section) != expected:
        raise AuditConfigError(f"Section 'audit' must hold exactly the keys {sorted(expected)}.")
    splits = section["splits"]
    if not isinstance(splits, list) or not all(isinstance(split, str) for split in splits):
        raise AuditConfigError(f"audit.splits: expected a list of split names: {splits!r}.")
    counts = {
        "progress_every": section["progress_every"],
        "max_listed_failures": section["max_listed_failures"],
    }
    for key, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise AuditConfigError(f"audit.{key}: expected int, got {value!r}.")
    return AuditScanConfig(splits=tuple(splits), **counts)


def validate_audit_config(config: AuditConfig) -> None:
    """Check the run description, the split names and the counters.

    Args:
        config: The configuration.

    Raises:
        AuditConfigError: Listing every violated constraint.
    """
    splits = config.audit.splits
    checks = [
        (bool(config.run.hypothesis.strip()), "run.hypothesis must not be empty"),
        (bool(config.run.what_changed.strip()), "run.what_changed must not be empty"),
        (config.run.seed >= 0, "run.seed must be >= 0"),
        (bool(splits), "audit.splits must not be empty"),
        (set(splits) <= set(labels.SPLITS), f"audit.splits must be in {labels.SPLITS}"),
        (len(set(splits)) == len(splits), "audit.splits must not repeat a split"),
        (config.audit.progress_every >= 1, "audit.progress_every must be >= 1"),
        (config.audit.max_listed_failures >= 0, "audit.max_listed_failures must be >= 0"),
    ]
    problems = [message for passed, message in checks if not passed]
    if problems:
        raise AuditConfigError("Invalid audit config: " + "; ".join(problems) + ".")


def load_manifests(splits: Sequence[str], data_config: DataConfig) -> dict[str, pd.DataFrame]:
    """Read every row of the manifest of each split. No subsetting is applied.

    Args:
        splits: Splits to read, in reading order.
        data_config: Holds the manifest directory.

    Returns:
        ``{split: manifest}``.

    Raises:
        ManifestError: If a manifest's columns do not match the contract.
        ValueError: If a manifest holds rows of another split.
    """
    manifests: dict[str, pd.DataFrame] = {}
    for split, path in manifest_paths(data_config, splits).items():
        manifest = read_manifest(path)
        wrong_split = manifest["split"] != split
        if wrong_split.any():
            raise ValueError(f"{path}: {int(wrong_split.sum())} rows are not in split {split!r}.")
        manifests[split] = manifest
        summary = summarize_split(manifest)
        logger.info(
            "%s manifest: %d rows, %d subjects (live %d, spoof %d).",
            split,
            summary.rows,
            summary.subjects,
            summary.live,
            summary.spoof,
        )
    return manifests


def read_audit_headers(
    manifests: Mapping[str, pd.DataFrame], dataset_root: Path, progress_every: int
) -> tuple[pd.DataFrame, list[HeaderFailure]]:
    """Read the JPEG header of every manifest row. No pixel data is decoded.

    Equal quantization table sets are interned, so the returned frame holds one
    :class:`JpegTables` object per distinct table set rather than one per row. Over the full
    manifests that is the difference between a few objects and hundreds of millions of integers.

    Args:
        manifests: ``{split: manifest}``, each with ``image_path``, ``label`` and ``spoof_type``.
        dataset_root: Directory the ``image_path`` values are relative to.
        progress_every: Log progress after this many rows.

    Returns:
        ``(headers, failures)``: a frame with ``AUDIT_COLUMNS``, one row per readable image, and the
        rows whose header could not be read.
    """
    interned: dict[JpegTables, JpegTables] = {}
    failures: list[HeaderFailure] = []
    rows: list[tuple[str, int, int, str, JpegTables]] = []
    total = sum(len(frame) for frame in manifests.values())
    scanned = 0
    for split, frame in manifests.items():
        for image_path, label, spoof_type in zip(
            frame["image_path"], frame["label"], frame["spoof_type"], strict=True
        ):
            scanned += 1
            try:
                image_format, tables = _read_audit_header(dataset_root / image_path)
            except ImageLoadError as error:
                failures.append(HeaderFailure(str(image_path), split, str(error)))
            else:
                canonical = interned.setdefault(tables, tables)
                rows.append((split, int(label), int(spoof_type), image_format, canonical))
            if scanned % progress_every == 0:
                _log_progress(scanned, total, len(failures), len(interned))
    _log_progress(scanned, total, len(failures), len(interned))
    return pd.DataFrame(rows, columns=list(AUDIT_COLUMNS)), failures


def _log_progress(scanned: int, total: int, failed: int, distinct: int) -> None:
    logger.info(
        "Read %d of %d headers; %d unreadable, %d distinct table sets so far.",
        scanned,
        total,
        failed,
        distinct,
    )


def _read_audit_header(path: Path) -> tuple[str, JpegTables]:
    """Read the format and quantization tables of one image without decoding pixels.

    Args:
        path: Image file path.

    Returns:
        ``(format, tables)``.

    Raises:
        ImageLoadError: If the file is missing, is not a JPEG, or its header cannot be parsed.
    """
    if not path.is_file():
        raise ImageLoadError(f"Image file not found: {path}")
    try:
        with Image.open(path) as image:
            return str(image.format), jpeg_tables(image, str(path))
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        raise ImageLoadError(f"Cannot read image header {path}: {error}") from error


def add_shares(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Add each table set's share of its class's rows, per split.

    Args:
        audit: From :func:`antispoof.eval.jpeg_tables.audit_quantization`.

    Returns:
        The same structure, with a ``share`` mapping added to every table-set entry. A share is
        null for a split with no rows of that class.
    """
    with_shares: dict[str, Any] = {}
    for class_name, entry in audit.items():
        class_rows = entry["rows"]
        table_sets = [
            {
                **table_set,
                "share": {
                    split: (count / class_rows[split] if class_rows[split] else None)
                    for split, count in table_set["rows"].items()
                },
            }
            for table_set in entry["table_sets"]
        ]
        with_shares[class_name] = {**entry, "table_sets": table_sets}
    return with_shares


def count_header_values(headers: pd.DataFrame) -> dict[str, Any]:
    """Count the format, mode and subsampling of the read headers, per split and per class.

    Args:
        headers: From :func:`read_audit_headers`.

    Returns:
        ``{split: {class: {"rows", "format", "mode", "subsampling"}}}``, each a value-count map.
    """
    counts: dict[str, Any] = {}
    for split in sorted(set(headers["split"])):
        per_class: dict[str, Any] = {}
        for value, name in CLASS_NAMES:
            rows = headers[(headers["split"] == split) & (headers["label"] == value)]
            tables = list(rows["tables"])
            per_class[name] = {
                "rows": int(len(rows)),
                "format": _value_counts(rows["format"]),
                "mode": _value_counts(item.mode for item in tables),
                "subsampling": _value_counts(
                    SUBSAMPLING_NAMES.get(item.subsampling, SUBSAMPLING_OTHER) for item in tables
                ),
            }
        counts[split] = per_class
    return counts


def _value_counts(values: Iterable[str]) -> dict[str, int]:
    return {value: int(count) for value, count in sorted(Counter(values).items())}


def spoof_type_crosstab(headers: pd.DataFrame, audit: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-tabulate each spoof table set against ``spoof_type``, per split.

    If attack types differ in their encoding, a per-species APCER computed later is partly a
    per-encoding APCER. This makes that visible before any per-species metric is reported.

    Args:
        headers: From :func:`read_audit_headers`.
        audit: From :func:`antispoof.eval.jpeg_tables.audit_quantization`, whose table sets are
            ordered by row count; ``rank`` is the position in that order, starting at 1.

    Returns:
        ``{"spoof_type_codes": [...], "table_sets": [{"rank", "match", "luma_mean", "counts"}]}``,
        where ``counts`` maps split to a map from ``spoof_type`` code to a row count.
    """
    spoof = headers[headers["label"] == labels.LABEL_SPOOF]
    codes = sorted({int(code) for code in spoof["spoof_type"]})
    splits = sorted(set(headers["split"]))
    rank_of = {
        _tables_from_entry(entry): rank
        for rank, entry in enumerate(audit[SPOOF_CLASS]["table_sets"], start=1)
    }
    tally = Counter(
        zip(
            (rank_of[item] for item in spoof["tables"]),
            spoof["split"],
            (int(code) for code in spoof["spoof_type"]),
            strict=True,
        )
    )
    table_sets = [
        {
            "rank": rank,
            "match": entry["match"],
            "luma_mean": entry["luma_mean"],
            "counts": {
                split: {str(code): tally[(rank, split, code)] for code in codes} for split in splits
            },
        }
        for rank, entry in enumerate(audit[SPOOF_CLASS]["table_sets"], start=1)
    ]
    return {"spoof_type_codes": codes, "table_sets": table_sets}


def _tables_from_entry(entry: Mapping[str, Any]) -> JpegTables:
    return JpegTables(
        mode=entry["mode"],
        subsampling=entry["subsampling_code"],
        tables=tuple(tuple(table) for table in entry["tables"]),
    )


def audit_resolved_config(inputs: AuditInputs) -> dict[str, Any]:
    """Return the fully resolved config that is hashed and saved with the run.

    Args:
        inputs: The audit inputs.

    Returns:
        JSON-compatible ``{"experiment", "data"}``.
    """
    resolved: dict[str, Any] = reproducibility.to_json_compatible(
        {"experiment": inputs.config.to_dict(), "data": asdict(inputs.data_config)}
    )
    return resolved


def run_audit(inputs: AuditInputs) -> AuditResult:
    """Scan every row of the configured manifests and write the run directory.

    Writes ``record.json`` and ``resolved_config.json`` (``docs/SCHEMA.md`` §3) and
    ``audit_summary.json`` to ``<output_dir>/<run_id>/``. The summary is rewritten as results
    arrive, so an interrupted run keeps what it had. No PAD metric is computed, so the record's
    ``metrics`` stay null.

    Args:
        inputs: Configs and paths.

    Returns:
        The completed record, the run directory and the summary.

    Raises:
        AuditConfigError, ManifestError, ValueError, ProvenanceError: Before the run directory is
            created.
        Exception: Anything raised during the scan. The record is first rewritten with status
            ``failed`` (``aborted`` on ``KeyboardInterrupt``).
    """
    validate_audit_config(inputs.config)
    manifests = load_manifests(inputs.config.audit.splits, inputs.data_config)
    record, run_dir = _start_record(inputs, manifests)
    summary = _new_summary(record, inputs.config)
    try:
        _scan_and_summarize(inputs, manifests, run_dir, summary)
    except KeyboardInterrupt as error:
        close_failed_record(record, STATUS_ABORTED, error, run_dir)
        raise
    except Exception as error:
        close_failed_record(record, STATUS_FAILED, error, run_dir)
        raise
    record["status"] = STATUS_COMPLETED
    write_json(record, run_dir / RECORD_FILENAME)
    return AuditResult(record=record, run_dir=run_dir, summary=summary)


def _start_record(
    inputs: AuditInputs, manifests: Mapping[str, pd.DataFrame]
) -> tuple[dict[str, Any], Path]:
    config = inputs.config
    determinism = reproducibility.seed_everything(config.run.seed)
    created_at = datetime.now(UTC)
    run_id = reproducibility.make_run_id(inputs.config_path.stem, created_at)
    header = RecordHeader(
        hypothesis=config.run.hypothesis,
        what_changed=config.run.what_changed,
        notes=config.run.notes,
        seed=config.run.seed,
        config_path=inputs.config_path,
        repo_root=inputs.repo_root,
        data_config=inputs.data_config,
        resolved_config=audit_resolved_config(inputs),
        manifest_splits=config.audit.splits,
    )
    environment = reproducibility.environment_info(torch.device(DEVICE_CPU), determinism)
    record = build_record(header, run_id, created_at, environment)
    record["data_subsets"] = {
        name: asdict(summarize_split(frame)) for name, frame in manifests.items()
    }
    record["training_epochs"] = None
    run_dir = inputs.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(audit_resolved_config(inputs), run_dir / RESOLVED_CONFIG_FILENAME)
    write_json(record, run_dir / RECORD_FILENAME)
    logger.info("JPEG table audit %s started; writing to %s", run_id, run_dir)
    return record, run_dir


def _new_summary(record: Mapping[str, Any], config: AuditConfig) -> dict[str, Any]:
    return {
        "run_id": record["run_id"],
        "inputs": {"pillow": record["environment"]["pillow"]},
        "splits": list(config.audit.splits),
        "rows_scanned": None,
        "headers_read": None,
        "quantization_audit": None,
        "header_value_counts": None,
        "spoof_type_crosstab": None,
        "failures": {"count": 0, "max_listed": config.audit.max_listed_failures, "listed": []},
    }


def _scan_and_summarize(
    inputs: AuditInputs,
    manifests: Mapping[str, pd.DataFrame],
    run_dir: Path,
    summary: dict[str, Any],
) -> None:
    scan = inputs.config.audit
    headers, failures = read_audit_headers(
        manifests, inputs.data_config.dataset_root, scan.progress_every
    )
    summary["rows_scanned"] = sum(int(len(frame)) for frame in manifests.values())
    summary["headers_read"] = int(len(headers))
    summary["failures"] = {
        "count": len(failures),
        "max_listed": scan.max_listed_failures,
        "listed": [asdict(failure) for failure in failures[: scan.max_listed_failures]],
    }
    write_json(summary, run_dir / AUDIT_SUMMARY_FILENAME)
    audit = add_shares(audit_quantization(headers))
    summary["quantization_audit"] = audit
    summary["header_value_counts"] = count_header_values(headers)
    summary["spoof_type_crosstab"] = spoof_type_crosstab(headers, audit)
    write_json(summary, run_dir / AUDIT_SUMMARY_FILENAME)


def format_audit_report(result: AuditResult) -> str:
    """Render the provenance, the table-set audit, the value counts and the cross-tab as text.

    Args:
        result: From :func:`run_audit`.

    Returns:
        Multi-line text. The last line states that this run gets no ledger row.
    """
    record, summary = result.record, result.summary
    lines = [
        f"JPEG table audit {record['run_id']}: status={record['status']}",
        f"  run dir: {result.run_dir}",
        f"  git_sha: {record['git_sha']}  git_dirty: {record['git_dirty']}",
        f"  config: {record['config_path']}  config_hash: {record['config_hash']}",
        f"  split: {record['split_name']}  split_sha256: {record['split_sha256']}",
        f"  manifest_sha256: {record['manifest_sha256']}",
        f"  pillow: {record['environment']['pillow']}  splits: {summary['splits']}",
        f"  rows scanned: {summary['rows_scanned']}  headers read: {summary['headers_read']}",
        f"  manifest counts: {record['data_subsets']}",
        *_audit_lines(summary["quantization_audit"]),
        *_counts_lines(summary["header_value_counts"]),
        *_crosstab_lines(summary["spoof_type_crosstab"]),
        *_failure_lines(summary["failures"]),
    ]
    if record["git_dirty"]:
        lines.append("WARNING: git_dirty is true; this run cannot be cited (docs/RULES.md §3).")
    lines.append(NO_LEDGER_ROW)
    return "\n".join(lines)


def _audit_lines(audit: Mapping[str, Any]) -> list[str]:
    lines = ["Quantization-table audit (headers only, no pixels decoded):"]
    for class_name, entry in audit.items():
        rows = ", ".join(f"{split} {count}" for split, count in entry["rows"].items())
        lines.append(
            f"  {class_name}: {entry['n_distinct_table_sets']} distinct table set(s); rows {rows}"
        )
        for rank, table_set in enumerate(entry["table_sets"], start=1):
            shares = ", ".join(
                f"{split} {count} ({_share_text(table_set['share'][split])})"
                for split, count in table_set["rows"].items()
            )
            lines.append(
                f"    [{rank}] {table_set['match']}: luma mean {table_set['luma_mean']}, "
                f"chroma mean {table_set['chroma_mean']}, {table_set['mode']} "
                f"{table_set['subsampling']}; rows {shares}"
            )
    return lines


def _share_text(share: float | None) -> str:
    return "n/a" if share is None else f"{100 * share:.2f}%"


def _counts_lines(counts: Mapping[str, Any]) -> list[str]:
    lines = ["Header value counts over the rows whose header was read:"]
    for split, per_class in counts.items():
        for class_name, entry in per_class.items():
            lines.append(
                f"  {split} {class_name} ({entry['rows']} rows): format {entry['format']}, "
                f"mode {entry['mode']}, subsampling {entry['subsampling']}"
            )
    return lines


def _crosstab_lines(crosstab: Mapping[str, Any]) -> list[str]:
    codes = crosstab["spoof_type_codes"]
    lines = [f"Spoof table set against spoof_type (codes {codes}):"]
    for table_set in crosstab["table_sets"]:
        head = f"  [{table_set['rank']}] {table_set['match']}"
        for split, per_code in table_set["counts"].items():
            counted = ", ".join(f"{code}={count}" for code, count in per_code.items())
            lines.append(f"{head} {split}: {counted}")
    return lines


def _failure_lines(failures: Mapping[str, Any]) -> list[str]:
    listed = failures["listed"]
    lines = [
        f"Unreadable headers: {failures['count']} "
        f"(listing up to {failures['max_listed']}; they did not stop the run)"
    ]
    lines.extend(f"  {entry['split']} {entry['image_path']}: {entry['error']}" for entry in listed)
    return lines
