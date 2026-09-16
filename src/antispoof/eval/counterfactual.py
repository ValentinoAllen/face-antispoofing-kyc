"""Counterfactual evaluation: does the baseline CNN use the JPEG-encoding trace of attack images?

The metadata probe showed that header features alone separate live from spoof. This evaluation
changes only how attack images are encoded, at evaluation time, and scores an unchanged baseline
checkpoint on the val subset under these arms:

- ``identity``: no change. Must reproduce the source run's counts exactly.
- ``reencode_same``: each spoof image decoded and re-encoded at the spoof class's quality. Controls
  for re-encoding itself.
- ``reencode_live_quality``: each spoof image re-encoded at the live class's quality.
- ``reencode_live_quality_and_size``: each spoof image resized to the size of a live train-subset
  image drawn with a seeded generator, then re-encoded at the live quality.

Live rows are never modified; subsampling is kept. The class qualities come from a header-only audit
of the quantization tables (:mod:`antispoof.eval.jpeg_tables`). The model and transform are rebuilt
from the source run's ``resolved_config.json``, whose hash must match the committed baseline config.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn

from antispoof.data import labels
from antispoof.data.build import summarize_split
from antispoof.data.config import DataConfig
from antispoof.data.dataset import DatasetItem, ImageTransform, ManifestDataset, load_rgb_image
from antispoof.data.transforms import build_baseline_transform
from antispoof.eval.jpeg_tables import (
    CLASS_NAME,
    RESAMPLE_FILTERS,
    ClassEncoding,
    Reencoding,
    audit_quantization,
    class_encodings,
    read_subset_headers,
    reencode_image,
)
from antispoof.eval.metadata_probe import SUBSAMPLING_NAMES
from antispoof.eval.pad_metrics import PadMetrics
from antispoof.models.factory import build_model
from antispoof.training import reproducibility
from antispoof.training.config import (
    EvalConfig,
    TrainConfig,
    TrainConfigError,
    parse_section,
    parse_train_config,
)
from antispoof.training.loop import evaluate, make_loader, resolve_device
from antispoof.training.run import (
    CHECKPOINT_FILENAME,
    RECORD_FILENAME,
    RESOLVED_CONFIG_FILENAME,
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    RecordHeader,
    build_record,
    close_failed_record,
    fill_pooled_metrics,
    load_subsets,
    manifest_paths,
    resolved_config,
    write_json,
)

logger = logging.getLogger(__name__)

SUMMARY_FILENAME = "counterfactual_summary.json"
PREDICTIONS_FILENAME_TEMPLATE = "predictions_{arm}.csv"

ARM_IDENTITY = "identity"
ARM_REENCODE_SAME = "reencode_same"
ARM_REENCODE_LIVE_QUALITY = "reencode_live_quality"
ARM_REENCODE_LIVE_QUALITY_AND_SIZE = "reencode_live_quality_and_size"
KNOWN_ARMS = (
    ARM_IDENTITY,
    ARM_REENCODE_SAME,
    ARM_REENCODE_LIVE_QUALITY,
    ARM_REENCODE_LIVE_QUALITY_AND_SIZE,
)
REQUIRED_ARMS = (ARM_IDENTITY, ARM_REENCODE_SAME, ARM_REENCODE_LIVE_QUALITY)
METRICS_ARM = ARM_REENCODE_LIVE_QUALITY
"""The arm whose pooled metrics fill the run record's ``metrics`` and the ledger row."""

LIVE = CLASS_NAME[labels.LABEL_LIVE]
SPOOF = CLASS_NAME[labels.LABEL_SPOOF]

COUNT_KEYS = ("n_attack", "n_attack_accepted", "n_bona_fide", "n_bona_fide_rejected")
EDIT_COLUMNS = (
    "reencode_quality",
    "reencode_subsampling",
    "resize_width",
    "resize_height",
    "size_source_image_path",
)
"""Columns appended to each arm's predictions; null on rows the arm leaves unchanged."""


class CounterfactualConfigError(ValueError):
    """Raised when the counterfactual config is invalid."""


class CounterfactualInputError(ValueError):
    """Raised when the source run folder or the data do not match the committed baseline."""


class CounterfactualCheckError(RuntimeError):
    """Raised when an arm fails a validity check: identity counts or an unchanged BPCER."""


@dataclass(frozen=True)
class CounterfactualRunConfig:
    """Run description and seed (``run:``)."""

    hypothesis: str
    what_changed: str
    notes: str
    seed: int


@dataclass(frozen=True)
class BaselineSourceConfig:
    """The committed experiment config the evaluated checkpoint must come from (``baseline:``)."""

    config: str


@dataclass(frozen=True)
class ArmsConfig:
    """Arms to evaluate, in order, and the resize filter of the size arm (``counterfactual:``)."""

    arms: tuple[str, ...]
    resize_filter: str


@dataclass(frozen=True)
class CounterfactualConfig:
    """A resolved counterfactual config. Fields are documented in the YAML file."""

    run: CounterfactualRunConfig
    baseline: BaselineSourceConfig
    counterfactual: ArmsConfig
    eval: EvalConfig

    def to_dict(self) -> dict[str, Any]:
        """Return the config as nested plain dicts, for hashing and the resolved-config file."""
        return asdict(self)


_SECTION_TYPES = {
    "run": CounterfactualRunConfig,
    "baseline": BaselineSourceConfig,
    "eval": EvalConfig,
}


@dataclass(frozen=True)
class CounterfactualInputs:
    """Everything a counterfactual run needs. ``data_config`` already carries CLI path overrides.

    Attributes:
        config: The counterfactual config.
        config_path: Path of the counterfactual config, under ``configs/``.
        baseline_config: The committed baseline experiment config.
        baseline_config_path: Path of that config, under ``configs/``.
        data_config: Manifest directory, dataset root and split assignment.
        source_run_dir: The baseline run folder with ``checkpoint.pt``, ``resolved_config.json``
            and ``record.json``.
        output_dir: Parent of the new run directory.
        repo_root: The repository root.
    """

    config: CounterfactualConfig
    config_path: Path
    baseline_config: TrainConfig
    baseline_config_path: Path
    data_config: DataConfig
    source_run_dir: Path
    output_dir: Path
    repo_root: Path


@dataclass(frozen=True)
class SourceRun:
    """The files of the evaluated baseline run folder."""

    run_dir: Path
    record: dict[str, Any]
    resolved: dict[str, Any]
    checkpoint: dict[str, Any]


@dataclass(frozen=True)
class LiveSize:
    """The size of one live train-subset image, drawn for one spoof val row."""

    image_path: str
    width: int
    height: int


@dataclass(frozen=True)
class SpoofEdit:
    """The change applied to one spoof row, and where its target size came from."""

    reencoding: Reencoding
    size_source_image_path: str | None = None


@dataclass(frozen=True)
class CounterfactualResult:
    """The final run record, its directory and the summary."""

    record: dict[str, Any]
    run_dir: Path
    summary: dict[str, Any]


def load_counterfactual_config(path: Path) -> CounterfactualConfig:
    """Load a YAML such as ``configs/counterfactual_jpeg.yaml`` into a validated config.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        CounterfactualConfigError: If sections or keys are missing or unknown, a value has the
            wrong type, or a value is invalid.
    """
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    names = {field.name for field in fields(CounterfactualConfig)}
    if not isinstance(document, dict) or set(document) != names:
        raise CounterfactualConfigError(
            f"{path}: expected exactly the top-level sections {sorted(names)}."
        )
    try:
        sections = {
            name: parse_section(name, document[name], kind) for name, kind in _SECTION_TYPES.items()
        }
    except TrainConfigError as error:
        raise CounterfactualConfigError(f"{path}: {error}") from error
    config = CounterfactualConfig(
        **sections, counterfactual=_parse_arms(document["counterfactual"])
    )
    validate_counterfactual_config(config)
    return config


def _parse_arms(section: object) -> ArmsConfig:
    if not isinstance(section, dict) or set(section) != {"arms", "resize_filter"}:
        raise CounterfactualConfigError(
            "Section 'counterfactual' must hold exactly the keys 'arms' and 'resize_filter'."
        )
    arms, resize_filter = section["arms"], section["resize_filter"]
    if not isinstance(arms, list) or not all(isinstance(arm, str) for arm in arms):
        raise CounterfactualConfigError(f"counterfactual.arms: expected a list of names: {arms!r}.")
    if not isinstance(resize_filter, str):
        raise CounterfactualConfigError(
            f"counterfactual.resize_filter: expected str, got {resize_filter!r}."
        )
    return ArmsConfig(arms=tuple(arms), resize_filter=resize_filter)


def validate_counterfactual_config(config: CounterfactualConfig) -> None:
    """Check values, arm names and arm order.

    Args:
        config: The configuration.

    Raises:
        CounterfactualConfigError: Listing every violated constraint.
    """
    arms = config.counterfactual.arms
    checks = [
        (bool(config.run.hypothesis.strip()), "run.hypothesis must not be empty"),
        (bool(config.run.what_changed.strip()), "run.what_changed must not be empty"),
        (config.run.seed >= 0, "run.seed must be >= 0"),
        (bool(config.baseline.config.strip()), "baseline.config must not be empty"),
        (set(arms) <= set(KNOWN_ARMS), f"counterfactual.arms must be in {KNOWN_ARMS}"),
        (len(set(arms)) == len(arms), "counterfactual.arms must not repeat an arm"),
        (set(REQUIRED_ARMS) <= set(arms), f"counterfactual.arms must include {REQUIRED_ARMS}"),
        (arms[:1] == (ARM_IDENTITY,), "counterfactual.arms must start with identity"),
        (
            config.counterfactual.resize_filter in RESAMPLE_FILTERS,
            f"counterfactual.resize_filter must be one of {RESAMPLE_FILTERS}",
        ),
        (0.0 <= config.eval.threshold <= 1.0, "eval.threshold must be in [0, 1]"),
        (bool(config.eval.threshold_rule.strip()), "eval.threshold_rule must not be empty"),
    ]
    problems = [message for passed, message in checks if not passed]
    if problems:
        raise CounterfactualConfigError(
            "Invalid counterfactual config: " + "; ".join(problems) + "."
        )


def load_source_run(run_dir: Path) -> SourceRun:
    """Read a baseline run folder: ``record.json``, ``resolved_config.json`` and ``checkpoint.pt``.

    Args:
        run_dir: The folder ``scripts/train.py`` wrote.

    Returns:
        The parsed files. The checkpoint is loaded on CPU with ``weights_only=True``.

    Raises:
        CounterfactualInputError: If a file is missing or the checkpoint has no model state dict.
    """
    names = (RECORD_FILENAME, RESOLVED_CONFIG_FILENAME, CHECKPOINT_FILENAME)
    missing = [name for name in names if not (run_dir / name).is_file()]
    if missing:
        raise CounterfactualInputError(
            f"{run_dir} is not a baseline run folder: missing {missing}."
        )
    record = json.loads((run_dir / RECORD_FILENAME).read_text(encoding="utf-8"))
    resolved = json.loads((run_dir / RESOLVED_CONFIG_FILENAME).read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / CHECKPOINT_FILENAME, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise CounterfactualInputError(f"{run_dir / CHECKPOINT_FILENAME} has no model_state_dict.")
    return SourceRun(run_dir=run_dir, record=record, resolved=resolved, checkpoint=checkpoint)


def check_source_config_hash(
    record: Mapping[str, Any],
    resolved: Mapping[str, Any],
    baseline_config: TrainConfig,
    data_config: DataConfig,
) -> str:
    """Require the source run to have used exactly the committed baseline config.

    Three hashes must be equal: the source ``record.json``'s ``config_hash``, the hash of its
    ``resolved_config.json``, and the hash of the committed baseline config resolved with this run's
    data config (so the path flags must match the source run's).

    Args:
        record: The source ``record.json``.
        resolved: The source ``resolved_config.json``.
        baseline_config: The committed baseline experiment config.
        data_config: This run's data config, after CLI path overrides.

    Returns:
        The matching hash.

    Raises:
        CounterfactualInputError: If the hashes differ; the message lists all three.
    """
    expected = reproducibility.config_hash(resolved_config(baseline_config, data_config))
    recorded = record.get("config_hash")
    from_file = reproducibility.config_hash(resolved)
    if not recorded == from_file == expected:
        raise CounterfactualInputError(
            "The --run-dir config_hash must match the committed baseline config: "
            f"record.json {recorded}, resolved_config.json {from_file}, committed config "
            f"resolved with this run's data paths {expected}."
        )
    return expected


def check_source_run(source: SourceRun, inputs: CounterfactualInputs) -> TrainConfig:
    """Check the source run folder against the committed baseline and rebuild its experiment.

    Args:
        source: From :func:`load_source_run`.
        inputs: The counterfactual inputs.

    Returns:
        The experiment config parsed from the source ``resolved_config.json``.

    Raises:
        CounterfactualInputError: If the run did not complete, came from another config, its
            hashes differ (:func:`check_source_config_hash`), the checkpoint belongs to another run,
            or the thresholds differ.
        TrainConfigError: If the resolved experiment is invalid.
    """
    record, checkpoint = source.record, source.checkpoint
    config_path = reproducibility.repo_relative_config_path(
        inputs.baseline_config_path, inputs.repo_root
    )
    if record.get("status") != STATUS_COMPLETED or record.get("config_path") != config_path:
        raise CounterfactualInputError(
            f"The --run-dir record must be a completed run of {config_path}: status "
            f"{record.get('status')!r}, config_path {record.get('config_path')!r}."
        )
    check_source_config_hash(record, source.resolved, inputs.baseline_config, inputs.data_config)
    identity = (checkpoint.get("run_id"), checkpoint.get("config_hash"))
    if identity != (record.get("run_id"), record.get("config_hash")):
        raise CounterfactualInputError(
            f"checkpoint.pt belongs to run {identity[0]} with config_hash {identity[1]}, not to "
            f"the record's run {record.get('run_id')}."
        )
    experiment = parse_train_config(source.resolved.get("experiment"), RESOLVED_CONFIG_FILENAME)
    if inputs.config.eval.threshold != experiment.eval.threshold:
        raise CounterfactualInputError(
            f"eval.threshold {inputs.config.eval.threshold} differs from the source run's "
            f"{experiment.eval.threshold}."
        )
    return experiment


def check_same_data(
    record: Mapping[str, Any], subsets: Mapping[str, pd.DataFrame], data_config: DataConfig
) -> None:
    """Require the manifests and subsets to be the ones the source run used.

    Args:
        record: The source ``record.json``.
        subsets: The subsets drawn with the source run's sizes and seed.
        data_config: This run's data config.

    Raises:
        CounterfactualInputError: If a manifest's SHA-256 or a subset's counts differ.
    """
    manifests = {
        path.name: reproducibility.sha256_file(path)
        for path in manifest_paths(data_config).values()
    }
    if record.get("manifest_sha256") != manifests:
        raise CounterfactualInputError(
            f"Manifest SHA-256s {manifests} differ from the source run's "
            f"{record.get('manifest_sha256')}."
        )
    counts = {name: asdict(summarize_split(frame)) for name, frame in subsets.items()}
    if record.get("data_subsets") != counts:
        raise CounterfactualInputError(
            f"Subsets {counts} differ from the source run's {record.get('data_subsets')}."
        )


class CounterfactualDataset(ManifestDataset):
    """A :class:`ManifestDataset` whose spoof rows may be re-encoded in memory before the transform.

    Rows without an edit are loaded exactly as :class:`ManifestDataset` loads them. Files on disk
    are never written.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        dataset_root: Path,
        transform: ImageTransform,
        edits: Mapping[int, SpoofEdit],
    ) -> None:
        """Store the rows and the per-row edits.

        Args:
            manifest: Frame with at least ``image_path`` and ``label`` columns.
            dataset_root: Directory the ``image_path`` values are relative to.
            transform: Applied to every (possibly re-encoded) RGB image.
            edits: Positional row index to edit. Only spoof rows may be edited.

        Raises:
            ValueError: If a column or label is invalid, or an edit targets a live or missing row.
        """
        super().__init__(manifest, dataset_root, transform)
        spoof = float(labels.LABEL_SPOOF)
        invalid = sorted(
            index for index in edits if not 0 <= index < len(self) or self._labels[index] != spoof
        )
        if invalid:
            raise ValueError(f"Edits may target spoof rows only; rows {invalid} are not spoof.")
        self._edits = dict(edits)

    def load_image(self, index: int) -> Image.Image:
        """Decode one image and apply its edit, if any.

        Args:
            index: Positional row index in the manifest.

        Returns:
            The RGB image the transform receives.

        Raises:
            ImageLoadError: If the image is missing or cannot be decoded.
            ReencodeError: If a re-encoded luma table does not match its target quality.
        """
        image = load_rgb_image(self._dataset_root / self._image_paths[index])
        edit = self._edits.get(index)
        return image if edit is None else reencode_image(image, edit.reencoding)

    def __getitem__(self, index: int) -> DatasetItem:
        """Load, edit and transform one image.

        Args:
            index: Positional row index in the manifest.

        Returns:
            ``(image, label, index)``.
        """
        return self._transform(self.load_image(index)), self._labels[index], index


def draw_live_sizes(headers: pd.DataFrame, n: int, seed: int) -> list[LiveSize]:
    """Draw, with replacement, the sizes of ``n`` live train-subset images.

    Args:
        headers: From :func:`antispoof.eval.jpeg_tables.read_subset_headers`.
        n: Number of draws, one per spoof val row in row order.
        seed: Seeds ``numpy.random.default_rng``.

    Returns:
        The drawn image paths and sizes.

    Raises:
        CounterfactualInputError: If the train subset has no live rows.
    """
    live_train = headers[
        (headers["split"] == labels.SPLIT_TRAIN) & (headers["label"] == labels.LABEL_LIVE)
    ]
    if live_train.empty:
        raise CounterfactualInputError("The train subset has no live rows to draw sizes from.")
    picks = live_train.iloc[np.random.default_rng(seed).integers(len(live_train), size=n)]
    return [
        LiveSize(str(path), int(width), int(height))
        for path, width, height in zip(
            picks["image_path"], picks["width"], picks["height"], strict=True
        )
    ]


def arm_edits(
    arm: str,
    val: pd.DataFrame,
    encodings: Mapping[str, ClassEncoding],
    live_sizes: Sequence[LiveSize],
    resize_filter: str,
) -> dict[int, SpoofEdit]:
    """Build the spoof-row edits of one arm. Subsampling is always the spoof class's.

    Args:
        arm: One of ``KNOWN_ARMS``.
        val: The val subset.
        encodings: From :func:`antispoof.eval.jpeg_tables.class_encodings`.
        live_sizes: From :func:`draw_live_sizes`, one per spoof val row; used by the size arm only.
        resize_filter: Resize filter name for the size arm.

    Returns:
        Positional row index to edit; empty for ``identity``.

    Raises:
        ValueError: If the arm is unknown or ``live_sizes`` does not match the spoof row count.
    """
    spoof_rows = [int(row) for row in np.flatnonzero(val["label"] == labels.LABEL_SPOOF)]
    live, spoof = encodings[LIVE], encodings[SPOOF]
    if arm == ARM_IDENTITY:
        return {}
    if arm in (ARM_REENCODE_SAME, ARM_REENCODE_LIVE_QUALITY):
        quality = spoof.quality if arm == ARM_REENCODE_SAME else live.quality
        return {row: SpoofEdit(Reencoding(quality, spoof.subsampling)) for row in spoof_rows}
    if arm != ARM_REENCODE_LIVE_QUALITY_AND_SIZE:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {KNOWN_ARMS}.")
    if len(live_sizes) != len(spoof_rows):
        raise ValueError(f"{len(live_sizes)} drawn sizes for {len(spoof_rows)} spoof rows.")
    return {
        row: SpoofEdit(
            Reencoding(live.quality, spoof.subsampling, (size.width, size.height), resize_filter),
            size_source_image_path=size.image_path,
        )
        for row, size in zip(spoof_rows, live_sizes, strict=True)
    }


def edit_columns(n_rows: int, edits: Mapping[int, SpoofEdit]) -> pd.DataFrame:
    """Describe each row's edit for the predictions file.

    Args:
        n_rows: Number of evaluated rows.
        edits: From :func:`arm_edits`.

    Returns:
        ``EDIT_COLUMNS``, one row per evaluated row, null where a row is unchanged.
    """
    rows = [edits.get(index) for index in range(n_rows)]
    sizes = [None if edit is None else edit.reencoding.size for edit in rows]
    return pd.DataFrame(
        {
            "reencode_quality": pd.array(
                [None if edit is None else edit.reencoding.quality for edit in rows], dtype="Int64"
            ),
            "reencode_subsampling": [
                None if edit is None else SUBSAMPLING_NAMES[edit.reencoding.subsampling]
                for edit in rows
            ],
            "resize_width": pd.array([None if s is None else s[0] for s in sizes], dtype="Int64"),
            "resize_height": pd.array([None if s is None else s[1] for s in sizes], dtype="Int64"),
            "size_source_image_path": [
                None if edit is None else edit.size_source_image_path for edit in rows
            ],
        }
    )


def check_identity(metrics: PadMetrics, source_metrics: Mapping[str, Any]) -> None:
    """Require the identity arm to reproduce the source run's counts exactly.

    Args:
        metrics: Pooled metrics of the identity arm.
        source_metrics: ``metrics`` of the source ``record.json``.

    Raises:
        CounterfactualCheckError: If any of ``COUNT_KEYS`` differs.
    """
    found = {key: getattr(metrics, key) for key in COUNT_KEYS}
    expected = {key: source_metrics.get(key) for key in COUNT_KEYS}
    if found != expected:
        raise CounterfactualCheckError(
            f"The identity arm must reproduce the source run's counts exactly: expected "
            f"{expected}, got {found}."
        )


def check_bpcer_unchanged(arm: str, metrics: PadMetrics, identity: PadMetrics) -> None:
    """Require an arm's BPCER to equal the identity arm's (live rows are never modified).

    Args:
        arm: The arm name, for the message.
        metrics: Pooled metrics of the arm.
        identity: Pooled metrics of the identity arm.

    Raises:
        CounterfactualCheckError: If the bona fide counts or BPCER differ.
    """
    found = (metrics.n_bona_fide_rejected, metrics.n_bona_fide, metrics.bpcer)
    expected = (identity.n_bona_fide_rejected, identity.n_bona_fide, identity.bpcer)
    if found != expected:
        raise CounterfactualCheckError(
            f"Arm {arm!r} changed BPCER: {found[0]}/{found[1]} bona fide rejected, against "
            f"{expected[0]}/{expected[1]} in the identity arm. Live rows must be unchanged."
        )


def load_source_model(
    experiment: TrainConfig, checkpoint: Mapping[str, Any], device: torch.device
) -> tuple[nn.Module, ImageTransform]:
    """Rebuild the source run's model and transform and load its weights.

    Args:
        experiment: From the source ``resolved_config.json``.
        checkpoint: The loaded ``checkpoint.pt``.
        device: The evaluation device.

    Returns:
        The model on ``device`` and the baseline transform.
    """
    model = build_model(experiment.model)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    return model, build_baseline_transform(experiment.model.input_size, model)


def describe_arm(arm: str, encodings: Mapping[str, ClassEncoding], resize_filter: str) -> str:
    """Describe in words what an arm does to spoof images.

    Args:
        arm: One of ``KNOWN_ARMS``.
        encodings: The class encodings.
        resize_filter: The size arm's filter.

    Returns:
        E.g. ``re-encoded at q75 (live quality), 4:2:0``.
    """
    live, spoof = encodings[LIVE], encodings[SPOOF]
    subsampling = SUBSAMPLING_NAMES[spoof.subsampling]
    descriptions = {
        ARM_IDENTITY: "none",
        ARM_REENCODE_SAME: f"re-encoded at q{spoof.quality} (spoof quality), {subsampling}",
        ARM_REENCODE_LIVE_QUALITY: f"re-encoded at q{live.quality} (live quality), {subsampling}",
        ARM_REENCODE_LIVE_QUALITY_AND_SIZE: (
            f"resized ({resize_filter}) to a drawn live train-subset size, re-encoded at "
            f"q{live.quality} (live quality), {subsampling}"
        ),
    }
    return descriptions[arm]


def counterfactual_resolved_config(inputs: CounterfactualInputs) -> dict[str, Any]:
    """Return the fully resolved config that is hashed and saved with the run.

    Args:
        inputs: The counterfactual inputs.

    Returns:
        JSON-compatible ``{"experiment", "baseline", "data"}``. ``baseline`` holds the committed
        baseline config's path and resolved hash.

    Raises:
        ProvenanceError: If the baseline config is not under ``configs/`` in the repository.
    """
    baseline_resolved = resolved_config(inputs.baseline_config, inputs.data_config)
    resolved: dict[str, Any] = reproducibility.to_json_compatible(
        {
            "experiment": inputs.config.to_dict(),
            "baseline": {
                "config_path": reproducibility.repo_relative_config_path(
                    inputs.baseline_config_path, inputs.repo_root
                ),
                "config_hash": reproducibility.config_hash(baseline_resolved),
            },
            "data": asdict(inputs.data_config),
        }
    )
    return resolved


def run_counterfactual(inputs: CounterfactualInputs) -> CounterfactualResult:
    """Audit the quantization tables, evaluate every arm, and write the run directory.

    Writes ``record.json`` and ``resolved_config.json`` (``docs/SCHEMA.md`` §3),
    ``counterfactual_summary.json`` and one ``predictions_<arm>.csv`` per arm to
    ``<output_dir>/<run_id>/``. The record's metrics are the ``reencode_live_quality`` arm's. The
    test split is never read.

    Args:
        inputs: Configs, paths and the source run folder.

    Returns:
        The completed record, the run directory and the summary.

    Raises:
        CounterfactualInputError, TrainConfigError, SplitLeakageError, SplitCoverageError,
            ProvenanceError: Before the run directory is created.
        Exception: Anything raised during the audit or the arms, e.g. ``QualityAuditError`` or
            ``CounterfactualCheckError``. The record is first rewritten with status ``failed``
            (``aborted`` on ``KeyboardInterrupt``).
    """
    source = load_source_run(inputs.source_run_dir)
    experiment = check_source_run(source, inputs)
    subsets = load_subsets(experiment, inputs.data_config)
    check_same_data(source.record, subsets, inputs.data_config)
    device = resolve_device(experiment.run.device)
    record, run_dir = _start_record(inputs, source, subsets, device)
    summary = _new_summary(record, inputs.config.eval, source.record)
    try:
        results = _audit_and_evaluate(inputs, source, experiment, subsets, device, run_dir, summary)
    except KeyboardInterrupt as error:
        close_failed_record(record, STATUS_ABORTED, error, run_dir)
        raise
    except Exception as error:
        close_failed_record(record, STATUS_FAILED, error, run_dir)
        raise
    record["status"] = STATUS_COMPLETED
    fill_pooled_metrics(record, results[METRICS_ARM], inputs.config.eval.threshold_rule)
    write_json(record, run_dir / RECORD_FILENAME)
    return CounterfactualResult(record=record, run_dir=run_dir, summary=summary)


def _start_record(
    inputs: CounterfactualInputs,
    source: SourceRun,
    subsets: Mapping[str, pd.DataFrame],
    device: torch.device,
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
        resolved_config=counterfactual_resolved_config(inputs),
    )
    environment = reproducibility.environment_info(device, determinism)
    record = build_record(header, run_id, created_at, environment)
    record["data_subsets"] = {
        name: asdict(summarize_split(frame)) for name, frame in subsets.items()
    }
    record["training_epochs"] = None
    record["metrics_arm"] = METRICS_ARM
    record["source_run"] = _source_run_entry(source)
    run_dir = inputs.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(counterfactual_resolved_config(inputs), run_dir / RESOLVED_CONFIG_FILENAME)
    write_json(record, run_dir / RECORD_FILENAME)
    logger.info("Counterfactual %s started on %s; writing to %s", run_id, device, run_dir)
    return record, run_dir


def _source_run_entry(source: SourceRun) -> dict[str, Any]:
    checkpoint_path = source.run_dir / CHECKPOINT_FILENAME
    return {
        "run_id": source.record["run_id"],
        "config_path": source.record["config_path"],
        "config_hash": source.record["config_hash"],
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": reproducibility.sha256_file(checkpoint_path),
    }


def _new_summary(
    record: Mapping[str, Any], eval_config: EvalConfig, source_record: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "run_id": record["run_id"],
        "source_run": record["source_run"],
        "inputs": {name: record["environment"][name] for name in ("pillow", "torch", "timm")},
        "quantization_audit": None,
        "class_encodings": None,
        "threshold": eval_config.threshold,
        "threshold_rule": eval_config.threshold_rule,
        "metrics_arm": METRICS_ARM,
        "identity_expected_counts": {key: source_record["metrics"].get(key) for key in COUNT_KEYS},
        "arms": {},
    }


def _audit_and_evaluate(
    inputs: CounterfactualInputs,
    source: SourceRun,
    experiment: TrainConfig,
    subsets: Mapping[str, pd.DataFrame],
    device: torch.device,
    run_dir: Path,
    summary: dict[str, Any],
) -> dict[str, PadMetrics]:
    headers = read_subset_headers(subsets, inputs.data_config.dataset_root)
    summary["quantization_audit"] = audit_quantization(headers)
    write_json(summary, run_dir / SUMMARY_FILENAME)
    encodings = class_encodings(summary["quantization_audit"])
    summary["class_encodings"] = {
        name: {"quality": encoding.quality, "subsampling": SUBSAMPLING_NAMES[encoding.subsampling]}
        for name, encoding in encodings.items()
    }
    val = subsets[labels.SPLIT_VAL]
    n_spoof = int((val["label"] == labels.LABEL_SPOOF).sum())
    live_sizes = draw_live_sizes(headers, n_spoof, inputs.config.run.seed)
    model, transform = load_source_model(experiment, source.checkpoint, device)
    results: dict[str, PadMetrics] = {}
    for arm in inputs.config.counterfactual.arms:
        edits = arm_edits(
            arm, val, encodings, live_sizes, inputs.config.counterfactual.resize_filter
        )
        dataset = CounterfactualDataset(val, inputs.data_config.dataset_root, transform, edits)
        loader = make_loader(
            dataset, experiment.data, shuffle=False, seed=experiment.run.seed, device=device
        )
        metrics, predictions = evaluate(model, loader, val, device, inputs.config.eval.threshold)
        if arm == ARM_IDENTITY:
            check_identity(metrics, source.record["metrics"])
        elif ARM_IDENTITY not in results:
            raise CounterfactualConfigError("counterfactual.arms must start with identity.")
        else:
            check_bpcer_unchanged(arm, metrics, results[ARM_IDENTITY])
        results[arm] = metrics
        predictions = pd.concat([predictions, edit_columns(len(val), edits)], axis=1)
        entry = {
            "spoof_edit": describe_arm(arm, encodings, inputs.config.counterfactual.resize_filter)
        }
        _write_arm(
            arm, entry | {"n_spoof_edited": len(edits)}, metrics, predictions, run_dir, summary
        )
    return results


def _write_arm(
    arm: str,
    entry: dict[str, Any],
    metrics: PadMetrics,
    predictions: pd.DataFrame,
    run_dir: Path,
    summary: dict[str, Any],
) -> None:
    filename = PREDICTIONS_FILENAME_TEMPLATE.format(arm=arm)
    predictions.to_csv(run_dir / filename, index=False)
    summary["arms"][arm] = {
        **entry,
        "pad_metrics": metrics.to_dict(),
        "predictions_file": filename,
    }
    write_json(summary, run_dir / SUMMARY_FILENAME)
    logger.info("Arm %s: %s", arm, metrics)


def format_counterfactual_report(result: CounterfactualResult) -> str:
    """Render the provenance, the quantization-table audit and a table of all arms as plain text.

    Args:
        result: From :func:`run_counterfactual`.

    Returns:
        Multi-line text.
    """
    record, summary = result.record, result.summary
    source = record["source_run"]
    lines = [
        f"Counterfactual {record['run_id']}: status={record['status']}",
        f"  run dir: {result.run_dir}",
        f"  git_sha: {record['git_sha']}  git_dirty: {record['git_dirty']}",
        f"  config: {record['config_path']}  config_hash: {record['config_hash']}",
        f"  split: {record['split_name']}  split_sha256: {record['split_sha256']}",
        f"  manifest_sha256: {record['manifest_sha256']}",
        f"  source run: {source['run_id']}  config: {source['config_path']}  "
        f"config_hash: {source['config_hash']}",
        f"  checkpoint: {source['checkpoint']}  sha256: {source['checkpoint_sha256']}",
        f"  device: {record['environment']['device']}  subsets: {record['data_subsets']}",
        *_audit_lines(summary["quantization_audit"], summary["class_encodings"]),
        *_arm_lines(summary),
    ]
    if record["git_dirty"]:
        lines.append("WARNING: git_dirty is true; this run cannot be cited (docs/RULES.md §3).")
    return "\n".join(lines)


def _audit_lines(audit: Mapping[str, Any], encodings: Mapping[str, Any] | None) -> list[str]:
    lines = ["Quantization-table audit (headers only, no pixels decoded):"]
    for class_name, entry in audit.items():
        rows = ", ".join(f"{split} {count}" for split, count in entry["rows"].items())
        lines.append(
            f"  {class_name}: {entry['n_distinct_table_sets']} distinct table set(s); rows {rows}"
        )
        for rank, table_set in enumerate(entry["table_sets"], start=1):
            set_rows = ", ".join(f"{split} {count}" for split, count in table_set["rows"].items())
            lines.append(
                f"    [{rank}] {table_set['match']}: luma mean {table_set['luma_mean']}, "
                f"chroma mean {table_set['chroma_mean']}, {table_set['mode']} "
                f"{table_set['subsampling']}; rows {set_rows}"
            )
    if encodings is not None:
        text = ", ".join(
            f"{name} q{value['quality']} {value['subsampling']}"
            for name, value in encodings.items()
        )
        lines.append(f"  class encodings: {text}")
    return lines


def _arm_lines(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        f"Arms on the val subset at threshold {summary['threshold']} "
        f"({summary['threshold_rule']}); live rows unchanged; record metrics = "
        f"{summary['metrics_arm']}:",
        f"  {'arm':<32} {'APCER (pooled)':<24} {'BPCER':<22} {'ACER (pooled)':<14} spoof edit",
    ]
    for arm, entry in summary["arms"].items():
        metrics = entry["pad_metrics"]
        apcer = f"{metrics['apcer']:.6f} = {metrics['n_attack_accepted']}/{metrics['n_attack']}"
        bpcer = (
            f"{metrics['bpcer']:.6f} = {metrics['n_bona_fide_rejected']}/{metrics['n_bona_fide']}"
        )
        lines.append(
            f"  {arm:<32} {apcer:<24} {bpcer:<22} {metrics['acer']:<14.6f} {entry['spoof_edit']}"
        )
    return lines
