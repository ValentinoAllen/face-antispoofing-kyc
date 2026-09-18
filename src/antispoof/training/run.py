"""Run one baseline experiment: subsets, training, validation and the run record.

This is the logic behind ``scripts/train.py``. Order of operations:

1. Read ``manifest_train.csv`` and ``manifest_val.csv``, draw seeded label-stratified subsets, and
   check them against the committed split assignment with ``validate_splits``.
2. Seed everything, then write the run record (``docs/SCHEMA.md`` §3) with ``status: running`` to
   ``<output_dir>/<run_id>/``.
3. Build the model and loaders, train, and evaluate on the val subset. The test manifest is never
   read.
4. Write the checkpoint and ``predictions.csv``, then the final record: ``completed``, or
   ``failed``/``aborted`` if anything raised.

The record helpers (:func:`build_record`, :func:`fill_pooled_metrics`, :func:`close_failed_record`
and :func:`write_json`) are public so that other entry points writing a run record can reuse them.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from antispoof.data import labels
from antispoof.data.build import MANIFEST_FILENAME, summarize_split
from antispoof.data.config import DataConfig
from antispoof.data.dataset import ManifestDataset, make_subset
from antispoof.data.manifest import read_manifest
from antispoof.data.splits import (
    SPLIT_ASSIGNMENT_COLUMNS,
    SplitCoverageError,
    load_split_assignment,
    validate_splits,
)
from antispoof.data.transforms import build_baseline_transform
from antispoof.eval.pad_metrics import PadMetrics
from antispoof.models.factory import build_model
from antispoof.training import reproducibility
from antispoof.training.config import TrainConfig, validate_train_config
from antispoof.training.loop import (
    EpochStats,
    evaluate,
    make_loader,
    resolve_device,
    train_epochs,
)

logger = logging.getLogger(__name__)

RECORD_FILENAME = "record.json"
CHECKPOINT_FILENAME = "checkpoint.pt"
PREDICTIONS_FILENAME = "predictions.csv"
RESOLVED_CONFIG_FILENAME = "resolved_config.json"

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_ABORTED = "aborted"

TRAIN_AND_VAL = (labels.SPLIT_TRAIN, labels.SPLIT_VAL)
"""The splits a run that trains or evaluates a model reads. The test split is not among them."""

EMPTY_CELL = "—"
POOLED_SUFFIX = " (pooled)"
DIRTY_NOTE = "git_dirty: not citable; "


@dataclass(frozen=True)
class RunInputs:
    """Everything a run needs. ``data_config`` already carries CLI path overrides."""

    train_config: TrainConfig
    config_path: Path
    data_config: DataConfig
    output_dir: Path
    repo_root: Path


@dataclass(frozen=True)
class RunResult:
    """The final run record, its directory, and per-epoch training stats."""

    record: dict[str, Any]
    run_dir: Path
    epochs: tuple[EpochStats, ...]


@dataclass(frozen=True)
class RecordHeader:
    """The description and provenance inputs every run record starts from.

    Attributes:
        hypothesis: One sentence, copied to the ledger.
        what_changed: Copied to the ledger's "what changed" column.
        notes: Free text.
        seed: The seed the run used.
        config_path: The experiment config file, under ``configs/`` in the repository.
        repo_root: The repository root, for the git state and the repo-relative config path.
        data_config: Split assignment path and manifest directory, after CLI overrides.
        resolved_config: The fully resolved, JSON-compatible config that is hashed.
        manifest_splits: Splits whose manifests the run reads and hashes into ``manifest_sha256``.
            Defaults to train and val.
    """

    hypothesis: str
    what_changed: str
    notes: str
    seed: int
    config_path: Path
    repo_root: Path
    data_config: DataConfig
    resolved_config: Mapping[str, Any]
    manifest_splits: tuple[str, ...] = TRAIN_AND_VAL


def manifest_paths(
    data_config: DataConfig, splits: Sequence[str] = TRAIN_AND_VAL
) -> dict[str, Path]:
    """Return the manifest file of each split a run reads, in reading order.

    Args:
        data_config: Holds the manifest directory.
        splits: Splits to include, in reading order. The default is train and val; only a run that
            reads the test split without evaluating a model passes all three.

    Returns:
        ``{split: <manifest_dir>/manifest_<split>.csv}``.
    """
    return {
        split: data_config.manifest_dir / MANIFEST_FILENAME.format(split=split) for split in splits
    }


def load_subsets(train_config: TrainConfig, data_config: DataConfig) -> dict[str, pd.DataFrame]:
    """Read the train and val manifests, draw the configured subsets, and validate them.

    Args:
        train_config: Subset sizes and seed.
        data_config: Manifest directory, split assignment path and excluded subjects.

    Returns:
        ``{"train": subset, "val": subset}``.

    Raises:
        ValueError: If a manifest row carries a split other than its file's.
        SplitLeakageError: If the subsets share a subject or disagree with the split assignment.
        SplitCoverageError: If a subset subject is not in the split assignment.
    """
    sizes = {
        labels.SPLIT_TRAIN: train_config.data.train_subset,
        labels.SPLIT_VAL: train_config.data.val_subset,
    }
    subsets: dict[str, pd.DataFrame] = {}
    for split, path in manifest_paths(data_config).items():
        manifest = read_manifest(path)
        wrong_split = manifest["split"] != split
        if wrong_split.any():
            raise ValueError(f"{path}: {int(wrong_split.sum())} rows are not in split {split!r}.")
        subsets[split] = make_subset(manifest, sizes[split], train_config.run.seed)
        summary = summarize_split(subsets[split])
        logger.info(
            "%s subset: %d of %d rows, %d subjects (live %d, spoof %d).",
            split,
            summary.rows,
            len(manifest),
            summary.subjects,
            summary.live,
            summary.spoof,
        )
    validate_against_assignment(subsets, data_config)
    return subsets


def validate_against_assignment(
    subsets: Mapping[str, pd.DataFrame], data_config: DataConfig
) -> None:
    """Check the subsets' subjects against the committed split assignment (``docs/SCHEMA.md`` §2).

    The assignment is loaded with ``load_split_assignment``. Its rows are stacked with the subsets'
    ``(subject_id, split)`` rows, so ``validate_splits`` catches both leakage between the subsets
    and any subject whose manifest split disagrees with the assignment.

    Args:
        subsets: Frames with ``subject_id`` and ``split`` columns.
        data_config: Split assignment path and excluded subjects.

    Raises:
        SplitCoverageError: If a subset subject is not in the assignment.
        SplitLeakageError: If the stacked frame violates subject disjointness.
    """
    assignment = load_split_assignment(
        data_config.split_assignment_path, data_config.excluded_subjects
    )
    frames = [frame[list(SPLIT_ASSIGNMENT_COLUMNS)] for frame in subsets.values()]
    unassigned = set(pd.concat(frames)["subject_id"]) - set(assignment["subject_id"])
    if unassigned:
        raise SplitCoverageError(
            f"Subjects missing from {data_config.split_assignment_path}: {sorted(unassigned)}."
        )
    stacked = pd.concat([assignment, *frames], ignore_index=True)
    validate_splits(stacked, data_config.excluded_subjects)


def resolved_config(train_config: TrainConfig, data_config: DataConfig) -> dict[str, Any]:
    """Return the fully resolved config that is hashed and saved with the run.

    Args:
        train_config: The experiment config, including any debug overrides.
        data_config: The data config after CLI path overrides.

    Returns:
        JSON-compatible ``{"experiment": ..., "data": ...}``.
    """
    resolved: dict[str, Any] = reproducibility.to_json_compatible(
        {"experiment": train_config.to_dict(), "data": asdict(data_config)}
    )
    return resolved


_METRIC_KEYS = (
    "apcer_max",
    "apcer_per_species",
    "bpcer",
    "acer",
    "bpcer_at_apcer_1pct",
    "n_bona_fide",
    "n_attack",
    "apcer_pooled",
    "acer_pooled",
    "n_attack_accepted",
    "n_bona_fide_rejected",
)


def build_record(
    header: RecordHeader, run_id: str, created_at: datetime, environment: Mapping[str, str | None]
) -> dict[str, Any]:
    """Build a run record (``docs/SCHEMA.md`` §3) with ``status: running`` and no results yet.

    ``split_sha256`` hashes the split assignment file, which is the split's identity
    (``docs/SCHEMA.md`` §2). ``manifest_sha256`` hashes each manifest the run reads
    (:func:`manifest_paths` over ``header.manifest_splits``), catching a changed manifest even when
    ``git_sha`` is unchanged.

    Args:
        header: The run description and provenance inputs.
        run_id: From :func:`reproducibility.make_run_id`.
        created_at: Launch time in UTC.
        environment: From :func:`reproducibility.environment_info`.

    Returns:
        The record.

    Raises:
        ProvenanceError: If the git state or the repo-relative config path cannot be determined.
    """
    git = reproducibility.git_state(header.repo_root)
    split_path = header.data_config.split_assignment_path
    return {
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "hypothesis": header.hypothesis,
        "what_changed": header.what_changed,
        "config_path": reproducibility.repo_relative_config_path(
            header.config_path, header.repo_root
        ),
        "config_hash": reproducibility.config_hash(header.resolved_config),
        "git_sha": git.sha,
        "git_dirty": git.dirty,
        "seed": header.seed,
        "split_name": split_path.name,
        "split_sha256": reproducibility.sha256_file(split_path),
        "manifest_sha256": {
            path.name: reproducibility.sha256_file(path)
            for path in manifest_paths(header.data_config, header.manifest_splits).values()
        },
        "environment": dict(environment),
        "status": STATUS_RUNNING,
        "eval_split": None,
        "threshold": None,
        "threshold_rule": None,
        "metrics": dict.fromkeys(_METRIC_KEYS),
        "artifacts": dict.fromkeys(("checkpoint", "onnx", "report_dir", "wandb_url")),
        "notes": header.notes,
    }


def new_record(
    inputs: RunInputs, run_id: str, created_at: datetime, environment: Mapping[str, str | None]
) -> dict[str, Any]:
    """Build the training run record with ``status: running`` (see :func:`build_record`).

    Args:
        inputs: The run inputs.
        run_id: From :func:`reproducibility.make_run_id`.
        created_at: Launch time in UTC.
        environment: From :func:`reproducibility.environment_info`.

    Returns:
        The record.

    Raises:
        ProvenanceError: If the git state or the repo-relative config path cannot be determined.
    """
    config = inputs.train_config
    header = RecordHeader(
        hypothesis=config.run.hypothesis,
        what_changed=config.run.what_changed,
        notes=config.run.notes,
        seed=config.run.seed,
        config_path=inputs.config_path,
        repo_root=inputs.repo_root,
        data_config=inputs.data_config,
        resolved_config=resolved_config(config, inputs.data_config),
    )
    return build_record(header, run_id, created_at, environment)


def fill_pooled_metrics(record: dict[str, Any], metrics: PadMetrics, threshold_rule: str) -> None:
    """Write pooled PAD metrics computed on the val subset into ``record``, in place.

    Sets ``eval_split`` to ``val``, ``threshold`` and ``threshold_rule``, and fills ``bpcer``, the
    counts, ``apcer_pooled`` and ``acer_pooled``. ``apcer_max``, ``acer`` and ``apcer_per_species``
    stay null: only pooled rates are computed.

    Args:
        record: From :func:`build_record`.
        metrics: Pooled metrics on the val subset.
        threshold_rule: How the threshold was chosen, from the config.
    """
    record["eval_split"] = labels.SPLIT_VAL
    record["threshold"] = metrics.threshold
    record["threshold_rule"] = threshold_rule
    record["metrics"].update(
        bpcer=metrics.bpcer,
        n_bona_fide=metrics.n_bona_fide,
        n_attack=metrics.n_attack,
        apcer_pooled=metrics.apcer,
        acer_pooled=metrics.acer,
        n_attack_accepted=metrics.n_attack_accepted,
        n_bona_fide_rejected=metrics.n_bona_fide_rejected,
    )


def complete_record(
    record: dict[str, Any],
    config: TrainConfig,
    metrics: PadMetrics,
    epochs: tuple[EpochStats, ...],
    checkpoint_path: Path,
) -> None:
    """Fill the evaluation results into ``record`` and mark it completed.

    Args:
        record: From :func:`new_record`; updated in place.
        config: The experiment config.
        metrics: Pooled metrics on the val subset.
        epochs: Per-epoch training stats.
        checkpoint_path: Where the checkpoint was written.
    """
    record["status"] = STATUS_COMPLETED
    fill_pooled_metrics(record, metrics, config.eval.threshold_rule)
    record["artifacts"]["checkpoint"] = checkpoint_path.as_posix()
    record["training_epochs"] = [asdict(stats) for stats in epochs]


def run_baseline(inputs: RunInputs) -> RunResult:
    """Train on the train subset, evaluate on the val subset, and write the run directory.

    Args:
        inputs: Configs, paths and the repository root.

    Returns:
        The completed record, the run directory and per-epoch stats.

    Raises:
        TrainConfigError, SplitLeakageError, SplitCoverageError, ProvenanceError: Before the run
            directory is created.
        Exception: Anything raised while training or evaluating (e.g. ``ImageLoadError``). The
            record is first rewritten with status ``failed`` (``aborted`` on
            ``KeyboardInterrupt``).
    """
    config = inputs.train_config
    validate_train_config(config)
    subsets = load_subsets(config, inputs.data_config)
    device = resolve_device(config.run.device)
    determinism = reproducibility.seed_everything(config.run.seed)
    created_at = datetime.now(UTC)
    run_id = reproducibility.make_run_id(inputs.config_path.stem, created_at)
    environment = reproducibility.environment_info(device, determinism)
    record = new_record(inputs, run_id, created_at, environment)
    record["data_subsets"] = {
        name: asdict(summarize_split(frame)) for name, frame in subsets.items()
    }
    run_dir = inputs.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(resolved_config(config, inputs.data_config), run_dir / RESOLVED_CONFIG_FILENAME)
    write_json(record, run_dir / RECORD_FILENAME)
    logger.info("Run %s started on %s; writing to %s", run_id, device, run_dir)
    try:
        epochs = _train_and_evaluate(inputs, subsets, device, run_dir, record)
    except KeyboardInterrupt as error:
        close_failed_record(record, STATUS_ABORTED, error, run_dir)
        raise
    except Exception as error:
        close_failed_record(record, STATUS_FAILED, error, run_dir)
        raise
    write_json(record, run_dir / RECORD_FILENAME)
    return RunResult(record=record, run_dir=run_dir, epochs=epochs)


def format_run_summary(result: RunResult) -> str:
    """Render the provenance, training stats and val metrics of a completed run as plain text.

    Args:
        result: From :func:`run_baseline`.

    Returns:
        Multi-line text.
    """
    record, metrics = result.record, result.record["metrics"]
    lines = [
        f"Run {record['run_id']}: status={record['status']}",
        f"  run dir: {result.run_dir}",
        f"  git_sha: {record['git_sha']}  git_dirty: {record['git_dirty']}",
        f"  config: {record['config_path']}  config_hash: {record['config_hash']}",
        f"  split: {record['split_name']}  split_sha256: {record['split_sha256']}",
        f"  device: {record['environment']['device']}  subsets: {record['data_subsets']}",
        *(
            f"  epoch {epoch}: mean train loss {stats.mean_loss:.6f}, wall time "
            f"{stats.wall_time_s:.1f} s, {stats.images_per_s:.1f} train images/s"
            for epoch, stats in enumerate(result.epochs, start=1)
        ),
        f"  eval on {record['eval_split']} subset at threshold {record['threshold']} "
        f"({record['threshold_rule']}):",
        f"    APCER (pooled) {metrics['apcer_pooled']:.6f} = "
        f"{metrics['n_attack_accepted']}/{metrics['n_attack']}",
        f"    BPCER {metrics['bpcer']:.6f} = "
        f"{metrics['n_bona_fide_rejected']}/{metrics['n_bona_fide']}",
        f"    ACER (pooled) {metrics['acer_pooled']:.6f}",
    ]
    if record["git_dirty"]:
        lines.append("WARNING: git_dirty is true; this run cannot be cited (docs/RULES.md §3).")
    return "\n".join(lines)


def format_ledger_row(record: Mapping[str, Any]) -> str:
    """Render one ``docs/EXPERIMENTS.md`` row, in column order, from a run record.

    Rates are percentages with two decimals. Pooled APCER and ACER are marked ``(pooled)``;
    metrics that were not computed are ``—``. Notes are prefixed with a warning when
    ``git_dirty`` is true.

    Args:
        record: A completed run record.

    Returns:
        The markdown table row.
    """
    metrics = record["metrics"]
    notes = (DIRTY_NOTE if record["git_dirty"] else "") + record["notes"]
    cells = [
        f"`{record['run_id']}`",
        record["created_at"][:10],
        record["hypothesis"],
        record["what_changed"],
        f"`{record['config_path']}`",
        f"`{record['git_sha'][:7]}`",
        _percent(metrics["apcer_pooled"], POOLED_SUFFIX),
        _percent(metrics["bpcer"]),
        _percent(metrics["acer_pooled"], POOLED_SUFFIX),
        _percent(metrics["bpcer_at_apcer_1pct"]),
        notes,
    ]
    return "| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |"


def close_failed_record(
    record: dict[str, Any], status: str, error: BaseException, run_dir: Path
) -> None:
    """Mark ``record`` as failed or aborted, store the error, and rewrite ``record.json``.

    Args:
        record: The running record; updated in place.
        status: ``failed`` or ``aborted``.
        error: The exception that ended the run.
        run_dir: The run directory holding ``record.json``.
    """
    record["status"] = status
    record["error"] = f"{type(error).__name__}: {error}"
    write_json(record, run_dir / RECORD_FILENAME)
    logger.error("Run %s %s: %s", record["run_id"], status, record["error"])


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write ``payload`` as indented JSON with a trailing newline. NaN and infinity are rejected.

    Args:
        payload: A JSON-compatible mapping.
        path: The file to write.

    Raises:
        ValueError: If ``payload`` contains NaN or infinity.
    """
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _train_and_evaluate(
    inputs: RunInputs,
    subsets: Mapping[str, pd.DataFrame],
    device: torch.device,
    run_dir: Path,
    record: dict[str, Any],
) -> tuple[EpochStats, ...]:
    config = inputs.train_config
    model = build_model(config.model).to(device)
    transform = build_baseline_transform(config.model.input_size, model)
    loaders = {
        split: make_loader(
            ManifestDataset(frame, inputs.data_config.dataset_root, transform),
            config.data,
            shuffle=split == labels.SPLIT_TRAIN,
            seed=config.run.seed,
            device=device,
        )
        for split, frame in subsets.items()
    }
    epochs = train_epochs(model, loaders[labels.SPLIT_TRAIN], config, device)
    metrics, predictions = evaluate(
        model, loaders[labels.SPLIT_VAL], subsets[labels.SPLIT_VAL], device, config.eval.threshold
    )
    checkpoint_path = run_dir / CHECKPOINT_FILENAME
    checkpoint = {"model_state_dict": model.state_dict(), "backbone": config.model.backbone}
    torch.save(
        {**checkpoint, "run_id": record["run_id"], "config_hash": record["config_hash"]},
        checkpoint_path,
    )
    predictions.to_csv(run_dir / PREDICTIONS_FILENAME, index=False)
    complete_record(record, config, metrics, epochs, checkpoint_path)
    if config.wandb.enabled:
        record["artifacts"]["wandb_url"] = _log_to_wandb(inputs, record)
    return epochs


def _log_to_wandb(inputs: RunInputs, record: Mapping[str, Any]) -> str:
    # Imported lazily: runs with wandb.enabled false never import W&B or need an API key.
    import wandb

    wandb_run = wandb.init(
        project=inputs.train_config.wandb.project,
        name=record["run_id"],
        config=resolved_config(inputs.train_config, inputs.data_config),
    )
    wandb_run.summary.update({k: v for k, v in record["metrics"].items() if v is not None})
    url = str(wandb_run.url)
    wandb_run.finish()
    return url


def _percent(value: float | None, suffix: str = "") -> str:
    return EMPTY_CELL if value is None else f"{100 * value:.2f}%{suffix}"
