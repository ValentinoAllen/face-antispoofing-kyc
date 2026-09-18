"""Tests for the experiment config loader and the committed experiment configs."""

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from antispoof.training.config import (
    CacheRefConfig,
    TrainConfigError,
    load_train_config,
    parse_train_config,
    with_cache_dir,
)
from antispoof.training.reproducibility import config_hash

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = REPO_ROOT / "configs" / "baseline.yaml"
BASELINE_CACHE_CONFIG = REPO_ROOT / "configs" / "baseline_cache.yaml"


@pytest.fixture
def baseline_document() -> dict[str, Any]:
    document: dict[str, Any] = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))
    return document


def _write(tmp_path: Path, document: dict[str, Any]) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_committed_baseline_config_has_the_agreed_values() -> None:
    config = load_train_config(BASELINE_CONFIG)
    assert (config.model.backbone, config.model.pretrained, config.model.input_size) == (
        "mobilenetv3_large_100",
        True,
        224,
    )
    assert (config.data.train_subset, config.data.val_subset) == (4000, 2000)
    assert (config.data.batch_size, config.data.num_workers) == (64, 4)
    assert (config.optim.lr, config.optim.weight_decay, config.optim.epochs) == (3e-4, 1e-2, 1)
    assert (config.run.seed, config.run.device) == (42, "auto")
    assert (config.eval.threshold, config.eval.threshold_rule) == (0.5, "fixed_config")
    assert config.wandb.enabled is False


def test_integer_is_accepted_for_a_float_key(
    tmp_path: Path, baseline_document: dict[str, Any]
) -> None:
    baseline_document["eval"]["threshold"] = 1
    config = load_train_config(_write(tmp_path, baseline_document))
    assert config.eval.threshold == 1.0
    assert isinstance(config.eval.threshold, float)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("optim", "lr", "3e-4", "optim.lr: expected float, got str"),
        ("run", "seed", True, "run.seed: expected int, got bool"),
        ("model", "pretrained", 1, "model.pretrained: expected bool, got int"),
        ("run", "device", "mps", "run.device must be one of"),
        ("eval", "threshold", 1.5, r"eval.threshold must be in \[0, 1\]"),
        ("optim", "epochs", 0, "optim.epochs must be >= 1"),
        ("data", "batch_size", 0, "data.batch_size must be > 0"),
    ],
)
def test_invalid_values_raise(
    tmp_path: Path,
    baseline_document: dict[str, Any],
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    baseline_document[section][key] = value
    with pytest.raises(TrainConfigError, match=message):
        load_train_config(_write(tmp_path, baseline_document))


def test_missing_and_unknown_keys_raise(tmp_path: Path, baseline_document: dict[str, Any]) -> None:
    missing = copy.deepcopy(baseline_document)
    del missing["optim"]["lr"]
    with pytest.raises(TrainConfigError, match=r"missing keys \['lr'\]"):
        load_train_config(_write(tmp_path, missing))
    unknown = copy.deepcopy(baseline_document)
    unknown["optim"]["momentum"] = 0.9
    with pytest.raises(TrainConfigError, match=r"unknown keys \['momentum'\]"):
        load_train_config(_write(tmp_path, unknown))


def test_unknown_section_raises(tmp_path: Path, baseline_document: dict[str, Any]) -> None:
    baseline_document["schedule"] = {"warmup": 1}
    with pytest.raises(TrainConfigError, match="top-level sections"):
        load_train_config(_write(tmp_path, baseline_document))


def test_parse_train_config_round_trips_the_resolved_experiment() -> None:
    config = load_train_config(BASELINE_CONFIG)
    assert parse_train_config(config.to_dict(), "resolved_config.json") == config
    with pytest.raises(TrainConfigError, match="resolved_config.json: expected exactly"):
        parse_train_config({"run": {}}, "resolved_config.json")


def test_baseline_config_resolves_without_a_cache_key() -> None:
    config = load_train_config(BASELINE_CONFIG)
    assert config.cache is None
    assert "cache" not in config.to_dict()
    assert set(config.to_dict()) == {"run", "model", "data", "optim", "eval", "wandb"}


def test_a_recorded_runs_config_hash_still_reproduces() -> None:
    """The uncached resolved config must not move: recorded runs cite its hash.

    ``counterfactual.check_source_config_hash`` re-derives the committed baseline config's hash and
    compares it with a source run's record, so a change to the resolved shape would invalidate every
    run in ``reports/runs/`` and break the counterfactual evaluation.
    """
    run_dir = REPO_ROOT / "reports" / "runs" / "20260918-130622-baseline"
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    record = json.loads((run_dir / "record.json").read_text(encoding="utf-8"))
    rebuilt = parse_train_config(resolved["experiment"], "resolved_config.json")
    assert rebuilt.to_dict() == resolved["experiment"]
    assert config_hash(resolved) == record["config_hash"]
    assert "cache" not in resolved["experiment"]


def test_the_cache_section_is_optional_and_parsed_when_present(
    tmp_path: Path, baseline_document: dict[str, Any]
) -> None:
    baseline_document["cache"] = {"name": "cache_v1", "dir": "/kaggle/working/cache"}
    config = load_train_config(_write(tmp_path, baseline_document))
    assert config.cache == CacheRefConfig(name="cache_v1", dir="/kaggle/working/cache")
    assert config.to_dict()["cache"] == {"name": "cache_v1", "dir": "/kaggle/working/cache"}
    assert parse_train_config(config.to_dict(), "resolved_config.json") == config


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({"name": "", "dir": "/cache"}, "cache.name must not be empty"),
        ({"name": "cache_v1", "dir": " "}, "cache.dir must not be empty"),
        ({"name": "cache_v1"}, r"missing keys \['dir'\]"),
        ({"name": "cache_v1", "dir": "/cache", "size": 256}, r"unknown keys \['size'\]"),
    ],
)
def test_invalid_cache_sections_are_rejected(
    tmp_path: Path, baseline_document: dict[str, Any], section: dict[str, Any], message: str
) -> None:
    baseline_document["cache"] = section
    with pytest.raises(TrainConfigError, match=message):
        load_train_config(_write(tmp_path, baseline_document))


def test_with_cache_dir_moves_the_path_and_keeps_the_name() -> None:
    config = load_train_config(BASELINE_CACHE_CONFIG)
    assert config.cache is not None
    moved = with_cache_dir(config, Path("/elsewhere/cache"))
    assert moved.cache == CacheRefConfig(name=config.cache.name, dir="/elsewhere/cache")
    assert with_cache_dir(config, None) == config
    assert with_cache_dir(load_train_config(BASELINE_CONFIG), None) == load_train_config(
        BASELINE_CONFIG
    )


def test_cache_dir_without_a_cache_section_is_rejected() -> None:
    with pytest.raises(TrainConfigError, match="needs an experiment config with a 'cache' section"):
        with_cache_dir(load_train_config(BASELINE_CONFIG), Path("/cache"))


def test_committed_cached_baseline_differs_from_the_baseline_only_in_run_and_cache() -> None:
    baseline = load_train_config(BASELINE_CONFIG)
    cached = load_train_config(BASELINE_CACHE_CONFIG)
    assert dataclasses.replace(baseline, run=cached.run, cache=cached.cache) == cached
    assert cached.cache == CacheRefConfig(name="cache_v1", dir="/kaggle/working/cache")
    assert cached.run.what_changed == (
        "Same baseline on normalized cache (256px, q90, no face crop)"
    )
    assert cached.run.notes == (
        "cache v1; same subsets, seed and threshold as the uncached baseline"
    )
    assert cached.run.hypothesis.startswith(
        "Training the same baseline on the size- and quality-normalized cache removes the "
        "header-level class signature."
    )
    assert ">= 3x" in cached.run.hypothesis and "<= 1.5x" in cached.run.hypothesis
    assert "1.86%" in cached.run.hypothesis
    assert (cached.run.seed, cached.model.input_size) == (baseline.run.seed, 224)
