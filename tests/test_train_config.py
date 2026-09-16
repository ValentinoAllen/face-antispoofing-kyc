"""Tests for the experiment config loader and the committed ``configs/baseline.yaml``."""

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from antispoof.training.config import TrainConfigError, load_train_config, parse_train_config

BASELINE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


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
