"""Tests for seeding, device selection, provenance helpers and the ledger row."""

import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import torch

from antispoof.training import reproducibility
from antispoof.training.loop import resolve_device
from antispoof.training.reproducibility import ProvenanceError
from antispoof.training.run import format_ledger_row

REPO_ROOT = Path(__file__).resolve().parents[1]


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "run_id": "20260915-010203-baseline",
        "created_at": "2026-09-15T01:02:03.000001+00:00",
        "hypothesis": "A | B",
        "what_changed": "First baseline",
        "config_path": "configs/baseline.yaml",
        "git_sha": "0123456789abcdef0123456789abcdef01234567",
        "git_dirty": False,
        "metrics": {
            "apcer_pooled": 0.125,
            "bpcer": 0.25,
            "acer_pooled": 0.1875,
            "bpcer_at_apcer_1pct": None,
        },
        "notes": "smoke",
    }
    record.update(overrides)
    return record


def test_config_hash_ignores_key_order_and_tracks_values() -> None:
    first = reproducibility.config_hash({"a": 1, "b": {"c": [1, 2], "d": "x"}})
    reordered = reproducibility.config_hash({"b": {"d": "x", "c": [1, 2]}, "a": 1})
    changed = reproducibility.config_hash({"a": 2, "b": {"c": [1, 2], "d": "x"}})
    assert first == reordered
    assert first != changed
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_to_json_compatible_converts_paths_and_tuples() -> None:
    value = {"root": Path("/kaggle/input"), "subjects": ("5028", "7332")}
    assert reproducibility.to_json_compatible(value) == {
        "root": "/kaggle/input",
        "subjects": ["5028", "7332"],
    }


def test_make_run_id_uses_utc_timestamp_and_slug() -> None:
    now = datetime(2026, 9, 15, 1, 2, 3, 999, tzinfo=UTC)
    assert reproducibility.make_run_id("baseline", now) == "20260915-010203-baseline"


@pytest.mark.parametrize(
    "now",
    [datetime(2026, 9, 15), datetime(2026, 9, 15, tzinfo=timezone(timedelta(hours=7)))],
    ids=["naive", "non-utc"],
)
def test_make_run_id_rejects_non_utc_times(now: datetime) -> None:
    with pytest.raises(ValueError, match="UTC"):
        reproducibility.make_run_id("baseline", now)


def test_repo_relative_config_path() -> None:
    config = REPO_ROOT / "configs" / "baseline.yaml"
    assert reproducibility.repo_relative_config_path(config, REPO_ROOT) == "configs/baseline.yaml"
    with pytest.raises(ProvenanceError, match="not under configs/"):
        reproducibility.repo_relative_config_path(REPO_ROOT / "pyproject.toml", REPO_ROOT)


def test_repo_relative_config_path_rejects_paths_outside_the_repo(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="not inside the repository"):
        reproducibility.repo_relative_config_path(tmp_path / "baseline.yaml", REPO_ROOT)


def test_git_state_reads_this_repository() -> None:
    state = reproducibility.git_state(REPO_ROOT)
    assert re.fullmatch(r"[0-9a-f]{40}", state.sha)
    assert isinstance(state.dirty, bool)


def test_git_state_outside_a_repository_raises(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="failed"):
        reproducibility.git_state(tmp_path)


def test_seed_everything_makes_torch_draws_repeatable() -> None:
    reproducibility.seed_everything(123)
    first = torch.rand(4)
    reproducibility.seed_everything(123)
    assert torch.equal(first, torch.rand(4))


def test_resolve_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="Unsupported device 'mps'"):
        resolve_device("mps")


def test_format_ledger_row_follows_experiments_column_order() -> None:
    assert format_ledger_row(_record()) == (
        "| `20260915-010203-baseline` | 2026-09-15 | A \\| B | First baseline | "
        "`configs/baseline.yaml` | `0123456` | 12.50% (pooled) | 25.00% | 18.75% (pooled) | — | "
        "smoke |"
    )


def test_format_ledger_row_flags_dirty_runs() -> None:
    row = format_ledger_row(_record(git_dirty=True))
    assert row.endswith("| git_dirty: not citable; smoke |")
