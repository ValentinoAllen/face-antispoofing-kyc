"""Seeding, determinism, and the provenance fields of the run record (``docs/SCHEMA.md`` §3)."""

import hashlib
import json
import os
import platform
import random
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch

CONFIGS_DIR = "configs"
"""Repository directory that every recorded ``config_path`` must be under."""

_CUBLAS_WORKSPACE_ENV = "CUBLAS_WORKSPACE_CONFIG"
_CUBLAS_WORKSPACE_VALUE = ":4096:8"
"""cuBLAS workspace setting PyTorch requires for deterministic CUDA matrix multiplication."""

_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_SLUG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_HASH_CHUNK_BYTES = 1 << 20


class ProvenanceError(RuntimeError):
    """Raised when a provenance field of the run record cannot be determined."""


@dataclass(frozen=True)
class GitState:
    """Commit and working-tree state at launch."""

    sha: str
    dirty: bool


def seed_everything(seed: int) -> str:
    """Seed Python, NumPy and PyTorch, and request deterministic algorithms.

    ``torch.manual_seed`` seeds the generators of every device (CPU, CUDA, MPS). DataLoader workers
    and shuffling are seeded separately with :func:`seed_worker` and :func:`make_generator`.

    Args:
        seed: The ``run.seed`` config value.

    Returns:
        A description of the determinism setting, for the run record.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault(_CUBLAS_WORKSPACE_ENV, _CUBLAS_WORKSPACE_VALUE)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return (
        "torch.use_deterministic_algorithms(True, warn_only=True), cudnn.deterministic=True, "
        "cudnn.benchmark=False; ops without a deterministic implementation log a warning "
        "instead of failing"
    )


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: seed NumPy and ``random`` from the worker's torch seed.

    Args:
        worker_id: Worker index, supplied by the DataLoader (the seed already encodes it).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a CPU generator seeded with ``seed``, for DataLoader shuffling and worker seeds.

    Args:
        seed: The ``run.seed`` config value.

    Returns:
        The seeded generator.
    """
    return torch.Generator().manual_seed(seed)


def find_repo_root(start: Path) -> Path:
    """Return the top level of the git repository containing ``start``.

    Args:
        start: Any directory inside the repository.

    Returns:
        The repository root.

    Raises:
        ProvenanceError: If git fails or ``start`` is not inside a repository.
    """
    return Path(_git(start, "rev-parse", "--show-toplevel").strip())


def git_state(repo_root: Path) -> GitState:
    """Read ``HEAD`` and whether the working tree has uncommitted or untracked changes.

    Args:
        repo_root: The repository root.

    Returns:
        The 40-character commit SHA and the dirty flag.

    Raises:
        ProvenanceError: If git fails or ``HEAD`` is not a 40-character hex SHA.
    """
    sha = _git(repo_root, "rev-parse", "HEAD").strip()
    if not _SHA_PATTERN.fullmatch(sha):
        raise ProvenanceError(f"Unexpected git SHA {sha!r}.")
    dirty = bool(_git(repo_root, "status", "--porcelain").strip())
    return GitState(sha=sha, dirty=dirty)


def repo_relative_config_path(config_path: Path, repo_root: Path) -> str:
    """Return ``config_path`` as a POSIX path relative to the repository root.

    Args:
        config_path: Path to the experiment config, absolute or relative to the working directory.
        repo_root: The repository root.

    Returns:
        E.g. ``configs/baseline.yaml``.

    Raises:
        ProvenanceError: If the file is not under ``configs/`` in the repository.
    """
    try:
        relative = config_path.resolve().relative_to(repo_root.resolve())
    except ValueError as error:
        raise ProvenanceError(f"{config_path} is not inside the repository {repo_root}.") from error
    if relative.parts[0] != CONFIGS_DIR:
        raise ProvenanceError(f"{config_path} is not under {CONFIGS_DIR}/ (docs/SCHEMA.md §3).")
    return relative.as_posix()


def to_json_compatible(value: Any) -> Any:
    """Convert paths to POSIX strings and tuples to lists, recursively.

    Args:
        value: A nested structure of mappings, sequences and scalars.

    Returns:
        The same structure, serializable with ``json``.
    """
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_json_compatible(item) for item in value]
    return value


def config_hash(resolved_config: Mapping[str, Any]) -> str:
    """SHA-256 hex digest of the fully resolved config serialized as JSON with sorted keys.

    Args:
        resolved_config: JSON-compatible resolved config.

    Returns:
        64 lowercase hex characters.
    """
    serialized = json.dumps(
        resolved_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes.

    Args:
        path: File to hash.

    Returns:
        64 lowercase hex characters.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_run_id(slug: str, now: datetime) -> str:
    """Build a ``YYYYMMDD-HHMMSS-<slug>`` run id in UTC (``docs/SCHEMA.md`` §3).

    Args:
        slug: Short name, e.g. the config file stem.
        now: Launch time. Must be timezone-aware UTC.

    Returns:
        The run id.

    Raises:
        ValueError: If ``now`` is not UTC or the slug has characters other than letters, digits,
            ``_`` and ``-``.
    """
    if now.utcoffset() != timedelta(0):
        raise ValueError(f"Run time must be timezone-aware UTC, got {now!r}.")
    if not _SLUG_PATTERN.fullmatch(slug):
        raise ValueError(f"Run slug {slug!r} must match {_SLUG_PATTERN.pattern}.")
    return f"{now:%Y%m%d-%H%M%S}-{slug}"


def environment_info(device: torch.device, determinism: str) -> dict[str, str | None]:
    """Describe the software environment for the run record.

    Args:
        device: The device the run uses.
        determinism: The description returned by :func:`seed_everything`.

    Returns:
        ``python``, ``torch``, ``cuda`` (null on CPU-only builds), ``device`` and ``platform`` from
        the contract, plus ``timm`` and ``deterministic_algorithms``.
    """
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "platform": platform.platform(),
        "timm": timm.__version__,
        "deterministic_algorithms": determinism,
    }


def _git(cwd: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
        )
    except OSError as error:
        raise ProvenanceError(f"Cannot run git in {cwd}: {error}") from error
    if result.returncode != 0:
        raise ProvenanceError(f"'git {' '.join(args)}' failed in {cwd}: {result.stderr.strip()}")
    return result.stdout
