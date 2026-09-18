"""Load and validate an experiment config such as ``configs/baseline.yaml``."""

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICES = (DEVICE_AUTO, DEVICE_CPU, DEVICE_CUDA)
"""Allowed ``run.device`` values. ``auto`` means cuda if available, else cpu. MPS is not used."""

OPTIONAL_SECTIONS = ("cache",)
"""Sections a config may leave out. An absent section is absent from the resolved config too."""


class TrainConfigError(ValueError):
    """Raised when an experiment config is missing keys or holds invalid values."""


@dataclass(frozen=True)
class RunConfig:
    """Run description and reproducibility settings (``run:``)."""

    hypothesis: str
    what_changed: str
    notes: str
    seed: int
    device: str


@dataclass(frozen=True)
class ModelConfig:
    """Backbone settings (``model:``)."""

    backbone: str
    pretrained: bool
    input_size: int


@dataclass(frozen=True)
class SubsetConfig:
    """Subset sizes and data loading (``data:``)."""

    train_subset: int
    val_subset: int
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer and schedule (``optim:``)."""

    lr: float
    weight_decay: float
    epochs: int


@dataclass(frozen=True)
class EvalConfig:
    """Operating threshold (``eval:``)."""

    threshold: float
    threshold_rule: str


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases logging (``wandb:``)."""

    enabled: bool
    project: str


@dataclass(frozen=True)
class CacheRefConfig:
    """The normalized image cache a run reads instead of the dataset root (``cache:``).

    Attributes:
        name: The cache's identity, which must equal the ``name`` in its ``cache_summary.json``.
            A directory alone does not identify a cache.
        dir: The cache root. Machine-specific, so ``scripts/train.py --cache-dir`` overrides it.
    """

    name: str
    dir: str


@dataclass(frozen=True)
class TrainConfig:
    """A resolved experiment config. Each field is documented in ``configs/baseline.yaml``."""

    run: RunConfig
    model: ModelConfig
    data: SubsetConfig
    optim: OptimConfig
    eval: EvalConfig
    wandb: WandbConfig
    cache: CacheRefConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the config as nested plain dicts, for hashing and the resolved-config file.

        ``cache`` is left out entirely when the config does not set it, so a config with no cache
        resolves, and therefore hashes, exactly as it did before the section existed.
        """
        document = asdict(self)
        if self.cache is None:
            del document["cache"]
        return document


def load_train_config(path: Path) -> TrainConfig:
    """Load an experiment YAML into a validated :class:`TrainConfig`.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        TrainConfigError: If sections or keys are missing or unknown, a value has the wrong type,
            or a value is out of range.
    """
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    return parse_train_config(document, str(path))


def parse_train_config(document: object, source: str) -> TrainConfig:
    """Parse and validate an experiment config that is already loaded as nested plain values.

    Used for YAML files and for the ``experiment`` section of a run's ``resolved_config.json``.

    Args:
        document: The parsed document, e.g. the output of :meth:`TrainConfig.to_dict`.
        source: Where the document came from, used in error messages.

    Returns:
        The validated configuration.

    Raises:
        TrainConfigError: If sections or keys are missing or unknown, a value has the wrong type,
            or a value is out of range.
    """
    required = {
        field.name: field.type
        for field in fields(TrainConfig)
        if field.name not in OPTIONAL_SECTIONS
    }
    optional: dict[str, Any] = {"cache": CacheRefConfig}
    unknown = set(document) - set(required) - set(optional) if isinstance(document, dict) else True
    if not isinstance(document, dict) or set(required) - set(document) or unknown:
        raise TrainConfigError(
            f"{source}: expected exactly the top-level sections {sorted(required)}, optionally "
            f"with {sorted(optional)}."
        )
    sections = {name: parse_section(name, document[name], kind) for name, kind in required.items()}
    present = {
        name: parse_section(name, document[name], kind)
        for name, kind in optional.items()
        if name in document
    }
    config = TrainConfig(**sections, **present)
    validate_train_config(config)
    return config


def validate_train_config(config: TrainConfig) -> None:
    """Check value ranges. Also run on configs changed after loading (debug overrides).

    Args:
        config: The configuration to check.

    Raises:
        TrainConfigError: Listing every violated constraint.
    """
    checks = [
        (bool(config.run.hypothesis.strip()), "run.hypothesis must not be empty"),
        (config.run.seed >= 0, "run.seed must be >= 0"),
        (config.run.device in DEVICES, f"run.device must be one of {DEVICES}"),
        (bool(config.model.backbone.strip()), "model.backbone must not be empty"),
        (config.model.input_size > 0, "model.input_size must be > 0"),
        (config.data.train_subset > 0, "data.train_subset must be > 0"),
        (config.data.val_subset > 0, "data.val_subset must be > 0"),
        (config.data.batch_size > 0, "data.batch_size must be > 0"),
        (config.data.num_workers >= 0, "data.num_workers must be >= 0"),
        (config.optim.lr > 0, "optim.lr must be > 0"),
        (config.optim.weight_decay >= 0, "optim.weight_decay must be >= 0"),
        (config.optim.epochs >= 1, "optim.epochs must be >= 1"),
        (0.0 <= config.eval.threshold <= 1.0, "eval.threshold must be in [0, 1]"),
        (bool(config.eval.threshold_rule.strip()), "eval.threshold_rule must not be empty"),
        (bool(config.wandb.project.strip()), "wandb.project must not be empty"),
    ]
    if config.cache is not None:
        checks.append((bool(config.cache.name.strip()), "cache.name must not be empty"))
        checks.append((bool(config.cache.dir.strip()), "cache.dir must not be empty"))
    problems = [message for passed, message in checks if not passed]
    if problems:
        raise TrainConfigError("Invalid experiment config: " + "; ".join(problems) + ".")


def with_cache_dir(config: TrainConfig, cache_dir: Path | None) -> TrainConfig:
    """Apply the ``--cache-dir`` path override to an experiment config.

    The flag moves only where the cache is; ``cache.name`` stays the config's, because that is what
    identifies the cache a run record cites.

    Args:
        config: The loaded experiment config.
        cache_dir: The cache root from the CLI, or ``None`` to keep the config's value.

    Returns:
        The config, with ``cache.dir`` replaced when the flag is given.

    Raises:
        TrainConfigError: If a cache directory is given but the config has no ``cache`` section.
    """
    if cache_dir is None:
        return config
    if config.cache is None:
        raise TrainConfigError(
            "--cache-dir needs an experiment config with a 'cache' section, "
            "for example configs/baseline_cache.yaml."
        )
    return replace(config, cache=replace(config.cache, dir=cache_dir.as_posix()))


def parse_section(name: str, section: object, section_type: Any) -> Any:
    """Parse one config section into a dataclass, checking its keys and value types.

    Args:
        name: Section name, used in error messages.
        section: The parsed YAML value of the section.
        section_type: A dataclass whose fields are ``str``, ``int``, ``float`` or ``bool``.

    Returns:
        An instance of ``section_type``.

    Raises:
        TrainConfigError: If the section is not a mapping, keys are missing or unknown, or a value
            has the wrong type.
    """
    if not isinstance(section, dict):
        raise TrainConfigError(f"Section {name!r} must be a mapping.")
    expected = {field.name: field.type for field in fields(section_type)}
    missing, unknown = set(expected) - set(section), set(section) - set(expected)
    if missing or unknown:
        raise TrainConfigError(
            f"Section {name!r}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}."
        )
    values = {
        key: _typed_value(f"{name}.{key}", section[key], kind) for key, kind in expected.items()
    }
    return section_type(**values)


def _typed_value(key: str, value: object, expected: Any) -> object:
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, bool) != (expected is bool) or not isinstance(value, expected):
        raise TrainConfigError(
            f"{key}: expected {expected.__name__}, got {type(value).__name__} ({value!r})."
        )
    return value
