"""Header-metadata probe: can how the images were captured and stored separate live from spoof?

In CelebA-Spoof the live images are CelebA web photos, while the dataset authors recaptured the
spoof images. The two classes may therefore differ in resolution, format, compression and metadata.
This probe reads **image headers only**, never pixel data, for exactly the baseline's train and val
subsets, and reports:

1. per-class value counts and quartiles of every header feature;
2. a descriptive ranking of the features by single-feature val ROC AUC (not a PAD metric);
3. pooled PAD metrics (:func:`antispoof.eval.pad_metrics.pad_metrics`) of two classifiers fitted
   on the train subset and evaluated on the val subset at the configured threshold;
4. optionally, how a baseline CNN's ``predictions.csv`` agrees with the primary classifier.

No part of ``image_path`` other than the file extension is a feature: the ``live/`` or ``spoof/``
path segment is the label by construction.
"""

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
from PIL import Image, JpegImagePlugin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from antispoof.data import labels
from antispoof.data.build import summarize_split
from antispoof.data.config import DataConfig
from antispoof.data.dataset import ImageLoadError
from antispoof.eval.pad_metrics import PadMetrics, pad_metrics
from antispoof.training import reproducibility
from antispoof.training.config import (
    DEVICE_CPU,
    EvalConfig,
    TrainConfig,
    TrainConfigError,
    parse_section,
    validate_train_config,
)
from antispoof.training.run import (
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
    write_json,
)

logger = logging.getLogger(__name__)

PROBE_SUMMARY_FILENAME = "probe_summary.json"
FEATURES_FILENAME = "features.csv"

IDENTIFIER_COLUMNS = ("image_path", "split", "label")
"""Columns kept in ``features.csv`` to join and group rows. They are never classifier inputs."""

HEADER_FEATURES = (
    "extension",
    "format",
    "mode",
    "width",
    "height",
    "aspect_ratio",
    "pixel_count",
    "file_size_bytes",
    "bytes_per_pixel",
    "jpeg_subsampling",
    "jpeg_progressive",
    "jpeg_luma_quant_mean",
    "has_exif",
    "has_icc_profile",
)
"""The header features, in ``features.csv`` order. Each is also ranked on its own."""

CATEGORICAL_FEATURES = ("extension", "format", "mode", "jpeg_subsampling")
NUMERIC_FEATURES = (
    "width",
    "height",
    "aspect_ratio",
    "pixel_count",
    "file_size_bytes",
    "bytes_per_pixel",
    "jpeg_luma_quant_mean",
)
FLAG_FEATURES = ("jpeg_progressive", "has_exif", "has_icc_profile")
JPEG_ONLY_FEATURES = ("jpeg_subsampling", "jpeg_progressive", "jpeg_luma_quant_mean")
"""Null for files that are not JPEG. Each has a ``<name>_missing`` indicator column."""

MISSING_INDICATORS = tuple(f"{name}_missing" for name in JPEG_ONLY_FEATURES)
MODEL_FEATURES = HEADER_FEATURES + MISSING_INDICATORS
"""Every classifier input column. ``image_path``, ``split`` and ``label`` are not among them."""

COUNTED_FEATURES = CATEGORICAL_FEATURES + FLAG_FEATURES

SUBSAMPLING_NAMES = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}
"""Names of ``JpegImagePlugin.get_sampling`` codes."""
SUBSAMPLING_OTHER = "other"
"""Subsampling of grayscale or CMYK JPEGs, or of a sampling Pillow does not recognize."""

MISSING_CATEGORY = "missing"
IMPUTE_STRATEGIES = ("mean", "median")
MAX_BINS_LIMIT = 255
"""Largest ``max_bins`` scikit-learn accepts for HistGradientBoosting."""

CLASS_NAMES = ((labels.LABEL_LIVE, "live"), (labels.LABEL_SPOOF, "spoof"))
SUBSET_SPLITS = (labels.SPLIT_TRAIN, labels.SPLIT_VAL)
PRIMARY_MODEL = "HistGradientBoostingClassifier"
SECONDARY_MODEL = "LogisticRegression"
AUC_LABEL = "val ROC AUC (descriptive ranking, not a PAD metric)"

HeaderValue = str | int | float | bool | None


class ProbeConfigError(ValueError):
    """Raised when the probe config is invalid or disagrees with the baseline config."""


class ProbeInputError(ValueError):
    """Raised when a predictions file does not match the val subset exactly."""


@dataclass(frozen=True)
class ProbeRunConfig:
    """Run description (``run:``), copied into the run record and the ledger row."""

    hypothesis: str
    what_changed: str
    notes: str


@dataclass(frozen=True)
class SubsetSourceConfig:
    """Where the subset sizes and seed come from (``subsets:``)."""

    config: str


@dataclass(frozen=True)
class FeatureTransformConfig:
    """Feature transforms for the logistic regression (``features:``)."""

    log1p: tuple[str, ...]


@dataclass(frozen=True)
class BoostingConfig:
    """HistGradientBoostingClassifier hyperparameters (``primary:``)."""

    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_bins: int
    early_stopping: bool


@dataclass(frozen=True)
class LogisticConfig:
    """Logistic regression hyperparameters (``secondary:``)."""

    c: float
    max_iter: int
    impute_strategy: str


@dataclass(frozen=True)
class ProbeConfig:
    """A resolved probe config. Each field is documented in ``configs/probe_metadata.yaml``."""

    run: ProbeRunConfig
    subsets: SubsetSourceConfig
    features: FeatureTransformConfig
    primary: BoostingConfig
    secondary: LogisticConfig
    eval: EvalConfig

    def to_dict(self) -> dict[str, Any]:
        """Return the config as nested plain dicts, for hashing and the resolved-config file."""
        return asdict(self)


_SECTION_TYPES = {
    "run": ProbeRunConfig,
    "subsets": SubsetSourceConfig,
    "primary": BoostingConfig,
    "secondary": LogisticConfig,
    "eval": EvalConfig,
}


@dataclass(frozen=True)
class ProbeInputs:
    """Everything a probe run needs. ``data_config`` already carries CLI path overrides.

    Attributes:
        probe_config: The probe config.
        config_path: Path of the probe config, under ``configs/``.
        subset_config: The experiment config whose subset sizes and seed the probe reuses.
        subset_config_path: Path of that config, under ``configs/``.
        data_config: Manifest directory, dataset root and split assignment.
        output_dir: Parent of the run directory.
        repo_root: The repository root.
        predictions_path: A baseline ``predictions.csv`` to compare with, or ``None``.
    """

    probe_config: ProbeConfig
    config_path: Path
    subset_config: TrainConfig
    subset_config_path: Path
    data_config: DataConfig
    output_dir: Path
    repo_root: Path
    predictions_path: Path | None


@dataclass(frozen=True)
class ProbeResult:
    """The final run record, its directory and the analysis summary."""

    record: dict[str, Any]
    run_dir: Path
    summary: dict[str, Any]


def load_probe_config(path: Path) -> ProbeConfig:
    """Load a probe YAML such as ``configs/probe_metadata.yaml`` into a validated config.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        ProbeConfigError: If sections or keys are missing or unknown, a value has the wrong type,
            or a value is out of range.
    """
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    names = {field.name for field in fields(ProbeConfig)}
    if not isinstance(document, dict) or set(document) != names:
        raise ProbeConfigError(f"{path}: expected exactly the top-level sections {sorted(names)}.")
    try:
        sections = {
            name: parse_section(name, document[name], kind) for name, kind in _SECTION_TYPES.items()
        }
    except TrainConfigError as error:
        raise ProbeConfigError(f"{path}: {error}") from error
    config = ProbeConfig(**sections, features=_parse_features(document["features"]))
    validate_probe_config(config)
    return config


def validate_probe_config(config: ProbeConfig) -> None:
    """Check value ranges and feature names.

    Args:
        config: The probe configuration.

    Raises:
        ProbeConfigError: Listing every violated constraint.
    """
    primary, secondary, log1p = config.primary, config.secondary, config.features.log1p
    checks = [
        (bool(config.run.hypothesis.strip()), "run.hypothesis must not be empty"),
        (bool(config.run.what_changed.strip()), "run.what_changed must not be empty"),
        (bool(config.subsets.config.strip()), "subsets.config must not be empty"),
        (set(log1p) <= set(NUMERIC_FEATURES), f"features.log1p must be in {NUMERIC_FEATURES}"),
        (len(set(log1p)) == len(log1p), "features.log1p must not repeat a feature"),
        (primary.learning_rate > 0, "primary.learning_rate must be > 0"),
        (primary.max_iter >= 1, "primary.max_iter must be >= 1"),
        (primary.max_leaf_nodes >= 2, "primary.max_leaf_nodes must be >= 2"),
        (primary.min_samples_leaf >= 1, "primary.min_samples_leaf must be >= 1"),
        (primary.l2_regularization >= 0, "primary.l2_regularization must be >= 0"),
        (
            2 <= primary.max_bins <= MAX_BINS_LIMIT,
            f"primary.max_bins must be in [2, {MAX_BINS_LIMIT}]",
        ),
        (secondary.c > 0, "secondary.c must be > 0"),
        (secondary.max_iter >= 1, "secondary.max_iter must be >= 1"),
        (
            secondary.impute_strategy in IMPUTE_STRATEGIES,
            f"secondary.impute_strategy must be one of {IMPUTE_STRATEGIES}",
        ),
        (0.0 <= config.eval.threshold <= 1.0, "eval.threshold must be in [0, 1]"),
        (bool(config.eval.threshold_rule.strip()), "eval.threshold_rule must not be empty"),
    ]
    problems = [message for passed, message in checks if not passed]
    if problems:
        raise ProbeConfigError("Invalid probe config: " + "; ".join(problems) + ".")


def check_probe_inputs(probe_config: ProbeConfig, subset_config: TrainConfig) -> None:
    """Validate the subset config and require the probe to use its threshold.

    Args:
        probe_config: The probe config.
        subset_config: The experiment config whose subsets the probe reuses.

    Raises:
        TrainConfigError: If the subset config is invalid.
        ProbeConfigError: If the probe threshold differs from ``subset_config.eval.threshold``.
    """
    validate_train_config(subset_config)
    if probe_config.eval.threshold != subset_config.eval.threshold:
        raise ProbeConfigError(
            f"eval.threshold {probe_config.eval.threshold} differs from the subset config's "
            f"{subset_config.eval.threshold}; the probe must use the baseline's threshold."
        )


def read_header_features(path: Path) -> dict[str, HeaderValue]:
    """Read the header features of one image file without decoding any pixel data.

    Only attributes that ``PIL.Image.open`` fills while parsing the header are used, plus the file
    size from ``stat``. For JPEG, Pillow parses every marker up to the start of scan (SOF, DQT,
    APP); for PNG, every chunk before the first IDAT. An EXIF chunk stored after IDAT in a PNG is
    therefore not seen, and ``has_exif`` is false for it.

    Args:
        path: Image file path.

    Returns:
        One value per name in ``HEADER_FEATURES``; the ``JPEG_ONLY_FEATURES`` are ``None`` for
        other formats.

    Raises:
        ImageLoadError: If the file does not exist, its header cannot be parsed, or it declares zero
            pixels. The message names the path.
    """
    if not path.is_file():
        raise ImageLoadError(f"Image file not found: {path}")
    try:
        with Image.open(path) as image:
            return _header_values(image, path)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        raise ImageLoadError(f"Cannot read image header {path}: {error}") from error


def _header_values(image: Image.Image, path: Path) -> dict[str, HeaderValue]:
    width, height = image.size
    pixel_count = width * height
    if pixel_count == 0:
        raise ImageLoadError(f"Image header of {path} declares zero pixels.")
    file_size_bytes = path.stat().st_size
    values: dict[str, HeaderValue] = {
        "extension": path.suffix.lower(),
        "format": image.format,
        "mode": image.mode,
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "pixel_count": pixel_count,
        "file_size_bytes": file_size_bytes,
        "bytes_per_pixel": file_size_bytes / pixel_count,
        "has_exif": "exif" in image.info,
        "has_icc_profile": bool(image.info.get("icc_profile")),
    }
    values.update(_jpeg_values(image))
    return values


def _jpeg_values(image: Image.Image) -> dict[str, HeaderValue]:
    if not isinstance(image, JpegImagePlugin.JpegImageFile):
        return dict.fromkeys(JPEG_ONLY_FEATURES)
    # layer holds (component id, h sampling, v sampling, quantization table id); Y comes first.
    luma_table = image.quantization.get(image.layer[0][3]) if image.layer else None
    return {
        "jpeg_subsampling": SUBSAMPLING_NAMES.get(
            JpegImagePlugin.get_sampling(image), SUBSAMPLING_OTHER
        ),
        "jpeg_progressive": bool(image.info.get("progressive") or image.info.get("progression")),
        "jpeg_luma_quant_mean": float(np.mean(luma_table)) if luma_table else None,
    }


_FEATURE_DTYPES = {
    **dict.fromkeys(CATEGORICAL_FEATURES, object),
    **dict.fromkeys(("width", "height", "pixel_count", "file_size_bytes"), "int64"),
    **dict.fromkeys(("aspect_ratio", "bytes_per_pixel", "jpeg_luma_quant_mean"), "float64"),
    "jpeg_progressive": "boolean",
    "has_exif": bool,
    "has_icc_profile": bool,
}


def extract_features(subset: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    """Read the header features of every row of a manifest subset.

    Args:
        subset: Frame with ``image_path``, ``split`` and ``label`` columns.
        dataset_root: Directory the ``image_path`` values are relative to.

    Returns:
        ``IDENTIFIER_COLUMNS`` followed by ``MODEL_FEATURES``, one row per subset row, in order.

    Raises:
        ImageLoadError: If an image header cannot be read.
    """
    records = [read_header_features(dataset_root / path) for path in subset["image_path"]]
    features = pd.DataFrame.from_records(records, columns=list(HEADER_FEATURES))
    features = features.astype(_FEATURE_DTYPES)
    for name in JPEG_ONLY_FEATURES:
        features[f"{name}_missing"] = features[name].isna()
    identifiers = subset[list(IDENTIFIER_COLUMNS)].reset_index(drop=True)
    return pd.concat([identifiers, features], axis=1)


def measured(value: float | None, na_reason: str | None = None) -> dict[str, Any]:
    """Wrap a value, or ``None`` with the reason it is not available (reported as ``n/a``).

    Args:
        value: The value, or ``None`` when it is undefined.
        na_reason: Why the value is undefined; required when ``value`` is ``None``.

    Returns:
        ``{"value": value, "na_reason": na_reason}``.

    Raises:
        ValueError: If ``value`` is ``None`` without a reason.
    """
    if value is None and not na_reason:
        raise ValueError("An unavailable value needs a reason.")
    return {"value": value, "na_reason": na_reason}


def class_summaries(features: pd.DataFrame) -> dict[str, Any]:
    """Describe the header features per split and class.

    Args:
        features: From :func:`extract_features`, for one or more splits.

    Returns:
        ``{split: {class: {"n", "counts", "numeric"}}}``. ``counts`` holds value counts, with
        ``null`` counted explicitly, of ``COUNTED_FEATURES``. ``numeric`` holds the median, first
        and third quartile and IQR of ``NUMERIC_FEATURES``.
    """
    summaries: dict[str, Any] = {}
    for split in SUBSET_SPLITS:
        in_split = features[features["split"] == split]
        summaries[split] = {
            name: _describe(in_split[in_split["label"] == value]) for value, name in CLASS_NAMES
        }
    return summaries


def _describe(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "n": len(rows),
        "counts": {name: _value_counts(rows[name]) for name in COUNTED_FEATURES},
        "numeric": {name: _quartiles(rows[name]) for name in NUMERIC_FEATURES},
    }


def _value_counts(column: pd.Series) -> dict[str, int]:
    counts = column.value_counts(dropna=False)
    return dict(
        sorted(("null" if pd.isna(key) else str(key), int(count)) for key, count in counts.items())
    )


def _quartiles(column: pd.Series) -> dict[str, Any]:
    values = numeric_column(column)
    values = values[~np.isnan(values)]
    if values.size == 0:
        empty = dict.fromkeys(("median", "q1", "q3", "iqr"))
        return {"n_non_null": 0, **empty, "na_reason": "no non-null values"}
    q1, median, q3 = (float(value) for value in np.percentile(values, [25, 50, 75]))
    return {
        "n_non_null": int(values.size),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "na_reason": None,
    }


def numeric_column(column: pd.Series) -> np.ndarray:
    """Convert a numeric, boolean or nullable column to float64 with NaN for missing values.

    Args:
        column: The column.

    Returns:
        A float64 array.
    """
    return np.asarray(column.astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan))


def learn_categories(train: pd.DataFrame, columns: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Collect the sorted non-null categories of each categorical column on the train subset.

    Args:
        train: Train features.
        columns: Columns to encode; only those in ``CATEGORICAL_FEATURES`` get categories.

    Returns:
        ``{column: categories}``.
    """
    return {
        name: tuple(sorted({value for value in train[name] if isinstance(value, str)}))
        for name in columns
        if name in CATEGORICAL_FEATURES
    }


def boosting_matrix(
    frame: pd.DataFrame, columns: Sequence[str], categories: Mapping[str, tuple[str, ...]]
) -> np.ndarray:
    """Encode features for HistGradientBoosting: category codes or floats, NaN for missing.

    Args:
        frame: Features.
        columns: Columns, in matrix order.
        categories: From :func:`learn_categories`. A category not seen on train becomes NaN.

    Returns:
        A float64 matrix of shape ``(len(frame), len(columns))``.
    """
    encoded: list[Any] = []
    for name in columns:
        if name in categories:
            codes = {category: float(code) for code, category in enumerate(categories[name])}
            encoded.append([codes.get(value, np.nan) for value in frame[name]])
        else:
            encoded.append(numeric_column(frame[name]))
    return np.asarray(np.column_stack(encoded), dtype=np.float64)


def fit_boosting(
    train: pd.DataFrame, columns: Sequence[str], config: BoostingConfig, seed: int
) -> tuple[Any, dict[str, tuple[str, ...]]]:
    """Fit the primary classifier (HistGradientBoosting, no class weights) on the train subset.

    Args:
        train: Train features with ``label``.
        columns: Input columns.
        config: Hyperparameters.
        seed: ``random_state``.

    Returns:
        The fitted model and the categories used to encode its inputs.
    """
    categories = learn_categories(train, columns)
    mask = np.array([bool(categories.get(name)) for name in columns])
    model = HistGradientBoostingClassifier(
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        max_bins=config.max_bins,
        early_stopping=config.early_stopping,
        categorical_features=mask if mask.any() else None,
        random_state=seed,
    )
    model.fit(boosting_matrix(train, columns, categories), _label_array(train))
    return model, categories


def logistic_frame(features: pd.DataFrame, log1p_columns: Sequence[str]) -> pd.DataFrame:
    """Prepare ``MODEL_FEATURES`` for the logistic regression pipeline.

    Args:
        features: Features.
        log1p_columns: Numeric columns transformed with ``log1p``.

    Returns:
        Categorical columns as strings with ``missing`` for null; other columns as float64 with NaN.
    """
    columns: dict[str, Any] = {}
    for name in MODEL_FEATURES:
        if name in CATEGORICAL_FEATURES:
            values = [
                value if isinstance(value, str) else MISSING_CATEGORY for value in features[name]
            ]
            columns[name] = pd.Series(values, dtype=object)
        else:
            numeric = numeric_column(features[name])
            columns[name] = np.log1p(numeric) if name in log1p_columns else numeric
    return pd.DataFrame(columns)


def fit_logistic(train: pd.DataFrame, config: ProbeConfig, seed: int) -> Any:
    """Fit the secondary classifier: one-hot categoricals, standardized numerics, no class weights.

    Args:
        train: Train features with ``label``.
        config: The probe config (``secondary`` and ``features.log1p``).
        seed: ``random_state``.

    Returns:
        The fitted pipeline; it takes :func:`logistic_frame` output.
    """
    categorical = [name for name in MODEL_FEATURES if name in CATEGORICAL_FEATURES]
    numeric = [name for name in MODEL_FEATURES if name not in CATEGORICAL_FEATURES]
    preprocess = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy=config.secondary.impute_strategy), StandardScaler()
                ),
                numeric,
            ),
        ]
    )
    classifier = LogisticRegression(
        C=config.secondary.c, max_iter=config.secondary.max_iter, random_state=seed
    )
    pipeline = make_pipeline(preprocess, classifier)
    pipeline.fit(logistic_frame(train, config.features.log1p), _label_array(train))
    return pipeline


def spoof_scores(model: Any, inputs: Any) -> np.ndarray:
    """Return the predicted probability of spoof for every input row.

    Args:
        model: A fitted scikit-learn classifier.
        inputs: Inputs in the form the model was fitted on.

    Returns:
        A float64 array of scores in [0, 1].
    """
    spoof_column = list(model.classes_).index(labels.LABEL_SPOOF)
    return np.asarray(model.predict_proba(inputs)[:, spoof_column], dtype=np.float64)


def single_feature_aucs(
    train: pd.DataFrame, val: pd.DataFrame, config: BoostingConfig, seed: int
) -> list[dict[str, Any]]:
    """Rank the header features by the val ROC AUC of the primary classifier fitted on each alone.

    This is a descriptive ranking, not a PAD metric.

    Args:
        train: Train features.
        val: Val features.
        config: Primary classifier hyperparameters.
        seed: ``random_state``.

    Returns:
        ``{"feature", "value", "na_reason"}`` per feature, highest AUC first, unavailable last.
    """
    val_labels = _label_array(val)
    results: list[dict[str, Any]] = []
    for name in HEADER_FEATURES:
        if train[name].isna().all():
            results.append({"feature": name, **measured(None, "no non-null value on train")})
        elif np.unique(val_labels).size < 2:
            results.append({"feature": name, **measured(None, "val has a single class")})
        else:
            model, categories = fit_boosting(train, [name], config, seed)
            scores = spoof_scores(model, boosting_matrix(val, [name], categories))
            auc = float(roc_auc_score(val_labels, scores))
            results.append({"feature": name, **measured(auc)})
    return sorted(results, key=_auc_sort_key)


def _auc_sort_key(entry: Mapping[str, Any]) -> tuple[bool, float]:
    value = entry["value"]
    return (value is None, 0.0 if value is None else -value)


def classifier_scores(
    train: pd.DataFrame, val: pd.DataFrame, config: ProbeConfig, seed: int
) -> dict[str, np.ndarray]:
    """Fit both classifiers on all ``MODEL_FEATURES`` of the train subset and score the val subset.

    Args:
        train: Train features.
        val: Val features.
        config: The probe config.
        seed: ``random_state``.

    Returns:
        ``{PRIMARY_MODEL: scores, SECONDARY_MODEL: scores}``, in val row order.
    """
    primary, categories = fit_boosting(train, MODEL_FEATURES, config.primary, seed)
    secondary = fit_logistic(train, config, seed)
    return {
        PRIMARY_MODEL: spoof_scores(primary, boosting_matrix(val, MODEL_FEATURES, categories)),
        SECONDARY_MODEL: spoof_scores(secondary, logistic_frame(val, config.features.log1p)),
    }


def read_predictions(path: Path) -> pd.DataFrame:
    """Read a baseline ``predictions.csv``.

    Args:
        path: The file written by ``scripts/train.py``.

    Returns:
        The predictions frame.

    Raises:
        ProbeInputError: If ``image_path``, ``label`` or ``score`` is missing.
    """
    predictions = pd.read_csv(path, dtype={"image_path": str, "subject_id": str})
    missing = {"image_path", "label", "score"} - set(predictions.columns)
    if missing:
        raise ProbeInputError(f"{path} is missing columns {sorted(missing)}.")
    return predictions


def check_predictions_match(predictions: pd.DataFrame, val: pd.DataFrame) -> None:
    """Require the predictions to cover exactly the val subset's images, once each.

    Args:
        predictions: Frame with ``image_path``.
        val: The val subset or its features.

    Raises:
        ProbeInputError: If an image is missing, extra or duplicated in the predictions.
    """
    predicted, expected = set(predictions["image_path"]), set(val["image_path"])
    duplicated = int(predictions["image_path"].duplicated().sum())
    missing, extra = len(expected - predicted), len(predicted - expected)
    if missing or extra or duplicated:
        raise ProbeInputError(
            "The predictions' image_path set must equal the val subset exactly: "
            f"{missing} missing, {extra} extra, {duplicated} duplicated."
        )


def compare_with_predictions(
    val: pd.DataFrame, primary_scores: np.ndarray, predictions: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    """Compare CNN scores with the primary metadata classifier's scores, per class.

    An error is a spoof scored below ``threshold`` or a live image scored at or above it.

    Args:
        val: Val features with ``image_path`` and ``label``, in the order of ``primary_scores``.
        primary_scores: Primary classifier scores on ``val``.
        predictions: Baseline predictions with ``image_path``, ``label`` and ``score``.
        threshold: The decision threshold.

    Returns:
        ``{"live": ..., "spoof": ...}`` with ``n``, ``spearman_rho``, error counts, the count of
        shared errors expected if the two were independent, and the fraction of CNN errors that
        are also metadata errors.

    Raises:
        ProbeInputError: If the image sets differ or a label disagrees.
    """
    check_predictions_match(predictions, val)
    cnn = predictions[["image_path", "label", "score"]].rename(
        columns={"label": "cnn_label", "score": "cnn_score"}
    )
    joined = (
        val[["image_path", "label"]]
        .assign(metadata_score=primary_scores)
        .merge(cnn, on="image_path", how="inner", validate="one_to_one")
    )
    disagreeing = int((joined["label"] != joined["cnn_label"]).sum())
    if disagreeing:
        raise ProbeInputError(f"{disagreeing} predictions carry a label that differs from val.")
    return {
        name: _class_comparison(joined[joined["label"] == value], threshold, value)
        for value, name in CLASS_NAMES
    }


def _class_comparison(rows: pd.DataFrame, threshold: float, label: int) -> dict[str, Any]:
    cnn_scores = rows["cnn_score"].to_numpy(dtype=np.float64)
    metadata_scores = rows["metadata_score"].to_numpy(dtype=np.float64)
    cnn_errors = _errors(cnn_scores, threshold, label)
    metadata_errors = _errors(metadata_scores, threshold, label)
    n, n_cnn = len(rows), int(cnn_errors.sum())
    n_metadata, n_both = int(metadata_errors.sum()), int((cnn_errors & metadata_errors).sum())
    return {
        "n": n,
        "spearman_rho": spearman_correlation(cnn_scores, metadata_scores),
        "cnn_errors": n_cnn,
        "metadata_errors": n_metadata,
        "both_errors": n_both,
        "both_errors_expected_if_independent": (
            measured(n_cnn * n_metadata / n) if n else measured(None, "no rows in this class")
        ),
        "cnn_errors_also_metadata_errors": (
            measured(n_both / n_cnn) if n_cnn else measured(None, "no CNN errors in this class")
        ),
    }


def _errors(scores: np.ndarray, threshold: float, label: int) -> np.ndarray:
    if label == labels.LABEL_SPOOF:
        return np.asarray(scores < threshold)
    return np.asarray(scores >= threshold)


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Spearman's rank correlation: the Pearson correlation of average ranks.

    Args:
        first: Scores.
        second: Scores of the same rows.

    Returns:
        :func:`measured` rho, or n/a when there are fewer than two rows or a score is constant.
    """
    if first.size < 2:
        return measured(None, "fewer than two rows")
    deviations = []
    for scores in (first, second):
        ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
        deviations.append(ranks - ranks.mean())
    denominator = float(np.sqrt(np.sum(deviations[0] ** 2) * np.sum(deviations[1] ** 2)))
    if denominator == 0.0:
        return measured(None, "a score is constant across this class")
    return measured(float(np.dot(deviations[0], deviations[1]) / denominator))


def analyze(
    train: pd.DataFrame,
    val: pd.DataFrame,
    config: ProbeConfig,
    seed: int,
    predictions: pd.DataFrame | None,
) -> tuple[dict[str, Any], PadMetrics]:
    """Run the four analyses on extracted features.

    Args:
        train: Train features.
        val: Val features.
        config: The probe config.
        seed: ``random_state`` for both classifiers.
        predictions: Baseline predictions to compare with, or ``None``.

    Returns:
        The summary and the primary classifier's pooled val metrics.

    Raises:
        PadMetricsError, ProbeInputError: If a class is absent, or the predictions do not match val.
    """
    threshold = config.eval.threshold
    scores = classifier_scores(train, val, config, seed)
    metrics = {
        name: pad_metrics(_label_array(val), values, threshold) for name, values in scores.items()
    }
    comparison = (
        None
        if predictions is None
        else compare_with_predictions(val, scores[PRIMARY_MODEL], predictions, threshold)
    )
    classifiers = {
        role: {"model": name, "pad_metrics": metrics[name].to_dict()}
        for role, name in (("primary", PRIMARY_MODEL), ("secondary", SECONDARY_MODEL))
    }
    summary = {
        "class_summaries": class_summaries(pd.concat([train, val], ignore_index=True)),
        "single_feature_aucs": {
            "label": AUC_LABEL,
            "features": single_feature_aucs(train, val, config.primary, seed),
        },
        "classifiers": {
            "threshold": threshold,
            "threshold_rule": config.eval.threshold_rule,
            **classifiers,
        },
        "predictions_comparison": comparison,
    }
    return summary, metrics[PRIMARY_MODEL]


def probe_resolved_config(inputs: ProbeInputs) -> dict[str, Any]:
    """Return the fully resolved probe config that is hashed and saved with the run.

    Args:
        inputs: The probe inputs.

    Returns:
        JSON-compatible ``{"experiment": ..., "subsets": ..., "data": ...}``. ``subsets`` holds the
        subset config's path, subset sizes and seed.

    Raises:
        ProvenanceError: If the subset config is not under ``configs/`` in the repository.
    """
    subset = inputs.subset_config
    resolved: dict[str, Any] = reproducibility.to_json_compatible(
        {
            "experiment": inputs.probe_config.to_dict(),
            "subsets": {
                "config_path": reproducibility.repo_relative_config_path(
                    inputs.subset_config_path, inputs.repo_root
                ),
                "train_subset": subset.data.train_subset,
                "val_subset": subset.data.val_subset,
                "seed": subset.run.seed,
            },
            "data": asdict(inputs.data_config),
        }
    )
    return resolved


def run_probe(inputs: ProbeInputs) -> ProbeResult:
    """Extract header features, fit and evaluate the probe classifiers, and write the run directory.

    Writes ``record.json`` and ``resolved_config.json`` (``docs/SCHEMA.md`` §3),
    ``probe_summary.json`` and ``features.csv`` to ``<output_dir>/<run_id>/``. The test split is
    never read and no pixel data is decoded.

    Args:
        inputs: Configs, paths and the optional predictions file.

    Returns:
        The completed record, the run directory and the summary.

    Raises:
        TrainConfigError, ProbeConfigError, ProbeInputError, SplitLeakageError, SplitCoverageError,
            ProvenanceError: Before the run directory is created.
        Exception: Anything raised while extracting or analyzing (e.g. ``ImageLoadError``). The
            record is first rewritten with status ``failed`` (``aborted`` on
            ``KeyboardInterrupt``).
    """
    check_probe_inputs(inputs.probe_config, inputs.subset_config)
    subsets = load_subsets(inputs.subset_config, inputs.data_config)
    predictions = None
    if inputs.predictions_path is not None:
        predictions = read_predictions(inputs.predictions_path)
        check_predictions_match(predictions, subsets[labels.SPLIT_VAL])
    record, run_dir = _start_record(inputs, subsets)
    try:
        summary = _analyze_and_write(inputs, subsets, predictions, run_dir, record)
    except KeyboardInterrupt as error:
        close_failed_record(record, STATUS_ABORTED, error, run_dir)
        raise
    except Exception as error:
        close_failed_record(record, STATUS_FAILED, error, run_dir)
        raise
    write_json(record, run_dir / RECORD_FILENAME)
    return ProbeResult(record=record, run_dir=run_dir, summary=summary)


def _start_record(
    inputs: ProbeInputs, subsets: Mapping[str, pd.DataFrame]
) -> tuple[dict[str, Any], Path]:
    seed = inputs.subset_config.run.seed
    determinism = reproducibility.seed_everything(seed)
    created_at = datetime.now(UTC)
    run_id = reproducibility.make_run_id(inputs.config_path.stem, created_at)
    environment = reproducibility.environment_info(torch.device(DEVICE_CPU), determinism)
    header = RecordHeader(
        hypothesis=inputs.probe_config.run.hypothesis,
        what_changed=inputs.probe_config.run.what_changed,
        notes=inputs.probe_config.run.notes,
        seed=seed,
        config_path=inputs.config_path,
        repo_root=inputs.repo_root,
        data_config=inputs.data_config,
        resolved_config=probe_resolved_config(inputs),
    )
    record = build_record(header, run_id, created_at, environment)
    record["data_subsets"] = {
        name: asdict(summarize_split(frame)) for name, frame in subsets.items()
    }
    record["training_epochs"] = None
    run_dir = inputs.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(probe_resolved_config(inputs), run_dir / RESOLVED_CONFIG_FILENAME)
    write_json(record, run_dir / RECORD_FILENAME)
    logger.info("Probe %s started; writing to %s", run_id, run_dir)
    return record, run_dir


def _analyze_and_write(
    inputs: ProbeInputs,
    subsets: Mapping[str, pd.DataFrame],
    predictions: pd.DataFrame | None,
    run_dir: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    root = inputs.data_config.dataset_root
    features = {split: extract_features(subsets[split], root) for split in SUBSET_SPLITS}
    pd.concat(features.values(), ignore_index=True).to_csv(run_dir / FEATURES_FILENAME, index=False)
    analysis, primary_metrics = analyze(
        features[labels.SPLIT_TRAIN],
        features[labels.SPLIT_VAL],
        inputs.probe_config,
        inputs.subset_config.run.seed,
        predictions,
    )
    path = inputs.predictions_path
    summary = {
        "run_id": record["run_id"],
        "inputs": {
            "predictions_path": None if path is None else path.as_posix(),
            "predictions_sha256": None if path is None else reproducibility.sha256_file(path),
            "pillow": record["environment"]["pillow"],
            "sklearn": record["environment"]["sklearn"],
        },
        **analysis,
    }
    record["status"] = STATUS_COMPLETED
    fill_pooled_metrics(record, primary_metrics, inputs.probe_config.eval.threshold_rule)
    write_json(summary, run_dir / PROBE_SUMMARY_FILENAME)
    return summary


def format_probe_report(result: ProbeResult) -> str:
    """Render the provenance and the full analysis of a completed probe run as plain text.

    Args:
        result: From :func:`run_probe`.

    Returns:
        Multi-line text.
    """
    record, summary = result.record, result.summary
    lines = [
        f"Probe {record['run_id']}: status={record['status']}",
        f"  run dir: {result.run_dir}",
        f"  git_sha: {record['git_sha']}  git_dirty: {record['git_dirty']}",
        f"  config: {record['config_path']}  config_hash: {record['config_hash']}",
        f"  split: {record['split_name']}  split_sha256: {record['split_sha256']}",
        f"  manifest_sha256: {record['manifest_sha256']}",
        f"  subsets: {record['data_subsets']}",
        f"  predictions: {summary['inputs']['predictions_path']}  "
        f"sha256: {summary['inputs']['predictions_sha256']}",
        *_class_summary_lines(summary["class_summaries"]),
        *_auc_lines(summary["single_feature_aucs"]),
        *_classifier_lines(summary["classifiers"]),
        *_comparison_lines(summary["predictions_comparison"]),
    ]
    if record["git_dirty"]:
        lines.append("WARNING: git_dirty is true; this run cannot be cited (docs/RULES.md §3).")
    return "\n".join(lines)


def _class_summary_lines(summaries: Mapping[str, Any]) -> list[str]:
    lines = ["Header features per class (no pixels decoded):"]
    for split, classes in summaries.items():
        for class_name, description in classes.items():
            lines.append(f"  [{split} / {class_name}] n={description['n']}")
            for name, counts in description["counts"].items():
                values = ", ".join(f"{value}={count}" for value, count in counts.items())
                lines.append(f"    {name}: {values or '—'}")
            for name, stats in description["numeric"].items():
                lines.append(f"    {name}: {_quartile_text(stats)}")
    return lines


def _quartile_text(stats: Mapping[str, Any]) -> str:
    if stats["median"] is None:
        return f"n/a ({stats['na_reason']})"
    return (
        f"median {stats['median']:.6g}, IQR {stats['iqr']:.6g} "
        f"(Q1 {stats['q1']:.6g}, Q3 {stats['q3']:.6g}), non-null {stats['n_non_null']}"
    )


def _auc_lines(section: Mapping[str, Any]) -> list[str]:
    lines = [f"Single-feature {section['label']}:"]
    for rank, entry in enumerate(section["features"], start=1):
        lines.append(f"  {rank:2d}. {entry['feature']}: {_measured_text(entry)}")
    return lines


def _classifier_lines(section: Mapping[str, Any]) -> list[str]:
    lines = [
        f"Classifiers fitted on train, evaluated on val at threshold {section['threshold']} "
        f"({section['threshold_rule']}):"
    ]
    for role in ("primary", "secondary"):
        metrics = section[role]["pad_metrics"]
        lines.append(
            f"  {role} {section[role]['model']}: "
            f"APCER (pooled) {metrics['apcer']:.6f} = {metrics['n_attack_accepted']}/"
            f"{metrics['n_attack']}, BPCER {metrics['bpcer']:.6f} = "
            f"{metrics['n_bona_fide_rejected']}/{metrics['n_bona_fide']}, "
            f"ACER (pooled) {metrics['acer']:.6f}"
        )
    return lines


def _comparison_lines(comparison: Mapping[str, Any] | None) -> list[str]:
    if comparison is None:
        return ["CNN vs primary metadata classifier: skipped (no --predictions)."]
    lines = ["CNN vs primary metadata classifier, per class:"]
    for class_name, stats in comparison.items():
        lines.append(
            f"  {class_name}: n={stats['n']}, "
            f"Spearman rho {_measured_text(stats['spearman_rho'])}, "
            f"CNN errors {stats['cnn_errors']}, metadata errors {stats['metadata_errors']}, "
            f"both {stats['both_errors']} (expected if independent: "
            f"{_measured_text(stats['both_errors_expected_if_independent'])}), "
            "fraction of CNN errors that are also metadata errors: "
            f"{_measured_text(stats['cnn_errors_also_metadata_errors'])}"
        )
    return lines


def _measured_text(entry: Mapping[str, Any]) -> str:
    value = entry["value"]
    return f"n/a ({entry['na_reason']})" if value is None else f"{value:.6f}"


def _label_array(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(frame["label"].to_numpy(dtype=np.int64))


def _parse_features(section: object) -> FeatureTransformConfig:
    if not isinstance(section, dict) or set(section) != {"log1p"}:
        raise ProbeConfigError("Section 'features' must hold exactly the key 'log1p'.")
    values = section["log1p"]
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ProbeConfigError(f"features.log1p: expected a list of feature names, got {values!r}.")
    return FeatureTransformConfig(log1p=tuple(values))
