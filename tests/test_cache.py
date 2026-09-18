"""Tests for the normalized image cache: its contract, its builder and its verification.

Synthetic images under ``tmp_path`` only; nothing reads ``data/``.
"""

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from antispoof.data import cache, cache_build, labels
from antispoof.data.build import SplitOutputs, write_outputs
from antispoof.data.config import DataConfig
from antispoof.data.dataset import ImageLoadError
from antispoof.data.manifest import MANIFEST_COLUMNS, ManifestError, build_manifest
from antispoof.eval.jpeg_tables import read_jpeg_header, reference_tables

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_CONFIG = REPO_ROOT / "configs" / "cache_v1.yaml"

CONFLICT_EXCLUDE = "exclude"
TINY_SIZE = 32
PROGRESS_EVERY = 4
SUBSAMPLING_420 = 2
TARGET_QUALITY = 90

TRAIN_SUBJECTS = {"0001": (2, 2), "0002": (2, 2)}
VAL_SUBJECTS = {"0003": (2, 2)}
ROWS = {labels.SPLIT_TRAIN: 8, labels.SPLIT_VAL: 4}
SPLITS = (labels.SPLIT_TRAIN, labels.SPLIT_VAL)

SOURCE_SIZES = ((20, 24), (48, 40), (64, 64))
"""Source sizes, both smaller and larger than TINY_SIZE, so the resize runs in both directions."""


def _write_noise_images(manifest: pd.DataFrame, root: Path) -> None:
    """Seeded noise JPEGs of varying size: live at quality 75, spoof at 95, as the mirror has."""
    for position, (image_path, label) in enumerate(
        zip(manifest["image_path"], manifest["label"], strict=True)
    ):
        width, height = SOURCE_SIZES[position % len(SOURCE_SIZES)]
        pixels = np.random.default_rng(position).integers(
            0, 256, size=(height, width, 3), dtype=np.uint8
        )
        path = root / image_path
        path.parent.mkdir(parents=True, exist_ok=True)
        quality = 95 if label == labels.LABEL_SPOOF else 75
        Image.fromarray(pixels, mode="RGB").save(
            path, "JPEG", quality=quality, subsampling=SUBSAMPLING_420
        )


@pytest.fixture
def cache_data(
    tmp_data_config: DataConfig, make_labels: Callable[..., dict[str, list[int]]]
) -> DataConfig:
    """Train and val manifests, their noise JPEGs and a split assignment, under ``tmp_path``."""
    manifests: dict[str, pd.DataFrame] = {}
    assignment_rows: list[tuple[str, str]] = []
    for split, subjects in ((labels.SPLIT_TRAIN, TRAIN_SUBJECTS), (labels.SPLIT_VAL, VAL_SUBJECTS)):
        manifest, _ = build_manifest(
            make_labels(labels.SPLIT_TRAIN, subjects), labels.SPLIT_TRAIN, CONFLICT_EXCLUDE
        )
        manifest["split"] = split
        manifests[split] = manifest
        _write_noise_images(manifest, tmp_data_config.dataset_root)
        assignment_rows.extend((subject, split) for subject in subjects)
    assignment = pd.DataFrame(assignment_rows, columns=["subject_id", "split"])
    outputs = SplitOutputs(manifests=manifests, assignment=assignment)
    write_outputs(outputs, tmp_data_config.manifest_dir, tmp_data_config.split_assignment_path)
    return tmp_data_config


def _config(target_size: int = TINY_SIZE, **verify: Any) -> cache_build.CacheConfig:
    config = cache_build.load_cache_config(CACHE_CONFIG)
    return dataclasses.replace(
        config,
        cache=dataclasses.replace(config.cache, target_size=target_size),
        verify=dataclasses.replace(config.verify, **verify),
        progress=dataclasses.replace(config.progress, every=PROGRESS_EVERY),
    )


def _inputs(
    data_config: DataConfig, cache_dir: Path, limit: int | None = None, **overrides: Any
) -> cache_build.CacheInputs:
    return cache_build.CacheInputs(
        config=_config(**overrides),
        config_path=CACHE_CONFIG,
        data_config=data_config,
        splits=SPLITS,
        output_dir=cache_dir,
        repo_root=REPO_ROOT,
        limit=limit,
    )


def _digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_cached_files_have_the_target_quality_subsampling_and_size(
    cache_data: DataConfig, tmp_path: Path
) -> None:
    report = cache_build.build_cache(_inputs(cache_data, tmp_path / "cache"))
    expected = reference_tables(TARGET_QUALITY, SUBSAMPLING_420)
    checked = 0
    for split in SPLITS:
        frame = cache.read_cache_manifest(report.cache_dir / f"cache_manifest_{split}.csv")
        for relative in frame[cache.CACHED_PATH_COLUMN]:
            header = read_jpeg_header(report.cache_dir / relative)
            assert (header.width, header.height) == (TINY_SIZE, TINY_SIZE)
            assert header.tables == expected
            checked += 1
    assert checked == sum(ROWS.values())


def test_source_sizes_differ_before_caching_so_the_resize_is_exercised(
    cache_data: DataConfig,
) -> None:
    manifest = pd.read_csv(cache_data.manifest_dir / "manifest_train.csv")
    sizes = {
        read_jpeg_header(cache_data.dataset_root / path).width for path in manifest["image_path"]
    }
    assert len(sizes) > 1 and sizes != {TINY_SIZE}


def test_cache_manifest_holds_the_contract_columns_and_mirrors_the_source_layout(
    cache_data: DataConfig, tmp_path: Path
) -> None:
    report = cache_build.build_cache(_inputs(cache_data, tmp_path / "cache"))
    frame = cache.read_cache_manifest(report.cache_dir / "cache_manifest_val.csv")
    assert tuple(frame.columns) == cache.CACHE_MANIFEST_COLUMNS
    assert list(frame.columns)[: len(MANIFEST_COLUMNS)] == list(MANIFEST_COLUMNS)
    assert len(frame) == ROWS[labels.SPLIT_VAL]
    assert frame["subject_id"].tolist() == ["0003"] * ROWS[labels.SPLIT_VAL]
    for image_path, relative, digest in zip(
        frame["image_path"],
        frame[cache.CACHED_PATH_COLUMN],
        frame[cache.SOURCE_SHA256_COLUMN],
        strict=True,
    ):
        source = Path(image_path)
        assert relative == f"val/{source.parts[-3]}/{source.parts[-2]}/{source.name}"
        assert not Path(relative).is_absolute()
        assert (report.cache_dir / relative).is_file()
        assert (
            digest
            == hashlib.sha256((cache_data.dataset_root / image_path).read_bytes()).hexdigest()
        )


def test_building_twice_is_deterministic_and_the_second_build_skips_every_row(
    cache_data: DataConfig, tmp_path: Path
) -> None:
    first = cache_build.build_cache(_inputs(cache_data, tmp_path / "cache"))
    cached_before = _digests(first.cache_dir)
    second = cache_build.build_cache(_inputs(cache_data, tmp_path / "cache"))
    fresh = cache_build.build_cache(_inputs(cache_data, tmp_path / "again"))

    for split in SPLITS:
        assert first.summary["splits"][split]["cached"] == ROWS[split]
        assert first.summary["splits"][split]["skipped"] == 0
        assert second.summary["splits"][split]["skipped"] == ROWS[split]
        assert second.summary["splits"][split]["cached"] == 0
        assert second.summary["splits"][split]["bytes"] == 0
    images = {name: digest for name, digest in cached_before.items() if name.endswith(".jpg")}
    assert {
        name: digest for name, digest in _digests(fresh.cache_dir).items() if name.endswith(".jpg")
    } == images
    assert {
        name: digest for name, digest in _digests(second.cache_dir).items() if name.endswith(".jpg")
    } == images


def test_a_changed_source_is_rebuilt_while_the_others_are_skipped(
    cache_data: DataConfig, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    cache_build.build_cache(_inputs(cache_data, cache_dir))
    manifest = pd.read_csv(cache_data.manifest_dir / "manifest_val.csv")
    changed = cache_data.dataset_root / manifest["image_path"][0]
    Image.fromarray(
        np.random.default_rng(99).integers(0, 256, size=(30, 30, 3), dtype=np.uint8), mode="RGB"
    ).save(changed, "JPEG", quality=70, subsampling=SUBSAMPLING_420)

    report = cache_build.build_cache(_inputs(cache_data, cache_dir))
    val = report.summary["splits"][labels.SPLIT_VAL]
    assert (val["cached"], val["skipped"]) == (1, ROWS[labels.SPLIT_VAL] - 1)
    train = report.summary["splits"][labels.SPLIT_TRAIN]
    assert (train["cached"], train["skipped"]) == (0, ROWS[labels.SPLIT_TRAIN])
    frame = cache.read_cache_manifest(cache_dir / "cache_manifest_val.csv")
    assert frame[cache.SOURCE_SHA256_COLUMN][0] == hashlib.sha256(changed.read_bytes()).hexdigest()


def test_the_dataset_root_is_never_written(cache_data: DataConfig, tmp_path: Path) -> None:
    before = _digests(cache_data.dataset_root)
    cache_build.build_cache(_inputs(cache_data, tmp_path / "cache"))
    assert _digests(cache_data.dataset_root) == before
    assert before, "the fixture must have written source images"


@pytest.mark.parametrize(
    "image_paths",
    [
        pytest.param(
            ["Data/train/0001/live/000001.jpg", "Data/test/0001/live/000001.jpg"],
            id="different source splits",
        ),
        pytest.param(
            ["Data/train/0001/live/000001.jpg", "Data/train/0001/live/000001.jpg"],
            id="repeated image_path",
        ),
    ],
)
def test_check_no_collisions_reports_rows_that_map_to_one_cached_path(
    image_paths: list[str],
) -> None:
    frame = pd.DataFrame({"image_path": [*image_paths, "Data/train/0001/spoof/000001.jpg"]})
    with pytest.raises(cache_build.CacheCollisionError, match="cached-path collision"):
        cache_build.check_no_collisions(frame, labels.SPLIT_TRAIN)


def test_live_and_spoof_rows_with_one_basename_do_not_collide() -> None:
    frame = pd.DataFrame(
        {
            "image_path": [
                "Data/train/0001/live/000001.jpg",
                "Data/train/0001/spoof/000001.jpg",
            ]
        }
    )
    cache_build.check_no_collisions(frame, labels.SPLIT_TRAIN)
    assert cache.cached_path("Data/train/0001/live/000001.jpg", "train") == (
        "train/0001/live/000001.jpg"
    )
    assert cache.cached_path("Data/test/0004/spoof/9.jpg", "test") == "test/0004/spoof/9.jpg"


def test_an_unreadable_source_is_recorded_as_a_failure_without_stopping_the_build(
    cache_data: DataConfig, tmp_path: Path
) -> None:
    manifest = pd.read_csv(cache_data.manifest_dir / "manifest_val.csv")
    missing = cache_data.dataset_root / manifest["image_path"][0]
    broken = cache_data.dataset_root / manifest["image_path"][1]
    missing.unlink()
    broken.write_bytes(b"not an image")

    report = cache_build.build_cache(_inputs(cache_data, tmp_path / "cache"))
    val = report.summary["splits"][labels.SPLIT_VAL]
    assert val["failed"] == 2
    assert val["cached"] == ROWS[labels.SPLIT_VAL] - 2
    listed = {entry["image_path"] for entry in report.summary["failures"]}
    assert listed == {manifest["image_path"][0], manifest["image_path"][1]}
    assert all(entry["split"] == labels.SPLIT_VAL for entry in report.summary["failures"])
    frame = cache.read_cache_manifest(report.cache_dir / "cache_manifest_val.csv")
    assert len(frame) == ROWS[labels.SPLIT_VAL] - 2
    assert "Unreadable source images: 2" in cache_build.format_report(report)


def test_verification_rejects_a_cached_file_that_is_not_the_target(
    cache_data: DataConfig, tmp_path: Path
) -> None:
    report = cache_build.build_cache(_inputs(cache_data, tmp_path / "cache"))
    frame = cache.read_cache_manifest(report.cache_dir / "cache_manifest_val.csv")
    relatives = list(frame[cache.CACHED_PATH_COLUMN])
    settings = cache_build.resolve_settings(_config().cache)
    verify = cache_build.VerifyConfig(sample_rows=len(relatives), seed=1)
    assert cache_build.verify_cache_sample(relatives, report.cache_dir, settings, verify) == {
        "sample_rows": len(relatives),
        "checked": len(relatives),
        "seed": 1,
    }

    wrong_quality = report.cache_dir / relatives[0]
    with Image.open(wrong_quality) as image:
        image.convert("RGB").save(wrong_quality, "JPEG", quality=60, subsampling=SUBSAMPLING_420)
    with pytest.raises(cache_build.CacheVerificationError, match="luma mean"):
        cache_build.verify_cache_sample(relatives, report.cache_dir, settings, verify)


def test_verification_rejects_a_cached_file_of_the_wrong_size(
    cache_data: DataConfig, tmp_path: Path
) -> None:
    report = cache_build.build_cache(_inputs(cache_data, tmp_path / "cache"))
    frame = cache.read_cache_manifest(report.cache_dir / "cache_manifest_val.csv")
    relatives = list(frame[cache.CACHED_PATH_COLUMN])
    settings = cache_build.resolve_settings(_config().cache)
    target = report.cache_dir / relatives[0]
    with Image.open(target) as image:
        image.convert("RGB").resize((TINY_SIZE // 2, TINY_SIZE)).save(
            target, "JPEG", quality=TARGET_QUALITY, subsampling=SUBSAMPLING_420
        )
    verify = cache_build.VerifyConfig(sample_rows=len(relatives), seed=1)
    with pytest.raises(cache_build.CacheVerificationError, match=f"not {TINY_SIZE}x{TINY_SIZE}"):
        cache_build.verify_cache_sample(relatives, report.cache_dir, settings, verify)


def test_limit_caches_only_the_first_rows_of_each_split(
    cache_data: DataConfig, tmp_path: Path
) -> None:
    report = cache_build.build_cache(_inputs(cache_data, tmp_path / "cache", limit=2))
    assert report.summary["limit"] == 2
    for split in SPLITS:
        assert report.summary["splits"][split]["rows"] == 2
        assert len(cache.read_cache_manifest(report.cache_dir / f"cache_manifest_{split}.csv")) == 2


def test_cache_summary_records_the_settings_layout_and_source_manifest_hashes(
    cache_data: DataConfig, tmp_path: Path
) -> None:
    report = cache_build.build_cache(_inputs(cache_data, tmp_path / "cache"))
    summary = json.loads(
        (report.cache_dir / cache.CACHE_SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    assert summary == report.summary
    assert summary["name"] == "cache_v1"
    assert summary["config_path"] == "configs/cache_v1.yaml"
    assert summary["settings"]["quality"] == TARGET_QUALITY
    assert summary["settings"]["subsampling"] == "4:2:0"
    assert summary["settings"]["resample_filter"] == "bicubic"
    assert summary["settings"]["face_crop"] is False
    assert summary["settings"]["target_size"] == TINY_SIZE
    assert "mirroring the source image_path" in summary["layout"]
    assert summary["source_manifest_sha256"] == {
        f"manifest_{split}.csv": hashlib.sha256(
            (cache_data.manifest_dir / f"manifest_{split}.csv").read_bytes()
        ).hexdigest()
        for split in SPLITS
    }
    for split in SPLITS:
        stats = summary["splits"][split]
        assert stats["verification"]["checked"] == ROWS[split]
        assert stats["cache_manifest"] == f"cache_manifest_{split}.csv"
        assert (
            stats["cache_manifest_sha256"]
            == hashlib.sha256(
                (report.cache_dir / f"cache_manifest_{split}.csv").read_bytes()
            ).hexdigest()
        )
        assert stats["bytes"] > 0 and stats["non_jpeg_suffix"] == 0
    assert "cache_v1" in cache_build.format_report(report)


def test_read_cached_rows_tolerates_a_truncated_manifest(
    cache_data: DataConfig, tmp_path: Path
) -> None:
    report = cache_build.build_cache(_inputs(cache_data, tmp_path / "cache"))
    path = report.cache_dir / "cache_manifest_val.csv"
    complete = cache.read_cached_rows(path)
    assert len(complete) == ROWS[labels.SPLIT_VAL]

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:-1]) + "Data/train/0003/live/00", encoding="utf-8")
    partial = cache.read_cached_rows(path)
    assert len(partial) == ROWS[labels.SPLIT_VAL] - 1
    assert set(partial) < set(complete)
    assert cache.read_cached_rows(tmp_path / "absent.csv") == {}


def test_read_cache_manifest_rejects_a_wrong_header(tmp_path: Path) -> None:
    path = tmp_path / "cache_manifest_train.csv"
    path.write_text("image_path,cached_path\na,b\n", encoding="utf-8")
    with pytest.raises(cache.CacheError, match="expected columns"):
        cache.read_cache_manifest(path)
    with pytest.raises(cache.CacheError, match="not found"):
        cache.read_cache_manifest(tmp_path / "absent.csv")


def test_read_cache_summary_rejects_a_directory_without_one(tmp_path: Path) -> None:
    with pytest.raises(cache.CacheError, match="cache_summary.json missing"):
        cache.read_cache_summary(tmp_path)


def test_cached_path_rejects_a_path_outside_the_dataset_layout() -> None:
    with pytest.raises(ManifestError):
        cache.cached_path("000001.jpg", labels.SPLIT_TRAIN)


def test_committed_cache_config_has_the_agreed_values() -> None:
    config = cache_build.load_cache_config(CACHE_CONFIG)
    assert config.cache == cache_build.CacheSettings(
        name="cache_v1",
        target_size=256,
        quality=90,
        subsampling="4:2:0",
        resample_filter="bicubic",
        face_crop=False,
    )
    assert config.verify == cache_build.VerifyConfig(sample_rows=200, seed=42)
    assert config.progress == cache_build.ProgressConfig(every=50000)
    assert cache_build.resolve_settings(config.cache).subsampling_code == SUBSAMPLING_420


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("cache", "target_size", 0, "cache.target_size must be > 0"),
        ("cache", "quality", 0, r"cache.quality must be in \[1, 100\]"),
        ("cache", "quality", 101, r"cache.quality must be in \[1, 100\]"),
        ("cache", "subsampling", "4:1:1", "cache.subsampling must be one of"),
        ("cache", "resample_filter", "sinc", "cache.resample_filter must be one of"),
        ("cache", "face_crop", True, "cache.face_crop must be false"),
        ("cache", "name", " ", "cache.name must not be empty"),
        ("cache", "target_size", "256", "cache.target_size: expected int"),
        ("verify", "sample_rows", -1, "verify.sample_rows must be >= 0"),
        ("progress", "every", 0, "progress.every must be >= 1"),
    ],
)
def test_invalid_cache_configs_are_rejected(
    tmp_path: Path, section: str, key: str, value: object, message: str
) -> None:
    document = yaml.safe_load(CACHE_CONFIG.read_text(encoding="utf-8"))
    document[section][key] = value
    path = tmp_path / "cache.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(cache_build.CacheConfigError, match=message):
        cache_build.load_cache_config(path)


def test_unknown_cache_section_or_key_is_rejected(tmp_path: Path) -> None:
    document = yaml.safe_load(CACHE_CONFIG.read_text(encoding="utf-8"))
    document["cache"]["sharpen"] = True
    path = tmp_path / "cache.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(cache_build.CacheConfigError, match=r"unknown keys \['sharpen'\]"):
        cache_build.load_cache_config(path)
    document = yaml.safe_load(CACHE_CONFIG.read_text(encoding="utf-8"))
    document["sharpen"] = {}
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(cache_build.CacheConfigError, match="top-level sections"):
        cache_build.load_cache_config(path)


def _subsets(data_config: DataConfig) -> dict[str, pd.DataFrame]:
    return {
        split: pd.read_csv(
            data_config.manifest_dir / f"manifest_{split}.csv", dtype={"subject_id": str}
        )
        for split in SPLITS
    }


def _manifest_sha256(data_config: DataConfig) -> dict[str, str]:
    return {
        f"manifest_{split}.csv": hashlib.sha256(
            (data_config.manifest_dir / f"manifest_{split}.csv").read_bytes()
        ).hexdigest()
        for split in SPLITS
    }


def test_load_cache_binds_a_matching_cache(cache_data: DataConfig, tmp_path: Path) -> None:
    report = cache_build.build_cache(_inputs(cache_data, tmp_path / "cache"))
    subsets = _subsets(cache_data)
    binding = cache.load_cache(report.cache_dir, "cache_v1", subsets, _manifest_sha256(cache_data))
    assert binding.name == "cache_v1"
    assert binding.settings["quality"] == TARGET_QUALITY
    assert set(binding.cached_paths) == set(SPLITS)
    for split in SPLITS:
        assert len(binding.cached_paths[split]) == ROWS[split]
        assert binding.cached_paths[split] == [
            cache.cached_path(str(image_path), split) for image_path in subsets[split]["image_path"]
        ]
    entry = binding.record_entry()
    assert entry["name"] == "cache_v1"
    assert entry["dir"] == report.cache_dir.as_posix()
    assert entry["settings"] == report.summary["settings"]
    assert set(entry["cache_manifest_sha256"]) == {
        f"cache_manifest_{split}.csv" for split in SPLITS
    }
    assert (
        entry["summary_sha256"]
        == hashlib.sha256(
            (report.cache_dir / cache.CACHE_SUMMARY_FILENAME).read_bytes()
        ).hexdigest()
    )


def test_load_cache_rejects_another_cache_name(cache_data: DataConfig, tmp_path: Path) -> None:
    report = cache_build.build_cache(_inputs(cache_data, tmp_path / "cache"))
    with pytest.raises(cache.CacheMismatchError, match="config asks for 'cache_v2'"):
        cache.load_cache(
            report.cache_dir, "cache_v2", _subsets(cache_data), _manifest_sha256(cache_data)
        )


def test_load_cache_rejects_a_cache_built_from_other_manifests(
    cache_data: DataConfig, tmp_path: Path
) -> None:
    report = cache_build.build_cache(_inputs(cache_data, tmp_path / "cache"))
    subsets = _subsets(cache_data)
    path = cache_data.manifest_dir / "manifest_val.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(cache.CacheMismatchError, match="but this run reads"):
        cache.load_cache(report.cache_dir, "cache_v1", subsets, _manifest_sha256(cache_data))


def test_load_cache_rejects_a_cache_that_misses_rows(
    cache_data: DataConfig, tmp_path: Path
) -> None:
    report = cache_build.build_cache(_inputs(cache_data, tmp_path / "cache", limit=2))
    with pytest.raises(cache.CacheMismatchError, match="rows this run needs"):
        cache.load_cache(
            report.cache_dir, "cache_v1", _subsets(cache_data), _manifest_sha256(cache_data)
        )


def test_load_cache_rejects_a_directory_that_is_not_a_cache(
    cache_data: DataConfig, tmp_path: Path
) -> None:
    with pytest.raises(cache.CacheError, match="cache_summary.json missing"):
        cache.load_cache(
            tmp_path / "empty", "cache_v1", _subsets(cache_data), _manifest_sha256(cache_data)
        )


def test_cached_dataset_reads_the_cache_and_keeps_labels_and_row_indices(
    cache_data: DataConfig, tmp_path: Path
) -> None:
    import timm

    from antispoof.data.transforms import build_baseline_transform

    report = cache_build.build_cache(_inputs(cache_data, tmp_path / "cache"))
    subsets = _subsets(cache_data)
    binding = cache.load_cache(report.cache_dir, "cache_v1", subsets, _manifest_sha256(cache_data))
    val = subsets[labels.SPLIT_VAL]
    transform = build_baseline_transform(
        16, timm.create_model("test_efficientnet", pretrained=False, num_classes=1)
    )
    dataset = cache.CachedManifestDataset(
        val, report.cache_dir, transform, binding.cached_paths[labels.SPLIT_VAL]
    )
    assert len(dataset) == ROWS[labels.SPLIT_VAL]
    for index in range(len(dataset)):
        image, label, row_index = dataset[index]
        assert image.shape == (3, 16, 16)
        assert (label, row_index) == (float(val.loc[index, "label"]), index)
        assert dataset.load_image(index).size == (TINY_SIZE, TINY_SIZE)

    # The pixels come from the cache, not the dataset root.
    for image_path in val["image_path"]:
        (cache_data.dataset_root / image_path).unlink()
    assert dataset[0][0].shape == (3, 16, 16)

    missing = report.cache_dir / binding.cached_paths[labels.SPLIT_VAL][1]
    missing.unlink()
    with pytest.raises(ImageLoadError, match=re.escape(str(missing))):
        dataset[1]


def test_cached_dataset_needs_one_path_per_row(cache_data: DataConfig, tmp_path: Path) -> None:
    val = _subsets(cache_data)[labels.SPLIT_VAL]
    with pytest.raises(ValueError, match="cached_paths has 1 entries for 4 manifest rows"):
        cache.CachedManifestDataset(val, tmp_path, lambda image: image, ["val/a/live/b.jpg"])  # type: ignore[arg-type]
