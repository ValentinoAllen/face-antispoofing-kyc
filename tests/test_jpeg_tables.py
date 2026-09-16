"""Tests for JPEG quantization-table reading, quality matching, re-encoding and the audit.

All images are synthetic and written under ``tmp_path``.
"""

import io
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageFile

from antispoof.data import labels
from antispoof.data.dataset import ImageLoadError
from antispoof.eval import jpeg_tables

LIVE_QUALITY = 75
SPOOF_QUALITY = 95
SUBSAMPLING_420 = 2
CUSTOM_TABLE_VALUE = 7


def noise_image(size: tuple[int, int], seed: int) -> Image.Image:
    """A seeded RGB noise image, so that re-encoding changes its pixels."""
    width, height = size
    pixels = np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def write_jpeg(path: Path, size: tuple[int, int], seed: int, **options: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    noise_image(size, seed).save(path, "JPEG", **options)
    return path


@pytest.fixture
def spy_on_file_loads(monkeypatch: pytest.MonkeyPatch) -> Callable[[], list[str]]:
    """Return a function that starts recording pixel loads of images opened from files or bytes."""

    def install() -> list[str]:
        calls: list[str] = []
        for owner in (Image.Image, ImageFile.ImageFile):
            original = owner.load

            def spy(self: Image.Image, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
                if isinstance(self, ImageFile.ImageFile):
                    calls.append(type(self).__name__)
                return _original(self, *args, **kwargs)

            monkeypatch.setattr(owner, "load", spy)
        return calls

    return install


@pytest.mark.parametrize("subsampling", [0, 1, 2])
@pytest.mark.parametrize("quality", [5, 50, LIVE_QUALITY, SPOOF_QUALITY, 100])
def test_matcher_recovers_the_quality_of_standard_encodings(
    tmp_path: Path, quality: int, subsampling: int
) -> None:
    path = write_jpeg(tmp_path / "image.jpg", (40, 30), 0, quality=quality, subsampling=subsampling)
    header = jpeg_tables.read_jpeg_header(path)
    assert (header.width, header.height) == (40, 30)
    assert header.tables.subsampling == subsampling
    assert jpeg_tables.matching_qualities(header.tables) == (quality,)


def test_custom_table_has_no_standard_match(tmp_path: Path) -> None:
    table = [CUSTOM_TABLE_VALUE] * 64
    path = write_jpeg(tmp_path / "custom.jpg", (40, 30), 0, qtables=[table, table], subsampling=2)
    tables = jpeg_tables.read_jpeg_header(path).tables
    assert tables.luma == tuple(table)
    assert jpeg_tables.matching_qualities(tables) == ()


def test_reference_tables_do_not_depend_on_image_size_or_content(tmp_path: Path) -> None:
    path = write_jpeg(tmp_path / "large.jpg", (450, 600), 3, quality=SPOOF_QUALITY, subsampling=2)
    tables = jpeg_tables.read_jpeg_header(path).tables
    assert tables == jpeg_tables.reference_tables(SPOOF_QUALITY, SUBSAMPLING_420)


def test_unreadable_or_non_jpeg_headers_raise_naming_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jpg"
    with pytest.raises(ImageLoadError, match=re.escape(str(missing))):
        jpeg_tables.read_jpeg_header(missing)
    png = tmp_path / "image.png"
    noise_image((8, 8), 0).save(png)
    with pytest.raises(ImageLoadError, match=re.escape(str(png))):
        jpeg_tables.read_jpeg_header(png)


def _audit_frames(root: Path, extra_spoof_quality: int | None) -> dict[str, pd.DataFrame]:
    """Train and val subsets: live at LIVE_QUALITY, spoof at SPOOF_QUALITY, one optional outlier."""
    frames = {}
    for split, n_live, n_spoof in ((labels.SPLIT_TRAIN, 3, 2), (labels.SPLIT_VAL, 2, 3)):
        rows = []
        for index in range(n_live + n_spoof):
            label = labels.LABEL_LIVE if index < n_live else labels.LABEL_SPOOF
            quality = LIVE_QUALITY if label == labels.LABEL_LIVE else SPOOF_QUALITY
            if split == labels.SPLIT_VAL and index == n_live and extra_spoof_quality is not None:
                quality = extra_spoof_quality
            image_path = f"Data/{split}/{index:06d}.jpg"
            write_jpeg(root / image_path, (20 + index, 30), index, quality=quality, subsampling=2)
            rows.append({"image_path": image_path, "split": split, "label": label})
        frames[split] = pd.DataFrame(rows)
    return frames


def test_audit_reads_headers_only_and_finds_one_quality_per_class(
    tmp_path: Path, spy_on_file_loads: Callable[[], list[str]]
) -> None:
    frames = _audit_frames(tmp_path, extra_spoof_quality=None)
    calls = spy_on_file_loads()
    headers = jpeg_tables.read_subset_headers(frames, tmp_path)
    audit = jpeg_tables.audit_quantization(headers)
    encodings = jpeg_tables.class_encodings(audit)
    assert calls == []
    with Image.open(tmp_path / frames[labels.SPLIT_TRAIN]["image_path"][0]) as image:
        image.load()
    assert calls and set(calls) == {"JpegImageFile"}, "the spy must record a real pixel load"

    assert list(headers.columns) == list(jpeg_tables.HEADER_COLUMNS)
    assert headers["width"].tolist() == [20, 21, 22, 23, 24, 20, 21, 22, 23, 24]
    live, spoof = audit["live"], audit["spoof"]
    assert live["rows"] == {"train": 3, "val": 2} and spoof["rows"] == {"train": 2, "val": 3}
    assert live["n_distinct_table_sets"] == spoof["n_distinct_table_sets"] == 1
    (live_set,) = live["table_sets"]
    assert (live_set["match"], live_set["matching_qualities"]) == ("q=75", [LIVE_QUALITY])
    assert (live_set["mode"], live_set["subsampling"]) == ("RGB", "4:2:0")
    reference = jpeg_tables.reference_tables(LIVE_QUALITY, SUBSAMPLING_420)
    assert live_set["luma_mean"] == float(np.mean(reference.luma))
    assert live_set["tables"] == [list(table) for table in reference.tables]
    assert encodings == {
        "live": jpeg_tables.ClassEncoding(LIVE_QUALITY, SUBSAMPLING_420),
        "spoof": jpeg_tables.ClassEncoding(SPOOF_QUALITY, SUBSAMPLING_420),
    }


def test_two_qualities_in_a_class_fail_the_audit(tmp_path: Path) -> None:
    frames = _audit_frames(tmp_path, extra_spoof_quality=90)
    audit = jpeg_tables.audit_quantization(jpeg_tables.read_subset_headers(frames, tmp_path))
    spoof = audit["spoof"]
    assert spoof["n_distinct_table_sets"] == 2
    assert [entry["rows"] for entry in spoof["table_sets"]] == [
        {"train": 2, "val": 2},
        {"train": 0, "val": 1},
    ]
    assert spoof["table_sets"][1]["match"] == "q=90"
    with pytest.raises(jpeg_tables.QualityAuditError, match="spoof has 2 distinct table sets"):
        jpeg_tables.class_encodings(audit)


def test_a_table_set_without_a_standard_match_fails_the_audit(tmp_path: Path) -> None:
    frames = _audit_frames(tmp_path, extra_spoof_quality=None)
    table = [CUSTOM_TABLE_VALUE] * 64
    for image_path in frames[labels.SPLIT_TRAIN]["image_path"][:3]:
        write_jpeg(tmp_path / image_path, (20, 30), 0, qtables=[table, table], subsampling=2)
    frames[labels.SPLIT_VAL] = frames[labels.SPLIT_VAL].iloc[2:]
    audit = jpeg_tables.audit_quantization(jpeg_tables.read_subset_headers(frames, tmp_path))
    (live_set,) = audit["live"]["table_sets"]
    assert live_set["match"] == jpeg_tables.NO_STANDARD_MATCH
    with pytest.raises(jpeg_tables.QualityAuditError, match="live table set matches no standard"):
        jpeg_tables.class_encodings(audit)


def test_reencoding_produces_the_target_table_and_size_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded: list[bytes] = []
    original = jpeg_tables.encode_jpeg

    def capture(image: Image.Image, quality: int, subsampling: int) -> bytes:
        data = original(image, quality, subsampling)
        encoded.append(data)
        return data

    image = noise_image((30, 40), 1)
    reencoding = jpeg_tables.Reencoding(LIVE_QUALITY, SUBSAMPLING_420, (24, 32), "bicubic")
    monkeypatch.setattr(jpeg_tables, "encode_jpeg", capture)
    first = jpeg_tables.reencode_image(image, reencoding)
    second = jpeg_tables.reencode_image(image, reencoding)

    assert first.size == (24, 32) and first.mode == "RGB"
    assert first.tobytes() == second.tobytes()
    assert encoded[-2] == encoded[-1]
    with Image.open(io.BytesIO(encoded[-1])) as reopened:
        tables = jpeg_tables.jpeg_tables(reopened, "captured")
    assert jpeg_tables.matching_qualities(tables) == (LIVE_QUALITY,)
    assert image.size == (30, 40)


def test_reencoding_raises_when_the_luma_table_is_not_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = jpeg_tables.reference_tables
    monkeypatch.setattr(
        jpeg_tables, "reference_tables", lambda quality, sub: original(quality + 1, sub)
    )
    reencoding = jpeg_tables.Reencoding(LIVE_QUALITY, SUBSAMPLING_420)
    with pytest.raises(jpeg_tables.ReencodeError, match="not the table of quality 75"):
        jpeg_tables.reencode_image(noise_image((16, 16), 0), reencoding)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((0, 2), "quality must be in"),
        ((101, 2), "quality must be in"),
        ((75, -1), "subsampling must be one of"),
        ((75, 2, (10, 10), None), "set together"),
        ((75, 2, None, "bicubic"), "set together"),
        ((75, 2, (10, 10), "sharpest"), "resample must be one of"),
        ((75, 2, (0, 10), "bicubic"), "size must be positive"),
    ],
)
def test_invalid_reencodings_are_rejected(arguments: tuple[Any, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        jpeg_tables.Reencoding(*arguments)
