"""JPEG quantization tables: read them from headers, match them to Pillow quality levels, re-encode.

Part 1 of the counterfactual evaluation (:mod:`antispoof.eval.counterfactual`) audits the tables of
the baseline's train and val subsets **from headers only**: no dataset pixel is decoded. Each
distinct table set is matched to the Pillow ``quality`` levels that reproduce it exactly, found by
encoding a tiny blank image in memory at every quality and reading its tables back.

:func:`reencode_image` re-encodes a decoded image in memory at a given quality and checks, by
reading the new header back, that the luma table is the one that quality produces.
"""

import functools
import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin

from antispoof.data.dataset import ImageLoadError
from antispoof.eval.metadata_probe import CLASS_NAMES, SUBSAMPLING_NAMES, SUBSAMPLING_OTHER

MIN_QUALITY = 1
MAX_QUALITY = 100
"""Pillow's JPEG ``quality`` scale (libjpeg ``jpeg_set_quality``). Every level is tried."""

REFERENCE_MODE = "RGB"
"""The only image mode the matcher encodes reference images in."""

_REFERENCE_SIDE = 16
"""Side of the blank reference image. Not a setting: libjpeg scales its standard tables by quality
alone, so the tables do not depend on image size or content (checked in the tests)."""

NO_STANDARD_MATCH = "no standard match"
CLASS_NAME = dict(CLASS_NAMES)
"""Class name of each label value: ``{0: "live", 1: "spoof"}``."""

RESAMPLE_FILTERS = tuple(member.name.lower() for member in Image.Resampling)
"""Names accepted as a resize filter, e.g. ``bicubic``."""

HEADER_COLUMNS = ("image_path", "split", "label", "width", "height", "tables")
"""Columns of :func:`read_subset_headers`. ``tables`` holds :class:`JpegTables` objects."""


class ReencodeError(RuntimeError):
    """Raised when a re-encoded image does not carry the luma table of its target quality."""


class QualityAuditError(RuntimeError):
    """Raised when the audit does not find exactly one standard quality per class."""


@dataclass(frozen=True)
class JpegTables:
    """The quantization tables of a JPEG, as Pillow reads them from the header.

    Attributes:
        mode: Pillow image mode, e.g. ``RGB``.
        subsampling: ``JpegImagePlugin.get_sampling`` code (``0`` 4:4:4, ``1`` 4:2:2, ``2``
            4:2:0, ``-1`` other).
        tables: The table each component uses, in component order (Y, Cb, Cr for RGB).
    """

    mode: str
    subsampling: int
    tables: tuple[tuple[int, ...], ...]

    @property
    def luma(self) -> tuple[int, ...]:
        """The table of the first (luminance) component."""
        return self.tables[0]

    @property
    def chroma(self) -> tuple[int, ...] | None:
        """The table of the second (chrominance) component, or ``None`` for one component."""
        return self.tables[1] if len(self.tables) > 1 else None


@dataclass(frozen=True)
class JpegHeader:
    """Image size and quantization tables read from a JPEG header."""

    width: int
    height: int
    tables: JpegTables


@dataclass(frozen=True)
class Reencoding:
    """How to re-encode one decoded image.

    Attributes:
        quality: Pillow JPEG quality in ``[MIN_QUALITY, MAX_QUALITY]``.
        subsampling: Subsampling code in ``SUBSAMPLING_NAMES``.
        size: ``(width, height)`` to resize to before encoding, or ``None`` to keep the size.
        resample: Resize filter name in ``RESAMPLE_FILTERS``; set exactly when ``size`` is.
    """

    quality: int
    subsampling: int
    size: tuple[int, int] | None = None
    resample: str | None = None

    def __post_init__(self) -> None:
        """Check the quality, subsampling, size and filter.

        Raises:
            ValueError: If a value is out of range, or only one of ``size`` and ``resample`` is set.
        """
        if not MIN_QUALITY <= self.quality <= MAX_QUALITY:
            raise ValueError(f"quality must be in [{MIN_QUALITY}, {MAX_QUALITY}]: {self.quality}.")
        if self.subsampling not in SUBSAMPLING_NAMES:
            raise ValueError(f"subsampling must be one of {sorted(SUBSAMPLING_NAMES)}.")
        if (self.size is None) != (self.resample is None):
            raise ValueError("size and resample must be set together.")
        if self.resample is not None and self.resample not in RESAMPLE_FILTERS:
            raise ValueError(f"resample must be one of {RESAMPLE_FILTERS}: {self.resample!r}.")
        if self.size is not None and min(self.size) < 1:
            raise ValueError(f"size must be positive: {self.size}.")


@dataclass(frozen=True)
class ClassEncoding:
    """The single standard encoding found for one class."""

    quality: int
    subsampling: int


def read_jpeg_header(path: Path) -> JpegHeader:
    """Read the size and quantization tables of a JPEG file without decoding any pixel data.

    Args:
        path: Image file path.

    Returns:
        The header.

    Raises:
        ImageLoadError: If the file does not exist, is not a JPEG, or its header cannot be parsed.
            The message names the path.
    """
    if not path.is_file():
        raise ImageLoadError(f"Image file not found: {path}")
    try:
        with Image.open(path) as image:
            width, height = image.size
            return JpegHeader(width, height, jpeg_tables(image, str(path)))
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        raise ImageLoadError(f"Cannot read image header {path}: {error}") from error


def jpeg_tables(image: Image.Image, source: str) -> JpegTables:
    """Collect the quantization table of each component of an opened JPEG.

    Args:
        image: An image opened with ``PIL.Image.open``; only header attributes are read.
        source: Names the image in error messages.

    Returns:
        The tables.

    Raises:
        ImageLoadError: If the image is not a JPEG or a component references a missing table.
    """
    if not isinstance(image, JpegImagePlugin.JpegImageFile):
        raise ImageLoadError(f"{source} is not a JPEG (format {image.format}).")
    # layer holds (component id, h sampling, v sampling, quantization table id) per component.
    try:
        tables = tuple(
            tuple(int(value) for value in image.quantization[component[3]])
            for component in image.layer
        )
    except KeyError as error:
        raise ImageLoadError(f"{source}: missing quantization table {error}.") from error
    if not tables:
        raise ImageLoadError(f"{source} declares no image components.")
    subsampling = JpegImagePlugin.get_sampling(image)
    return JpegTables(mode=image.mode, subsampling=subsampling, tables=tables)


def encode_jpeg(image: Image.Image, quality: int, subsampling: int) -> bytes:
    """Encode an image as JPEG in memory with Pillow.

    Args:
        image: The image.
        quality: Pillow JPEG quality.
        subsampling: Subsampling code.

    Returns:
        The JPEG bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=subsampling)
    return buffer.getvalue()


@functools.cache
def reference_tables(quality: int, subsampling: int) -> JpegTables:
    """Return the tables Pillow writes at ``quality``, read back from a tiny blank image.

    Args:
        quality: Pillow JPEG quality in ``[MIN_QUALITY, MAX_QUALITY]``.
        subsampling: Subsampling code in ``SUBSAMPLING_NAMES``.

    Returns:
        The tables of an ``RGB`` image encoded in memory at that quality and subsampling.

    Raises:
        ValueError: If the quality or subsampling is out of range.
    """
    Reencoding(quality, subsampling)  # Raises ValueError for an out-of-range argument.
    blank = Image.new(REFERENCE_MODE, (_REFERENCE_SIDE, _REFERENCE_SIDE))
    with Image.open(io.BytesIO(encode_jpeg(blank, quality, subsampling))) as encoded:
        return jpeg_tables(encoded, f"reference JPEG at quality {quality}")


def matching_qualities(tables: JpegTables) -> tuple[int, ...]:
    """Find every Pillow quality whose tables equal ``tables`` exactly, at the same subsampling.

    Args:
        tables: Tables read from a header.

    Returns:
        The matching qualities in increasing order; empty ("no standard match") for a custom table,
        or when the mode is not ``RGB`` or the subsampling is not a standard code.
    """
    if tables.mode != REFERENCE_MODE or tables.subsampling not in SUBSAMPLING_NAMES:
        return ()
    return tuple(
        quality
        for quality in range(MIN_QUALITY, MAX_QUALITY + 1)
        if reference_tables(quality, tables.subsampling) == tables
    )


def reencode_image(image: Image.Image, reencoding: Reencoding) -> Image.Image:
    """Resize (optionally) and re-encode a decoded image in memory, then decode the result.

    After encoding, the header is read back and its luma table compared with
    :func:`reference_tables` for the target quality.

    Args:
        image: A decoded RGB image. It is not modified.
        reencoding: Target quality, subsampling and optional size.

    Returns:
        The decoded re-encoded image, in RGB.

    Raises:
        ReencodeError: If the re-encoded luma table differs from the target quality's.
    """
    if reencoding.size is not None and reencoding.resample is not None:
        resample = Image.Resampling[reencoding.resample.upper()]
        image = image.resize(reencoding.size, resample=resample)
    data = encode_jpeg(image, reencoding.quality, reencoding.subsampling)
    expected = reference_tables(reencoding.quality, reencoding.subsampling).luma
    with Image.open(io.BytesIO(data)) as encoded:
        found = jpeg_tables(encoded, "re-encoded image").luma
        if found != expected:
            raise ReencodeError(
                f"Re-encoded luma table (mean {np.mean(found)}) is not the table of quality "
                f"{reencoding.quality} (mean {np.mean(expected)})."
            )
        return encoded.convert("RGB")


def read_subset_headers(subsets: Mapping[str, pd.DataFrame], dataset_root: Path) -> pd.DataFrame:
    """Read the JPEG header of every row of the subsets. No pixel data is decoded.

    Args:
        subsets: Frames with ``image_path``, ``split`` and ``label``, e.g. ``{"train", "val"}``.
        dataset_root: Directory the ``image_path`` values are relative to.

    Returns:
        ``HEADER_COLUMNS``, one row per subset row, subsets concatenated in mapping order.

    Raises:
        ImageLoadError: If a header cannot be read or a file is not a JPEG.
    """
    rows: list[dict[str, Any]] = []
    for frame in subsets.values():
        for image_path, split, label in zip(
            frame["image_path"], frame["split"], frame["label"], strict=True
        ):
            header = read_jpeg_header(dataset_root / image_path)
            rows.append(
                {
                    "image_path": image_path,
                    "split": split,
                    "label": int(label),
                    "width": header.width,
                    "height": header.height,
                    "tables": header.tables,
                }
            )
    return pd.DataFrame(rows, columns=list(HEADER_COLUMNS))


def audit_quantization(headers: pd.DataFrame) -> dict[str, Any]:
    """Describe the distinct quantization table sets of each class and their matching qualities.

    Args:
        headers: From :func:`read_subset_headers`.

    Returns:
        ``{class: {"rows", "n_distinct_table_sets", "table_sets"}}``. ``rows`` counts rows per
        split. Each table set has its mode, subsampling, luma and chroma means, the tables, rows per
        split, ``matching_qualities`` and ``match`` (``q=<quality>`` or ``no standard match``).
        Table sets are ordered by row count, largest first.
    """
    splits = sorted(set(headers["split"]))
    audit: dict[str, Any] = {}
    for value, name in CLASS_NAMES:
        rows = headers[headers["label"] == value]
        counts: dict[JpegTables, dict[str, int]] = {}
        for tables, split in zip(rows["tables"], rows["split"], strict=True):
            per_split = counts.setdefault(tables, dict.fromkeys(splits, 0))
            per_split[split] += 1
        ordered = sorted(counts.items(), key=lambda item: (-sum(item[1].values()), item[0].luma))
        audit[name] = {
            "rows": {split: int((rows["split"] == split).sum()) for split in splits},
            "n_distinct_table_sets": len(ordered),
            "table_sets": [_table_set_entry(tables, per_split) for tables, per_split in ordered],
        }
    return audit


def _table_set_entry(tables: JpegTables, rows: Mapping[str, int]) -> dict[str, Any]:
    qualities = matching_qualities(tables)
    chroma = tables.chroma
    return {
        "mode": tables.mode,
        "subsampling": SUBSAMPLING_NAMES.get(tables.subsampling, SUBSAMPLING_OTHER),
        "subsampling_code": tables.subsampling,
        "luma_mean": float(np.mean(tables.luma)),
        "chroma_mean": None if chroma is None else float(np.mean(chroma)),
        "tables": [list(table) for table in tables.tables],
        "rows": dict(rows),
        "matching_qualities": list(qualities),
        "match": ", ".join(f"q={quality}" for quality in qualities) or NO_STANDARD_MATCH,
    }


def class_encodings(audit: Mapping[str, Any]) -> dict[str, ClassEncoding]:
    """Return the single standard encoding of each class, or fail.

    Args:
        audit: From :func:`audit_quantization`.

    Returns:
        ``{"live": ClassEncoding, "spoof": ClassEncoding}``.

    Raises:
        QualityAuditError: Unless every class has exactly one distinct table set over all audited
            splits and that set matches exactly one quality. The message lists what was found.
    """
    problems = []
    for _, name in CLASS_NAMES:
        table_sets = audit[name]["table_sets"]
        if len(table_sets) != 1:
            problems.append(f"{name} has {len(table_sets)} distinct table sets")
        elif len(table_sets[0]["matching_qualities"]) != 1:
            problems.append(f"{name} table set matches {table_sets[0]['match']}")
    if problems:
        raise QualityAuditError(
            "The audit must find exactly one standard quality per class: " + "; ".join(problems)
        )
    encodings = {}
    for _, name in CLASS_NAMES:
        (entry,) = audit[name]["table_sets"]
        encodings[name] = ClassEncoding(entry["matching_qualities"][0], entry["subsampling_code"])
    return encodings
