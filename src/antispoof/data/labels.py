"""Layout of the CelebA-Spoof annotation vector and the dataset's path vocabulary.

Every entry in a CelebA-Spoof label file maps a relative image path of the form
``Data/{train,test}/{subject_id}/{live,spoof}/{filename}`` to a list of 44 integers. This module is
the only place in the codebase that spells out the vector layout or the path vocabulary; all other
code imports the names defined here.

Indices 40 (spoof type), 41 (illumination), 42 (environment) and 43 (live/spoof label) are verified
against the Kaggle mirror recorded in ``docs/SCHEMA.md`` §1.

The codes at indices 40–42 are **1-indexed** categories. ``CODE_NOT_APPLICABLE`` (0) is not a
category: it means "not applicable" and belongs on live rows only. A raw code is never a zero-based
class index; any class index must be derived from it explicitly.
"""

from collections.abc import Sequence

VECTOR_LENGTH = 44
"""Number of integers in every annotation vector."""

FACE_ATTRIBUTES = slice(0, 40)
"""The 40 CelebA face attributes. Populated for live images only; all zero for spoof images."""

INDEX_SPOOF_TYPE = 40
"""Spoof type code (verified). 1-indexed; ``CODE_NOT_APPLICABLE`` on live images."""

INDEX_ILLUMINATION = 41
"""Illumination condition code (verified). 1-indexed; ``CODE_NOT_APPLICABLE`` on live images."""

INDEX_ENVIRONMENT = 42
"""Environment code (verified). 1-indexed; ``CODE_NOT_APPLICABLE`` on live images."""

ATTACK_CODE_INDICES = (INDEX_SPOOF_TYPE, INDEX_ILLUMINATION, INDEX_ENVIRONMENT)
"""Indices whose codes describe an attack and are ``CODE_NOT_APPLICABLE`` on live images."""

CODE_NOT_APPLICABLE = 0
"""Value at ``ATTACK_CODE_INDICES`` on live images. Not a category; category codes start at 1."""

INDEX_LABEL = 43
"""Live/spoof label (verified). This is the training target."""

LABEL_LIVE = 0
LABEL_SPOOF = 1
LABEL_VALUES = (LABEL_LIVE, LABEL_SPOOF)

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SOURCE_SPLITS = (SPLIT_TRAIN, SPLIT_TEST)
"""Splits that exist as directories in the dataset. Validation is carved out of train."""
SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

PATH_KIND_LIVE = "live"
PATH_KIND_SPOOF = "spoof"
PATH_KIND_TO_LABEL = {PATH_KIND_LIVE: LABEL_LIVE, PATH_KIND_SPOOF: LABEL_SPOOF}
"""The label implied by the ``live``/``spoof`` directory an image is stored under."""


class LabelVectorError(ValueError):
    """Raised when an annotation vector does not have the expected layout."""


def validate_vector(vector: Sequence[int]) -> None:
    """Check that an annotation vector has exactly ``VECTOR_LENGTH`` entries.

    Args:
        vector: One annotation vector from a label file.

    Raises:
        LabelVectorError: If the vector does not have ``VECTOR_LENGTH`` entries.
    """
    if len(vector) != VECTOR_LENGTH:
        raise LabelVectorError(
            f"Annotation vector has {len(vector)} entries; expected {VECTOR_LENGTH}."
        )


def get_label(vector: Sequence[int]) -> int:
    """Read the live/spoof label (index 43).

    Args:
        vector: One annotation vector from a label file.

    Returns:
        ``LABEL_LIVE`` or ``LABEL_SPOOF``.

    Raises:
        LabelVectorError: If the vector length is wrong or the label is not a known value.
    """
    validate_vector(vector)
    label = int(vector[INDEX_LABEL])
    if label not in LABEL_VALUES:
        raise LabelVectorError(
            f"Label {label} at index {INDEX_LABEL} is not one of {LABEL_VALUES}."
        )
    return label


def get_spoof_type(vector: Sequence[int]) -> int:
    """Read the spoof type code (index 40). Zero for live images.

    Args:
        vector: One annotation vector from a label file.

    Returns:
        The raw spoof type code.

    Raises:
        LabelVectorError: If the vector length is wrong.
    """
    validate_vector(vector)
    return int(vector[INDEX_SPOOF_TYPE])


def get_illumination(vector: Sequence[int]) -> int:
    """Read the illumination condition code (index 41). Zero for live images.

    Args:
        vector: One annotation vector from a label file.

    Returns:
        The raw, 1-indexed illumination code, or ``CODE_NOT_APPLICABLE``.

    Raises:
        LabelVectorError: If the vector length is wrong.
    """
    validate_vector(vector)
    return int(vector[INDEX_ILLUMINATION])


def get_environment(vector: Sequence[int]) -> int:
    """Read the environment code (index 42). Zero for live images.

    Args:
        vector: One annotation vector from a label file.

    Returns:
        The raw, 1-indexed environment code, or ``CODE_NOT_APPLICABLE``.

    Raises:
        LabelVectorError: If the vector length is wrong.
    """
    validate_vector(vector)
    return int(vector[INDEX_ENVIRONMENT])


def validate_attack_codes(vector: Sequence[int]) -> None:
    """Check the 1-indexed code convention at ``ATTACK_CODE_INDICES`` against the label.

    A spoof vector (index 43) must carry a category code, never ``CODE_NOT_APPLICABLE``, at every
    attack-code index. A live vector must carry ``CODE_NOT_APPLICABLE`` at all of them.

    Args:
        vector: One annotation vector from a label file.

    Raises:
        LabelVectorError: If the vector length or label is invalid, a spoof vector has
            ``CODE_NOT_APPLICABLE`` at an attack-code index, or a live vector has anything else.
            The message names the offending indices.
    """
    label = get_label(vector)
    codes = {index: int(vector[index]) for index in ATTACK_CODE_INDICES}
    if label == LABEL_SPOOF:
        bad = sorted(index for index, code in codes.items() if code == CODE_NOT_APPLICABLE)
        rule = f"spoof vector has not-applicable code {CODE_NOT_APPLICABLE}"
    else:
        bad = sorted(index for index, code in codes.items() if code != CODE_NOT_APPLICABLE)
        rule = f"live vector must have code {CODE_NOT_APPLICABLE}"
    if bad:
        found = ", ".join(f"index {index} = {codes[index]}" for index in bad)
        raise LabelVectorError(f"Attack code convention violated: {rule} ({found}).")
