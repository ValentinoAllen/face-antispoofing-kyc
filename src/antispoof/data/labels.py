"""Layout of the CelebA-Spoof annotation vector and the dataset's path vocabulary.

Every entry in a CelebA-Spoof label file maps a relative image path of the form
``Data/{train,test}/{subject_id}/{live,spoof}/{filename}`` to a list of 44 integers. This module is
the only place in the codebase that spells out the vector layout or the path vocabulary; all other
code imports the names defined here.

Index 40 (spoof type) and index 43 (live/spoof label) are verified against the Kaggle mirror
recorded in ``docs/SCHEMA.md`` §1. The meaning of indices 41 and 42 is provisional, so their names
state only the index.
"""

from collections.abc import Sequence

VECTOR_LENGTH = 44
"""Number of integers in every annotation vector."""

FACE_ATTRIBUTES = slice(0, 40)
"""The 40 CelebA face attributes. Populated for live images only; all zero for spoof images."""

INDEX_SPOOF_TYPE = 40
"""Spoof type code (verified). Zero for live images."""

# TODO: provisionally the illumination condition, by documentation convention only. Confirm with
# unique-value counts on the mirror before renaming. Zero for live images (measured).
INDEX_ATTR_41 = 41

# TODO: provisionally the environment, by documentation convention only. Confirm with unique-value
# counts on the mirror before renaming. Zero for live images (measured).
INDEX_ATTR_42 = 42

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


def get_attr_41(vector: Sequence[int]) -> int:
    """Read the raw code at index 41.

    TODO: provisionally the illumination condition; confirm by unique-value counts before renaming.

    Args:
        vector: One annotation vector from a label file.

    Returns:
        The raw code at index 41.

    Raises:
        LabelVectorError: If the vector length is wrong.
    """
    validate_vector(vector)
    return int(vector[INDEX_ATTR_41])


def get_attr_42(vector: Sequence[int]) -> int:
    """Read the raw code at index 42.

    TODO: provisionally the environment; confirm by unique-value counts before renaming.

    Args:
        vector: One annotation vector from a label file.

    Returns:
        The raw code at index 42.

    Raises:
        LabelVectorError: If the vector length is wrong.
    """
    validate_vector(vector)
    return int(vector[INDEX_ATTR_42])
