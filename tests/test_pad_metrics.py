"""Tests for pooled APCER, BPCER and ACER, with hand-computed expected values."""

import math

import pytest

from antispoof.eval.pad_metrics import PadMetrics, PadMetricsError, pad_metrics


def test_hand_computed_mixed_case() -> None:
    # 4 attacks: scores 0.9, 0.6 are >= 0.5 (detected); 0.4, 0.1 are accepted as bona fide.
    # 6 bona fide: scores 0.7, 0.55 are >= 0.5 (rejected); 0.2, 0.1, 0.3, 0.0 are accepted.
    labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    scores = [0.9, 0.6, 0.4, 0.1, 0.2, 0.7, 0.1, 0.3, 0.55, 0.0]
    result = pad_metrics(labels, scores, threshold=0.5)
    assert result == PadMetrics(
        threshold=0.5,
        apcer=2 / 4,
        bpcer=2 / 6,
        acer=(2 / 4 + 2 / 6) / 2,
        n_attack=4,
        n_bona_fide=6,
        n_attack_accepted=2,
        n_bona_fide_rejected=2,
    )
    assert math.isclose(result.acer, 5 / 12)


def test_everything_predicted_spoof() -> None:
    # Every score is >= threshold: no attack is accepted, every bona fide is rejected.
    result = pad_metrics([1, 1, 0, 0, 0], [0.9, 0.5, 0.5, 0.8, 1.0], threshold=0.5)
    assert (result.apcer, result.bpcer, result.acer) == (0.0, 1.0, 0.5)
    assert (result.n_attack_accepted, result.n_bona_fide_rejected) == (0, 3)


def test_threshold_zero_classifies_everything_as_spoof() -> None:
    result = pad_metrics([1, 0, 0], [0.0, 0.0, 0.3], threshold=0.0)
    assert (result.apcer, result.bpcer, result.acer) == (0.0, 1.0, 0.5)


def test_everything_predicted_live() -> None:
    result = pad_metrics([1, 1, 1, 0], [0.1, 0.2, 0.49, 0.0], threshold=0.5)
    assert (result.apcer, result.bpcer, result.acer) == (1.0, 0.0, 0.5)


def test_all_correct() -> None:
    result = pad_metrics([1, 0, 1, 0], [0.8, 0.2, 0.5, 0.49], threshold=0.5)
    assert (result.apcer, result.bpcer, result.acer) == (0.0, 0.0, 0.0)


def test_all_wrong() -> None:
    result = pad_metrics([1, 0, 1, 0], [0.2, 0.8, 0.49, 0.5], threshold=0.5)
    assert (result.apcer, result.bpcer, result.acer) == (1.0, 1.0, 1.0)


def test_score_equal_to_threshold_is_classified_as_attack() -> None:
    result = pad_metrics([1, 0], [0.5, 0.5], threshold=0.5)
    assert (result.n_attack_accepted, result.n_bona_fide_rejected) == (0, 1)


def test_float_labels_are_accepted() -> None:
    result = pad_metrics([1.0, 0.0], [0.9, 0.1], threshold=0.5)
    assert (result.n_attack, result.n_bona_fide) == (1, 1)


@pytest.mark.parametrize(
    ("labels", "scores"),
    [([1, 1, 1], [0.1, 0.6, 0.9]), ([0, 0], [0.1, 0.9]), ([], [])],
    ids=["no-bona-fide", "no-attacks", "empty"],
)
def test_absent_class_is_undefined(labels: list[int], scores: list[float]) -> None:
    with pytest.raises(PadMetricsError, match="Both classes are required"):
        pad_metrics(labels, scores, threshold=0.5)


@pytest.mark.parametrize(
    ("labels", "scores", "threshold", "message"),
    [
        ([1, 0], [0.5], 0.5, "equal in length"),
        ([[1, 0]], [[0.5, 0.5]], 0.5, "1-D"),
        ([1, 2], [0.5, 0.5], 0.5, "labels must be in"),
        ([1, 0], [0.5, float("nan")], 0.5, "finite"),
        ([1, 0], [0.5, 0.5], 1.5, r"threshold must be in \[0, 1\]"),
        ([1, 0], [0.5, 0.5], float("nan"), r"threshold must be in \[0, 1\]"),
    ],
)
def test_malformed_inputs_raise(
    labels: list[int], scores: list[float], threshold: float, message: str
) -> None:
    with pytest.raises(PadMetricsError, match=message):
        pad_metrics(labels, scores, threshold)
