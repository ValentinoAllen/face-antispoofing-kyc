"""Presentation attack detection (PAD) error rates per ISO/IEC 30107-3.

Attack presentations (spoof, label ``1``) are the positive class. A presentation is classified as an
attack when its spoof score is ``>= threshold``.
"""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

from antispoof.data.labels import LABEL_SPOOF, LABEL_VALUES


class PadMetricsError(ValueError):
    """Raised when PAD metrics are undefined or their inputs are malformed."""


@dataclass(frozen=True)
class PadMetrics:
    """Pooled PAD error rates at one threshold, with the counts behind them.

    Attributes:
        threshold: Decision threshold; a score ``>= threshold`` is classified as an attack.
        apcer: ``n_attack_accepted / n_attack``, pooled over all attack species.
        bpcer: ``n_bona_fide_rejected / n_bona_fide``.
        acer: ``(apcer + bpcer) / 2``.
        n_attack: Attack presentations evaluated.
        n_bona_fide: Bona fide presentations evaluated.
        n_attack_accepted: Attack presentations classified as bona fide (score < threshold).
        n_bona_fide_rejected: Bona fide presentations classified as attacks (score >= threshold).
    """

    threshold: float
    apcer: float
    bpcer: float
    acer: float
    n_attack: int
    n_bona_fide: int
    n_attack_accepted: int
    n_bona_fide_rejected: int

    def to_dict(self) -> dict[str, float | int]:
        """Return the metrics as a plain dict."""
        return asdict(self)


def pad_metrics(labels: ArrayLike, scores: ArrayLike, threshold: float) -> PadMetrics:
    """Compute pooled APCER, BPCER and ACER at a fixed threshold.

    Definitions, following ISO/IEC 30107-3:

    - APCER: proportion of attack presentations incorrectly classified as bona fide. The standard
      computes it per presentation attack instrument species and reports the maximum; here it is
      pooled over all attacks.
    - BPCER: proportion of bona fide presentations incorrectly classified as attacks.
    - ACER: ``(APCER + BPCER) / 2``. Not an ISO/IEC 30107-3 metric; reported for comparison with
      the literature (``docs/ARCHITECTURE.md`` ADR-004).

    Decision rule: attack iff ``score >= threshold``.

    Args:
        labels: Ground truth per presentation: ``1`` = attack (spoof), ``0`` = bona fide (live).
        scores: Spoof score per presentation, e.g. ``sigmoid(logit)``.
        threshold: Decision threshold in ``[0, 1]``.

    Returns:
        The rates and the counts behind them.

    Raises:
        PadMetricsError: If ``labels`` and ``scores`` are not 1-D and of equal length, a label is
            not 0 or 1, a score is not finite, the threshold is outside ``[0, 1]``, or either class
            is absent. A rate with a zero denominator is undefined, not zero.
    """
    is_attack, score_array = _validate(labels, scores, threshold)
    predicted_attack = score_array >= threshold
    n_attack = int(np.count_nonzero(is_attack))
    n_bona_fide = int(is_attack.size) - n_attack
    n_attack_accepted = int(np.count_nonzero(is_attack & ~predicted_attack))
    n_bona_fide_rejected = int(np.count_nonzero(~is_attack & predicted_attack))
    apcer = n_attack_accepted / n_attack
    bpcer = n_bona_fide_rejected / n_bona_fide
    return PadMetrics(
        threshold=threshold,
        apcer=apcer,
        bpcer=bpcer,
        acer=(apcer + bpcer) / 2,
        n_attack=n_attack,
        n_bona_fide=n_bona_fide,
        n_attack_accepted=n_attack_accepted,
        n_bona_fide_rejected=n_bona_fide_rejected,
    )


def _validate(
    labels: ArrayLike, scores: ArrayLike, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    label_array = np.asarray(labels)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.ndim != 1 or score_array.shape != label_array.shape:
        raise PadMetricsError(
            f"labels and scores must be 1-D and equal in length, got shapes "
            f"{label_array.shape} and {score_array.shape}."
        )
    if not 0.0 <= threshold <= 1.0:
        raise PadMetricsError(f"threshold must be in [0, 1], got {threshold}.")
    if not np.isin(label_array, LABEL_VALUES).all():
        raise PadMetricsError(f"labels must be in {LABEL_VALUES}.")
    if not np.isfinite(score_array).all():
        raise PadMetricsError("scores must all be finite.")
    is_attack = label_array == LABEL_SPOOF
    if not is_attack.any() or is_attack.all():
        raise PadMetricsError(
            f"Both classes are required: {int(is_attack.sum())} attack and "
            f"{int((~is_attack).sum())} bona fide presentations. A rate with a zero denominator "
            "is undefined."
        )
    return is_attack, score_array
