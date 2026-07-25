"""Validation-based threshold selection for meta-cognitive review flags."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {vector.shape}.")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values.")
    return vector


def _unit_interval(value: float, *, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie within [0, 1].")


@dataclass(frozen=True)
class ThresholdSelection:
    """A validation-selected threshold and the metrics that justify it."""

    threshold: float
    policy: str
    higher_is_riskier: bool
    precision: float
    recall: float
    f1: float
    review_rate: float
    false_review_rate: float
    flagged_count: int
    failure_count: int
    detected_failure_count: int
    missed_failure_count: int
    candidate_count: int
    constraints_satisfied: bool

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite.")
        if self.policy not in ("max_f1", "constrained"):
            raise ValueError("policy must be 'max_f1' or 'constrained'.")
        for name in ("precision", "recall", "f1", "review_rate", "false_review_rate"):
            _unit_interval(getattr(self, name), name=name)
        for name in (
            "flagged_count",
            "failure_count",
            "detected_failure_count",
            "missed_failure_count",
            "candidate_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-safe representation of the selection."""

        return {
            "threshold": float(self.threshold),
            "policy": self.policy,
            "higher_is_riskier": bool(self.higher_is_riskier),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "review_rate": float(self.review_rate),
            "false_review_rate": float(self.false_review_rate),
            "flagged_count": int(self.flagged_count),
            "failure_count": int(self.failure_count),
            "detected_failure_count": int(self.detected_failure_count),
            "missed_failure_count": int(self.missed_failure_count),
            "candidate_count": int(self.candidate_count),
            "constraints_satisfied": bool(self.constraints_satisfied),
        }


def select_review_threshold(
    scores: np.ndarray,
    failure_labels: np.ndarray,
    *,
    target_precision: Optional[float] = None,
    max_false_review_rate: Optional[float] = None,
    higher_is_riskier: bool = True,
    on_infeasible: str = "raise",
) -> ThresholdSelection:
    """Select a review threshold on a labeled validation split.

    ``scores`` and ``failure_labels`` must come from a validation split
    disjoint from any split used for reported test metrics. This function
    cannot verify that; leakage here silently invalidates every downstream
    detection number.

    ``failure_labels`` marks, per validation sample, whether that sample is
    a ground-truth failure of the kind ``scores`` is meant to detect (for
    example a shortcut-consistent semantic failure for ``shortcut_risk``, or
    an out-of-distribution sample for ``neural_familiarity``).

    With ``higher_is_riskier=True`` (the default) the returned threshold is
    meant to be applied as ``scores >= threshold``; with ``False``, as
    ``scores <= threshold`` (for example for familiarity, where low values
    are risky).

    With neither ``target_precision`` nor ``max_false_review_rate`` supplied,
    the threshold maximizing F1 is returned. With either supplied, the
    smallest threshold (most sensitive) satisfying all supplied constraints
    is returned. If no candidate satisfies the constraints,
    ``on_infeasible="raise"`` (the default) raises ``ValueError`` naming the
    best achievable precision/false-review-rate; ``on_infeasible="max_f1"``
    instead falls back to the unconstrained max-F1 threshold and reports
    ``constraints_satisfied=False``.
    """

    score_vector = _finite_vector(scores, name="scores")
    raw_labels = np.asarray(failure_labels)
    if raw_labels.ndim != 1:
        raise ValueError(
            f"failure_labels must be one-dimensional; got {raw_labels.shape}."
        )
    if raw_labels.shape[0] != score_vector.shape[0]:
        raise ValueError("scores and failure_labels must have the same length.")
    labels = raw_labels.astype(bool)
    if not np.any(labels):
        raise ValueError(
            "failure_labels must contain at least one failure to calibrate against."
        )
    if target_precision is not None:
        _unit_interval(target_precision, name="target_precision")
    if max_false_review_rate is not None:
        _unit_interval(max_false_review_rate, name="max_false_review_rate")
    if on_infeasible not in ("raise", "max_f1"):
        raise ValueError("on_infeasible must be 'raise' or 'max_f1'.")

    risk = score_vector if higher_is_riskier else -score_vector
    failure_count = int(np.count_nonzero(labels))
    negative_count = labels.shape[0] - failure_count

    candidates = np.unique(risk)
    candidate_count = candidates.shape[0]
    precisions = np.empty(candidate_count, dtype=np.float64)
    recalls = np.empty(candidate_count, dtype=np.float64)
    f1_scores = np.empty(candidate_count, dtype=np.float64)
    review_rates = np.empty(candidate_count, dtype=np.float64)
    false_review_rates = np.empty(candidate_count, dtype=np.float64)
    flagged_counts = np.empty(candidate_count, dtype=np.int64)
    true_positive_counts = np.empty(candidate_count, dtype=np.int64)

    for index, candidate in enumerate(candidates):
        flags = risk >= candidate
        true_positives = int(np.count_nonzero(flags & labels))
        false_positives = int(np.count_nonzero(flags & ~labels))
        flagged = int(np.count_nonzero(flags))

        precision = true_positives / flagged if flagged else 0.0
        recall = true_positives / failure_count
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )

        precisions[index] = precision
        recalls[index] = recall
        f1_scores[index] = f1
        review_rates[index] = flagged / labels.shape[0]
        false_review_rates[index] = (
            false_positives / negative_count if negative_count else 0.0
        )
        flagged_counts[index] = flagged
        true_positive_counts[index] = true_positives

    def _best_f1_index() -> int:
        best = f1_scores.max()
        tied = np.flatnonzero(np.isclose(f1_scores, best))
        return int(tied.max())  # prefer the larger threshold: fewer reviews on ties

    if target_precision is None and max_false_review_rate is None:
        policy = "max_f1"
        selected = _best_f1_index()
        constraints_satisfied = True
    else:
        satisfies = np.ones(candidate_count, dtype=bool)
        if target_precision is not None:
            satisfies &= precisions >= target_precision
        if max_false_review_rate is not None:
            satisfies &= false_review_rates <= max_false_review_rate

        if np.any(satisfies):
            policy = "constrained"
            selected = int(np.flatnonzero(satisfies).min())  # most sensitive feasible
            constraints_satisfied = True
        elif on_infeasible == "max_f1":
            policy = "max_f1"
            selected = _best_f1_index()
            constraints_satisfied = False
        else:
            raise ValueError(
                "No threshold satisfies the requested constraints on this "
                f"validation split (best achievable precision={precisions.max():.3f}, "
                f"best achievable false_review_rate={false_review_rates.min():.3f}). "
                "Loosen target_precision/max_false_review_rate or pass "
                "on_infeasible='max_f1' to fall back."
            )

    threshold = candidates[selected] if higher_is_riskier else -candidates[selected]

    return ThresholdSelection(
        threshold=float(threshold),
        policy=policy,
        higher_is_riskier=higher_is_riskier,
        precision=float(precisions[selected]),
        recall=float(recalls[selected]),
        f1=float(f1_scores[selected]),
        review_rate=float(review_rates[selected]),
        false_review_rate=float(false_review_rates[selected]),
        flagged_count=int(flagged_counts[selected]),
        failure_count=failure_count,
        detected_failure_count=int(true_positive_counts[selected]),
        missed_failure_count=failure_count - int(true_positive_counts[selected]),
        candidate_count=candidate_count,
        constraints_satisfied=constraints_satisfied,
    )
