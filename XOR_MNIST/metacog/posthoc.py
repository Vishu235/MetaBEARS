"""Post-hoc detector baselines and selective-prediction curves.

This module consumes predictions already written by a frozen MetaBEARS run.
It never trains a model or selects a threshold on held-out test labels. All
detector comparisons are threshold-free ranking evaluations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np

from .consistency import predictive_disagreement, probe_concept_consistency
from .experiment import _average_precision, _binary_auroc


PathLike = Union[str, Path]


DETECTOR_DESCRIPTIONS = {
    "task_uncertainty": "One minus ensemble-mean task confidence.",
    "task_entropy": "Normalized entropy of the ensemble-mean task distribution.",
    "label_disagreement": "Task-label Jensen-Shannon ensemble disagreement.",
    "ensemble_concept_disagreement": (
        "Mean concept-level Jensen-Shannon ensemble disagreement."
    ),
    "concept_vote_disagreement": "Mean concept-level hard-vote disagreement.",
    "concept_instability_without_perturbation": (
        "Fixed 0.6/0.4 concept distribution/vote instability without an intervention."
    ),
    "perturbation_js": (
        "Concept-distribution Jensen-Shannon change under the intervention only."
    ),
    "concept_instability_with_perturbation": (
        "Probabilistic-OR combination of base and intervention instability."
    ),
    "full_metabears": (
        "MetaBEARS concept instability multiplied by task stability and confidence."
    ),
}

TARGET_DEFINITIONS = {
    "task_invariance_failure": (
        "The clean task prediction is correct and the intervened prediction is wrong."
    ),
    "semantic_instability": (
        "Clean and intervened task predictions are correct and unchanged, but at "
        "least one aligned concept prediction changes."
    ),
}


@dataclass(frozen=True)
class PosthocRunAnalysis:
    """Tabular outputs for one seed/intervention result directory."""

    metrics: Sequence[Mapping[str, Any]]
    precision_recall_curve: Sequence[Mapping[str, Any]]
    risk_coverage_curve: Sequence[Mapping[str, Any]]


def _load_prediction_artifact(path: PathLike) -> Dict[str, np.ndarray]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Prediction artifact is missing: {source}")
    required = (
        "concept_member_probabilities",
        "label_member_probabilities",
        "labels",
        "concepts",
    )
    with np.load(source, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(
                f"Prediction artifact {source} is missing: {', '.join(missing)}"
            )
        return {name: np.asarray(archive[name]) for name in required}


def _task_targets(values: np.ndarray) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1:
        raise ValueError("Task targets must be one-dimensional.")
    return labels.astype(np.int64)


def _concept_targets(values: np.ndarray) -> np.ndarray:
    concepts = np.asarray(values)
    if concepts.ndim < 2:
        raise ValueError("Concept targets must contain a concept axis.")
    return concepts.reshape(concepts.shape[0], -1).astype(np.int64)


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    safe = np.clip(values, 1e-12, 1.0)
    return -np.sum(values * np.log(safe), axis=-1) / np.log(values.shape[-1])


def detector_scores_and_targets(
    base: Mapping[str, np.ndarray],
    perturbed: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Reconstruct detector scores and controlled held-out failure labels."""

    base_concept_probabilities = np.asarray(
        base["concept_member_probabilities"], dtype=np.float64
    )
    perturbed_concept_probabilities = np.asarray(
        perturbed["concept_member_probabilities"], dtype=np.float64
    )
    base_label_probabilities = np.asarray(
        base["label_member_probabilities"], dtype=np.float64
    )
    perturbed_label_probabilities = np.asarray(
        perturbed["label_member_probabilities"], dtype=np.float64
    )
    if base_concept_probabilities.shape != perturbed_concept_probabilities.shape:
        raise ValueError("Base and intervention concept probabilities must match.")
    if base_label_probabilities.shape != perturbed_label_probabilities.shape:
        raise ValueError("Base and intervention task probabilities must match.")

    labels = _task_targets(base["labels"])
    perturbed_labels = _task_targets(perturbed["labels"])
    concepts = _concept_targets(base["concepts"])
    perturbed_concepts = _concept_targets(perturbed["concepts"])
    if not np.array_equal(labels, perturbed_labels) or not np.array_equal(
        concepts, perturbed_concepts
    ):
        raise ValueError("Base and intervention artifacts must share targets.")

    task_mean = base_label_probabilities.mean(axis=0)
    perturbed_task_mean = perturbed_label_probabilities.mean(axis=0)
    base_task = np.argmax(task_mean, axis=-1)
    perturbed_task = np.argmax(perturbed_task_mean, axis=-1)
    task_confidence = np.max(task_mean, axis=-1)
    label_disagreement = predictive_disagreement(base_label_probabilities)

    consistency = probe_concept_consistency(
        base_concept_probabilities,
        perturbed_member_probabilities=(
            perturbed_concept_probabilities[:, :, np.newaxis, :, :]
        ),
    )
    base_concept_instability = (
        0.6 * consistency.ensemble_js + 0.4 * consistency.vote_disagreement
    )
    combined_concept_instability = consistency.instability

    base_concept_predictions = np.argmax(
        base_concept_probabilities.mean(axis=0), axis=-1
    ).reshape(concepts.shape)
    perturbed_concept_predictions = np.argmax(
        perturbed_concept_probabilities.mean(axis=0), axis=-1
    ).reshape(concepts.shape)
    valid_concepts = concepts >= 0
    concept_predictions_match = np.all(
        (base_concept_predictions == perturbed_concept_predictions)
        | ~valid_concepts,
        axis=1,
    )
    base_task_correct = base_task == labels
    perturbed_task_correct = perturbed_task == labels
    task_predictions_match = base_task == perturbed_task

    targets = {
        "task_invariance_failure": base_task_correct & ~perturbed_task_correct,
        "semantic_instability": (
            base_task_correct
            & perturbed_task_correct
            & task_predictions_match
            & ~concept_predictions_match
        ),
    }
    scores = {
        "task_uncertainty": 1.0 - task_confidence,
        "task_entropy": _normalized_entropy(task_mean),
        "label_disagreement": label_disagreement,
        "ensemble_concept_disagreement": consistency.ensemble_js,
        "concept_vote_disagreement": consistency.vote_disagreement,
        "concept_instability_without_perturbation": base_concept_instability,
        "perturbation_js": consistency.perturbation_js,
        "concept_instability_with_perturbation": combined_concept_instability,
        "full_metabears": np.clip(
            combined_concept_instability
            * (1.0 - label_disagreement)
            * task_confidence,
            0.0,
            1.0,
        ),
    }
    return scores, targets


def precision_recall_curve(
    risk_scores: np.ndarray, labels: np.ndarray
) -> List[Dict[str, Any]]:
    """Return a tie-aware high-risk-first precision/recall curve."""

    scores = np.asarray(risk_scores, dtype=np.float64).reshape(-1)
    actual = np.asarray(labels, dtype=bool).reshape(-1)
    if scores.shape != actual.shape or scores.size == 0:
        raise ValueError("risk_scores and labels must be non-empty equal vectors.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("risk_scores contains non-finite values.")

    positives = int(np.count_nonzero(actual))
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = actual[order]
    rows: List[Dict[str, Any]] = [
        {
            "threshold": None,
            "reviewed_count": 0,
            "review_rate": 0.0,
            "precision": 1.0,
            "recall": 0.0,
        }
    ]
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        true_positives = int(np.count_nonzero(sorted_labels[:stop]))
        rows.append(
            {
                "threshold": float(sorted_scores[start]),
                "reviewed_count": stop,
                "review_rate": stop / scores.size,
                "precision": true_positives / stop,
                "recall": true_positives / positives if positives else 0.0,
            }
        )
        start = stop
    return rows


def risk_coverage_curve(
    risk_scores: np.ndarray, labels: np.ndarray
) -> List[Dict[str, Any]]:
    """Return a tie-aware low-risk-first selective-risk curve."""

    scores = np.asarray(risk_scores, dtype=np.float64).reshape(-1)
    actual = np.asarray(labels, dtype=bool).reshape(-1)
    if scores.shape != actual.shape or scores.size == 0:
        raise ValueError("risk_scores and labels must be non-empty equal vectors.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("risk_scores contains non-finite values.")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = actual[order]
    rows: List[Dict[str, Any]] = []
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        rows.append(
            {
                "maximum_accepted_risk": float(sorted_scores[start]),
                "accepted_count": stop,
                "coverage": stop / scores.size,
                "selective_risk": float(np.mean(sorted_labels[:stop])),
            }
        )
        start = stop
    return rows


def _area_under_risk_coverage(curve: Sequence[Mapping[str, Any]]) -> float:
    area = 0.0
    previous_coverage = 0.0
    for row in curve:
        coverage = float(row["coverage"])
        area += float(row["selective_risk"]) * (coverage - previous_coverage)
        previous_coverage = coverage
    return area


def _risk_at_coverage(
    curve: Sequence[Mapping[str, Any]], target_coverage: float
) -> float:
    for row in curve:
        if float(row["coverage"]) >= target_coverage:
            return float(row["selective_risk"])
    return float(curve[-1]["selective_risk"])


def _review_rate_at_recall(
    curve: Sequence[Mapping[str, Any]], target_recall: float, positives: int
) -> Any:
    if positives == 0:
        return None
    for row in curve:
        if float(row["recall"]) >= target_recall:
            return float(row["review_rate"])
    return 1.0


def evaluate_detector_arrays(
    base: Mapping[str, np.ndarray],
    perturbed: Mapping[str, np.ndarray],
) -> PosthocRunAnalysis:
    """Evaluate all detector rankings against both intervention targets."""

    scores, targets = detector_scores_and_targets(base, perturbed)
    metric_rows: List[Mapping[str, Any]] = []
    precision_recall_rows: List[Mapping[str, Any]] = []
    risk_coverage_rows: List[Mapping[str, Any]] = []
    for target_name, target in targets.items():
        positives = int(np.count_nonzero(target))
        negatives = int(target.size - positives)
        evaluable = positives > 0 and negatives > 0
        for detector_name, detector_score in scores.items():
            pr_curve = precision_recall_curve(detector_score, target)
            rc_curve = risk_coverage_curve(detector_score, target)
            metric_rows.append(
                {
                    "target": target_name,
                    "detector": detector_name,
                    "samples": int(target.size),
                    "positive_count": positives,
                    "prevalence": positives / target.size,
                    "evaluable": evaluable,
                    "auroc": (
                        _binary_auroc(detector_score, target) if evaluable else None
                    ),
                    "average_precision": (
                        _average_precision(detector_score, target)
                        if evaluable
                        else None
                    ),
                    "aurc": _area_under_risk_coverage(rc_curve),
                    "risk_at_80_coverage": _risk_at_coverage(rc_curve, 0.8),
                    "review_rate_at_95_recall": _review_rate_at_recall(
                        pr_curve, 0.95, positives
                    ),
                }
            )
            precision_recall_rows.extend(
                {"target": target_name, "detector": detector_name, **row}
                for row in pr_curve
            )
            risk_coverage_rows.extend(
                {"target": target_name, "detector": detector_name, **row}
                for row in rc_curve
            )
    return PosthocRunAnalysis(
        metrics=metric_rows,
        precision_recall_curve=precision_recall_rows,
        risk_coverage_curve=risk_coverage_rows,
    )


def evaluate_result_directory(
    result_directory: PathLike,
    *,
    seed: int,
    intervention: str,
) -> PosthocRunAnalysis:
    """Load one frozen run's held-out artifacts and evaluate detectors."""

    directory = Path(result_directory).expanduser().resolve()
    base = _load_prediction_artifact(directory / "id_test_predictions.npz")
    perturbed = _load_prediction_artifact(
        directory / "id_test_intervention_predictions.npz"
    )
    analysis = evaluate_detector_arrays(base, perturbed)

    def annotate(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        return [
            {
                "seed": int(seed),
                "intervention": str(intervention),
                **dict(row),
            }
            for row in rows
        ]

    return PosthocRunAnalysis(
        metrics=annotate(analysis.metrics),
        precision_recall_curve=annotate(analysis.precision_recall_curve),
        risk_coverage_curve=annotate(analysis.risk_coverage_curve),
    )
