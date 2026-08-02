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
from .experiment import _average_precision, _binary_auroc, _paired_probability_js


PathLike = Union[str, Path]


DETECTOR_DESCRIPTIONS = {
    "task_uncertainty": "One minus ensemble-mean task confidence.",
    "task_entropy": "Normalized entropy of the ensemble-mean task distribution.",
    "task_distribution_js": (
        "Jensen-Shannon change in the ensemble-mean task distribution under "
        "the intervention."
    ),
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
    "validation_fitted_fusion_v2": (
        "Protocol-v2 non-negative fusion fitted by cross-validation on the "
        "primary validation intervention only."
    ),
    "intervention_calibrated_fusion_v3": (
        "Protocol-v3 fusion with v2 weights and threshold, using only the "
        "current intervention's unlabeled validation scores to recalibrate "
        "empirical percentile references."
    ),
    "leave_one_intervention_out_fusion_v4": (
        "Protocol-v4 fusion fitted on validation artifacts from every training "
        "intervention while excluding the evaluated intervention entirely."
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
    "controlled_failure_union": (
        "The union of task-invariance failure and semantic instability."
    ),
}


@dataclass(frozen=True)
class PosthocRunAnalysis:
    """Tabular outputs for one seed/intervention result directory."""

    metrics: Sequence[Mapping[str, Any]]
    precision_recall_curve: Sequence[Mapping[str, Any]]
    risk_coverage_curve: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class ValidationFusionModel:
    """Frozen validation-fitted linear fusion over percentile risk channels."""

    signal_names: Tuple[str, ...]
    weights: np.ndarray
    references: Mapping[str, np.ndarray]
    threshold: float
    threshold_target_recall: float
    cross_validation_folds: int
    out_of_fold_average_precision: float
    out_of_fold_auroc: float
    validation_positive_count: int
    validation_prevalence: float
    validation_review_rate: float
    validation_precision: float
    validation_recall: float
    validation_f1: float
    validation_group_count: int | None = None
    validation_min_group_recall: float | None = None

    def score(
        self,
        raw_scores: Mapping[str, np.ndarray],
        *,
        references: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        active_references = self.references if references is None else references
        transformed = []
        for name in self.signal_names:
            if name not in raw_scores:
                raise ValueError(f"Fusion input is missing signal '{name}'.")
            if name not in active_references:
                raise ValueError(f"Fusion references are missing signal '{name}'.")
            transformed.append(
                _empirical_midrank_percentile(
                    active_references[name], np.asarray(raw_scores[name])
                )
            )
        matrix = np.column_stack(transformed)
        return np.clip(matrix @ self.weights, 0.0, 1.0)

    def to_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "threshold": self.threshold,
            "threshold_target_recall": self.threshold_target_recall,
            "cross_validation_folds": self.cross_validation_folds,
            "out_of_fold_average_precision": self.out_of_fold_average_precision,
            "out_of_fold_auroc": self.out_of_fold_auroc,
            "validation_positive_count": self.validation_positive_count,
            "validation_prevalence": self.validation_prevalence,
            "validation_review_rate": self.validation_review_rate,
            "validation_precision": self.validation_precision,
            "validation_recall": self.validation_recall,
            "validation_f1": self.validation_f1,
            "validation_group_count": self.validation_group_count,
            "validation_min_group_recall": self.validation_min_group_recall,
        }
        record.update(
            {
                f"weight_{name}": float(weight)
                for name, weight in zip(self.signal_names, self.weights)
            }
        )
        return record


@dataclass(frozen=True)
class FusionRunAnalysis:
    """Ranking and validation-selected threshold results for one held-out run."""

    analysis: PosthocRunAnalysis
    threshold_metrics: Sequence[Mapping[str, Any]]


def _load_prediction_artifact(
    path: PathLike, *, include_targets: bool = True
) -> Dict[str, np.ndarray]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Prediction artifact is missing: {source}")
    required = [
        "concept_member_probabilities",
        "label_member_probabilities",
    ]
    if include_targets:
        required.extend(("labels", "concepts"))
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


def _detector_components(
    base: Mapping[str, np.ndarray],
    perturbed: Mapping[str, np.ndarray],
) -> Tuple[
    Dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute score channels and predictions without reading target arrays."""

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
    ).reshape(base_concept_probabilities.shape[1], -1)
    perturbed_concept_predictions = np.argmax(
        perturbed_concept_probabilities.mean(axis=0), axis=-1
    ).reshape(perturbed_concept_probabilities.shape[1], -1)
    scores = {
        "task_uncertainty": 1.0 - task_confidence,
        "task_entropy": _normalized_entropy(task_mean),
        "task_distribution_js": _paired_probability_js(
            task_mean, perturbed_task_mean
        ),
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
    return (
        scores,
        base_task,
        perturbed_task,
        base_concept_predictions,
        perturbed_concept_predictions,
    )


def detector_scores(
    base: Mapping[str, np.ndarray],
    perturbed: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Compute detector channels without inspecting labels or concept targets."""

    scores, _, _, _, _ = _detector_components(base, perturbed)
    return scores


def detector_scores_and_targets(
    base: Mapping[str, np.ndarray],
    perturbed: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Reconstruct detector scores and controlled held-out failure labels."""

    (
        scores,
        base_task,
        perturbed_task,
        base_concept_predictions,
        perturbed_concept_predictions,
    ) = _detector_components(base, perturbed)
    labels = _task_targets(base["labels"])
    perturbed_labels = _task_targets(perturbed["labels"])
    concepts = _concept_targets(base["concepts"])
    perturbed_concepts = _concept_targets(perturbed["concepts"])
    if not np.array_equal(labels, perturbed_labels) or not np.array_equal(
        concepts, perturbed_concepts
    ):
        raise ValueError("Base and intervention artifacts must share targets.")
    if (
        labels.size != base_task.size
        or concepts.shape != base_concept_predictions.shape
    ):
        raise ValueError("Prediction and target sample shapes must match.")

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
    targets["controlled_failure_union"] = (
        targets["task_invariance_failure"] | targets["semantic_instability"]
    )
    return scores, targets


def calibrate_fusion_references(
    base: Mapping[str, np.ndarray],
    perturbed: Mapping[str, np.ndarray],
    *,
    signal_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Fit percentile references from predictions, without reading targets."""

    names = tuple(str(name) for name in signal_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("signal_names must be a non-empty unique sequence.")
    scores = detector_scores(base, perturbed)
    missing = [name for name in names if name not in scores]
    if missing:
        raise ValueError(f"Unknown fusion signals: {', '.join(missing)}")
    return {
        name: np.sort(np.asarray(scores[name], dtype=np.float64))
        for name in names
    }


def _empirical_midrank_percentile(
    reference: np.ndarray, values: np.ndarray
) -> np.ndarray:
    observed = np.asarray(reference, dtype=np.float64).reshape(-1)
    requested = np.asarray(values, dtype=np.float64).reshape(-1)
    if observed.size == 0 or not np.all(np.isfinite(observed)):
        raise ValueError("Fusion references must contain finite values.")
    if not np.all(np.isfinite(requested)):
        raise ValueError("Fusion inputs must contain finite values.")
    ordered = np.sort(observed)
    left = np.searchsorted(ordered, requested, side="left")
    right = np.searchsorted(ordered, requested, side="right")
    return (left + right) / (2.0 * ordered.size)


def _integer_compositions(total: int, parts: int) -> List[Tuple[int, ...]]:
    if total < 0 or parts < 1:
        raise ValueError("total must be non-negative and parts must be positive.")
    if parts == 1:
        return [(total,)]
    output: List[Tuple[int, ...]] = []
    for first in range(total + 1):
        for remainder in _integer_compositions(total - first, parts - 1):
            output.append((first, *remainder))
    return output


def _stratified_fold_ids(
    labels: np.ndarray, requested_folds: int, *, seed: int
) -> Tuple[np.ndarray, int]:
    actual = np.asarray(labels, dtype=bool).reshape(-1)
    positives = int(np.count_nonzero(actual))
    negatives = int(actual.size - positives)
    effective_folds = min(int(requested_folds), positives, negatives)
    if effective_folds < 2:
        return np.zeros(actual.size, dtype=np.int64), 1
    fold_ids = np.empty(actual.size, dtype=np.int64)
    random = np.random.default_rng(int(seed))
    for class_value in (False, True):
        indices = np.flatnonzero(actual == class_value)
        random.shuffle(indices)
        fold_ids[indices] = np.arange(indices.size) % effective_folds
    return fold_ids, effective_folds


def _flag_metrics(flags: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    predicted = np.asarray(flags, dtype=bool).reshape(-1)
    actual = np.asarray(labels, dtype=bool).reshape(-1)
    if predicted.shape != actual.shape:
        raise ValueError("flags and labels must have equal shapes.")
    true_positives = int(np.count_nonzero(predicted & actual))
    false_positives = int(np.count_nonzero(predicted & ~actual))
    false_negatives = int(np.count_nonzero(~predicted & actual))
    flagged = true_positives + false_positives
    positives = true_positives + false_negatives
    precision = true_positives / flagged if flagged else 0.0
    recall = true_positives / positives if positives else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "flagged_count": flagged,
        "positive_count": positives,
        "review_rate": flagged / actual.size,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def fit_validation_fusion(
    base: Mapping[str, np.ndarray],
    perturbed: Mapping[str, np.ndarray],
    *,
    signal_names: Sequence[str],
    target_name: str = "controlled_failure_union",
    weight_grid_step: float = 0.1,
    cross_validation_folds: int = 5,
    threshold_target_recall: float = 0.95,
    seed: int = 0,
) -> ValidationFusionModel:
    """Select monotone fusion weights using intervention validation labels only."""

    if not 0.0 < weight_grid_step <= 1.0:
        raise ValueError("weight_grid_step must lie within (0, 1].")
    units = int(round(1.0 / weight_grid_step))
    if not np.isclose(units * weight_grid_step, 1.0):
        raise ValueError("weight_grid_step must divide one exactly.")
    if not 0.0 < threshold_target_recall <= 1.0:
        raise ValueError("threshold_target_recall must lie within (0, 1].")
    names = tuple(str(name) for name in signal_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("signal_names must be a non-empty unique sequence.")

    scores, targets = detector_scores_and_targets(base, perturbed)
    if target_name not in targets:
        raise ValueError(f"Unknown fusion target '{target_name}'.")
    missing = [name for name in names if name not in scores]
    if missing:
        raise ValueError(f"Unknown fusion signals: {', '.join(missing)}")
    target = np.asarray(targets[target_name], dtype=bool)
    positives = int(np.count_nonzero(target))
    negatives = int(target.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Fusion fitting requires positive and negative validation cases.")

    fold_ids, effective_folds = _stratified_fold_ids(
        target, cross_validation_folds, seed=seed
    )
    transformed = np.empty((target.size, len(names)), dtype=np.float64)
    if effective_folds == 1:
        for signal_index, name in enumerate(names):
            transformed[:, signal_index] = _empirical_midrank_percentile(
                scores[name], scores[name]
            )
    else:
        for fold in range(effective_folds):
            held_out = fold_ids == fold
            fit = ~held_out
            for signal_index, name in enumerate(names):
                transformed[held_out, signal_index] = (
                    _empirical_midrank_percentile(
                        np.asarray(scores[name])[fit],
                        np.asarray(scores[name])[held_out],
                    )
                )

    best_key = None
    best_weights = None
    for composition in _integer_compositions(units, len(names)):
        weights = np.asarray(composition, dtype=np.float64) / units
        fused = transformed @ weights
        average_precision = _average_precision(fused, target)
        auroc = _binary_auroc(fused, target)
        nonzero = int(np.count_nonzero(weights))
        key = (
            average_precision,
            auroc,
            -nonzero,
            tuple(float(value) for value in weights),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_weights = weights
    if best_key is None or best_weights is None:
        raise RuntimeError("Fusion weight search produced no candidates.")

    references = {name: np.sort(np.asarray(scores[name])) for name in names}
    full_matrix = np.column_stack(
        [
            _empirical_midrank_percentile(references[name], scores[name])
            for name in names
        ]
    )
    validation_score = full_matrix @ best_weights
    threshold_row = next(
        row
        for row in precision_recall_curve(validation_score, target)
        if row["threshold"] is not None
        and float(row["recall"]) >= threshold_target_recall
    )
    threshold = float(threshold_row["threshold"])
    threshold_metrics = _flag_metrics(validation_score >= threshold, target)
    return ValidationFusionModel(
        signal_names=names,
        weights=best_weights,
        references=references,
        threshold=threshold,
        threshold_target_recall=threshold_target_recall,
        cross_validation_folds=effective_folds,
        out_of_fold_average_precision=float(best_key[0]),
        out_of_fold_auroc=float(best_key[1]),
        validation_positive_count=positives,
        validation_prevalence=positives / target.size,
        validation_review_rate=float(threshold_metrics["review_rate"]),
        validation_precision=float(threshold_metrics["precision"]),
        validation_recall=float(threshold_metrics["recall"]),
        validation_f1=float(threshold_metrics["f1"]),
    )


def fit_leave_one_intervention_out_fusion(
    validation_pairs: Mapping[
        str, Tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]
    ],
    *,
    signal_names: Sequence[str],
    target_name: str = "controlled_failure_union",
    weight_grid_step: float = 0.1,
    threshold_target_recall: float = 0.95,
) -> ValidationFusionModel:
    """Fit using intervention-blocked validation folds only.

    Each supplied intervention is held out once while percentile references
    are built from the remaining interventions. The final references and
    threshold pool all supplied training interventions. The future evaluation
    intervention must therefore not appear in ``validation_pairs``.
    """

    if len(validation_pairs) < 2:
        raise ValueError(
            "Leave-one-intervention-out fitting requires at least two "
            "training interventions."
        )
    if not 0.0 < weight_grid_step <= 1.0:
        raise ValueError("weight_grid_step must lie within (0, 1].")
    units = int(round(1.0 / weight_grid_step))
    if not np.isclose(units * weight_grid_step, 1.0):
        raise ValueError("weight_grid_step must divide one exactly.")
    if not 0.0 < threshold_target_recall <= 1.0:
        raise ValueError("threshold_target_recall must lie within (0, 1].")
    names = tuple(str(name) for name in signal_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("signal_names must be a non-empty unique sequence.")

    interventions = tuple(sorted(str(name) for name in validation_pairs))
    score_sets: Dict[str, Dict[str, np.ndarray]] = {}
    target_sets: Dict[str, np.ndarray] = {}
    for intervention in interventions:
        base, perturbed = validation_pairs[intervention]
        scores, targets = detector_scores_and_targets(base, perturbed)
        if target_name not in targets:
            raise ValueError(f"Unknown fusion target '{target_name}'.")
        missing = [name for name in names if name not in scores]
        if missing:
            raise ValueError(f"Unknown fusion signals: {', '.join(missing)}")
        score_sets[intervention] = {
            name: np.asarray(scores[name], dtype=np.float64) for name in names
        }
        target_sets[intervention] = np.asarray(
            targets[target_name], dtype=bool
        )
        positives = int(np.count_nonzero(target_sets[intervention]))
        negatives = int(target_sets[intervention].size - positives)
        if positives == 0 or negatives == 0:
            raise ValueError(
                "Each training intervention requires positive and negative cases."
            )

    transformed_folds = []
    target_folds = []
    for held_out in interventions:
        training = [name for name in interventions if name != held_out]
        transformed_folds.append(
            np.column_stack(
                [
                    _empirical_midrank_percentile(
                        np.concatenate(
                            [score_sets[name][signal] for name in training]
                        ),
                        score_sets[held_out][signal],
                    )
                    for signal in names
                ]
            )
        )
        target_folds.append(target_sets[held_out])
    best_key = None
    best_weights = None
    for composition in _integer_compositions(units, len(names)):
        weights = np.asarray(composition, dtype=np.float64) / units
        fold_scores = [matrix @ weights for matrix in transformed_folds]
        average_precision = float(
            np.mean(
                [
                    _average_precision(scores, target)
                    for scores, target in zip(fold_scores, target_folds)
                ]
            )
        )
        auroc = float(
            np.mean(
                [
                    _binary_auroc(scores, target)
                    for scores, target in zip(fold_scores, target_folds)
                ]
            )
        )
        nonzero = int(np.count_nonzero(weights))
        key = (
            average_precision,
            auroc,
            -nonzero,
            tuple(float(value) for value in weights),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_weights = weights
    if best_key is None or best_weights is None:
        raise RuntimeError("Fusion weight search produced no candidates.")

    references = {
        signal: np.sort(
            np.concatenate(
                [score_sets[intervention][signal] for intervention in interventions]
            )
        )
        for signal in names
    }
    full_scores = {
        intervention: np.column_stack(
            [
                _empirical_midrank_percentile(
                    references[signal], score_sets[intervention][signal]
                )
                for signal in names
            ]
        )
        @ best_weights
        for intervention in interventions
    }
    group_thresholds = []
    for intervention in interventions:
        threshold_row = next(
            row
            for row in precision_recall_curve(
                full_scores[intervention], target_sets[intervention]
            )
            if row["threshold"] is not None
            and float(row["recall"]) >= threshold_target_recall
        )
        group_thresholds.append(float(threshold_row["threshold"]))
    threshold = min(group_thresholds)
    validation_score = np.concatenate(
        [full_scores[intervention] for intervention in interventions]
    )
    pooled_target = np.concatenate(
        [target_sets[intervention] for intervention in interventions]
    )
    threshold_metrics = _flag_metrics(
        validation_score >= threshold, pooled_target
    )
    minimum_group_recall = min(
        _flag_metrics(
            full_scores[intervention] >= threshold,
            target_sets[intervention],
        )["recall"]
        for intervention in interventions
    )
    return ValidationFusionModel(
        signal_names=names,
        weights=best_weights,
        references=references,
        threshold=threshold,
        threshold_target_recall=threshold_target_recall,
        cross_validation_folds=len(interventions),
        out_of_fold_average_precision=float(best_key[0]),
        out_of_fold_auroc=float(best_key[1]),
        validation_positive_count=int(np.count_nonzero(pooled_target)),
        validation_prevalence=float(np.mean(pooled_target)),
        validation_review_rate=float(threshold_metrics["review_rate"]),
        validation_precision=float(threshold_metrics["precision"]),
        validation_recall=float(threshold_metrics["recall"]),
        validation_f1=float(threshold_metrics["f1"]),
        validation_group_count=len(interventions),
        validation_min_group_recall=float(minimum_group_recall),
    )


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


def _evaluate_score_mapping(
    scores: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray]
) -> PosthocRunAnalysis:
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


def evaluate_detector_arrays(
    base: Mapping[str, np.ndarray],
    perturbed: Mapping[str, np.ndarray],
) -> PosthocRunAnalysis:
    """Evaluate all detector rankings against intervention-derived targets."""

    scores, targets = detector_scores_and_targets(base, perturbed)
    return _evaluate_score_mapping(scores, targets)


def evaluate_fusion_arrays(
    model: ValidationFusionModel,
    base: Mapping[str, np.ndarray],
    perturbed: Mapping[str, np.ndarray],
    *,
    detector_name: str = "validation_fitted_fusion_v2",
    references: Mapping[str, np.ndarray] | None = None,
) -> FusionRunAnalysis:
    """Apply one validation-fitted model to an untouched held-out split."""

    scores, targets = detector_scores_and_targets(base, perturbed)
    fused = model.score(scores, references=references)
    analysis = _evaluate_score_mapping({detector_name: fused}, targets)
    flags = fused >= model.threshold
    threshold_metrics = []
    for target_name, target in targets.items():
        threshold_metrics.append(
            {
                "target": target_name,
                "detector": detector_name,
                "threshold": model.threshold,
                **_flag_metrics(flags, target),
            }
        )
    return FusionRunAnalysis(
        analysis=analysis,
        threshold_metrics=threshold_metrics,
    )


def fit_validation_fusion_from_result_directory(
    result_directory: PathLike,
    **fit_arguments: Any,
) -> ValidationFusionModel:
    """Fit Protocol-v2 fusion from one primary validation intervention."""

    directory = Path(result_directory).expanduser().resolve()
    base = _load_prediction_artifact(directory / "validation_predictions.npz")
    perturbed = _load_prediction_artifact(
        directory / "validation_intervention_predictions.npz"
    )
    return fit_validation_fusion(base, perturbed, **fit_arguments)


def fit_leave_one_intervention_out_fusion_from_result_directories(
    result_directories: Mapping[str, PathLike],
    **fit_arguments: Any,
) -> ValidationFusionModel:
    """Load training-intervention validation artifacts and fit Protocol v4."""

    validation_pairs = {}
    for intervention, result_directory in result_directories.items():
        directory = Path(result_directory).expanduser().resolve()
        validation_pairs[str(intervention)] = (
            _load_prediction_artifact(directory / "validation_predictions.npz"),
            _load_prediction_artifact(
                directory / "validation_intervention_predictions.npz"
            ),
        )
    return fit_leave_one_intervention_out_fusion(
        validation_pairs,
        **fit_arguments,
    )


def calibrate_fusion_references_from_result_directory(
    result_directory: PathLike,
    *,
    signal_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Fit unlabeled validation references for one seed and intervention."""

    directory = Path(result_directory).expanduser().resolve()
    base = _load_prediction_artifact(
        directory / "validation_predictions.npz", include_targets=False
    )
    perturbed = _load_prediction_artifact(
        directory / "validation_intervention_predictions.npz",
        include_targets=False,
    )
    return calibrate_fusion_references(
        base,
        perturbed,
        signal_names=signal_names,
    )


def evaluate_fusion_result_directory(
    model: ValidationFusionModel,
    result_directory: PathLike,
    *,
    seed: int,
    intervention: str,
    detector_name: str = "validation_fitted_fusion_v2",
    references: Mapping[str, np.ndarray] | None = None,
) -> FusionRunAnalysis:
    """Evaluate a frozen fusion on one held-out control directory."""

    directory = Path(result_directory).expanduser().resolve()
    base = _load_prediction_artifact(directory / "id_test_predictions.npz")
    perturbed = _load_prediction_artifact(
        directory / "id_test_intervention_predictions.npz"
    )
    result = evaluate_fusion_arrays(
        model,
        base,
        perturbed,
        detector_name=detector_name,
        references=references,
    )

    def annotate(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        return [
            {
                "seed": int(seed),
                "intervention": str(intervention),
                **dict(row),
            }
            for row in rows
        ]

    return FusionRunAnalysis(
        analysis=PosthocRunAnalysis(
            metrics=annotate(result.analysis.metrics),
            precision_recall_curve=annotate(
                result.analysis.precision_recall_curve
            ),
            risk_coverage_curve=annotate(result.analysis.risk_coverage_curve),
        ),
        threshold_metrics=annotate(result.threshold_metrics),
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
