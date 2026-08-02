"""Validation-calibrated, end-to-end MetaBEARS experiment orchestration."""

from dataclasses import dataclass
from itertools import islice
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from .familiarity import familiarity_from_reference
from .integration import (
    EnsemblePredictions,
    collect_ensemble_predictions,
    ensemble_leave_one_out_reference_distances,
    ensemble_nearest_reference_distances,
)
from .interventions import PredictionIntervention
from .report import MetaCognitiveReport, build_meta_cognitive_report
from .thresholds import select_review_threshold


PathLike = Union[str, Path]


def _unit_interval(value: float, *, name: str) -> float:
    converted = float(value)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must lie within [0, 1].")
    return converted


def _require_experiment_fields(predictions: EnsemblePredictions) -> None:
    if predictions.member_representations is None:
        raise ValueError("The experiment requires member representations.")
    if predictions.labels is None or predictions.concepts is None:
        raise ValueError("The experiment requires task and concept targets.")


def _target_vectors(predictions: EnsemblePredictions) -> tuple:
    _require_experiment_fields(predictions)
    labels = np.asarray(predictions.labels)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1:
        raise ValueError("Task targets must be a one-dimensional vector.")

    concepts = np.asarray(predictions.concepts)
    if concepts.ndim < 2:
        raise ValueError("Concept targets must contain a concept axis.")
    concepts = concepts.reshape(concepts.shape[0], -1)
    return labels.astype(np.int64), concepts.astype(np.int64)


def shortcut_proxy_labels(predictions: EnsemblePredictions) -> np.ndarray:
    """Mark validation samples with a task-correct/concept-wrong mismatch.

    This is an observable calibration proxy, not ground-truth evidence that a
    model used a reasoning shortcut. Controlled shortcut experiments must
    still evaluate the final detector against intervention-derived labels.
    """

    labels, concepts = _target_vectors(predictions)
    predicted_labels = np.argmax(
        predictions.label_member_probabilities.mean(axis=0), axis=-1
    )
    predicted_concepts = np.argmax(
        predictions.concept_member_probabilities.mean(axis=0), axis=-1
    ).reshape(concepts.shape[0], -1)
    if predicted_concepts.shape != concepts.shape:
        raise ValueError(
            "Predicted and target concept tensors must have matching sample "
            "and concept dimensions."
        )

    valid_concepts = np.all(concepts >= 0, axis=1)
    concept_error = np.any(predicted_concepts != concepts, axis=1)
    task_correct = predicted_labels == labels
    return valid_concepts & concept_error & task_correct


def _classification_metrics(flags: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    predicted = np.asarray(flags, dtype=bool)
    actual = np.asarray(labels, dtype=bool)
    if predicted.ndim != 1 or actual.ndim != 1 or predicted.shape != actual.shape:
        raise ValueError("flags and labels must be equal-length vectors.")

    true_positives = int(np.count_nonzero(predicted & actual))
    false_positives = int(np.count_nonzero(predicted & ~actual))
    false_negatives = int(np.count_nonzero(~predicted & actual))
    true_negatives = int(np.count_nonzero(~predicted & ~actual))
    flagged = true_positives + false_positives
    positives = true_positives + false_negatives
    negatives = false_positives + true_negatives
    precision = true_positives / flagged if flagged else 0.0
    recall = true_positives / positives if positives else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positives / negatives if negatives else 0.0,
        "flagged_count": flagged,
        "positive_count": positives,
        "true_positive_count": true_positives,
        "false_positive_count": false_positives,
        "false_negative_count": false_negatives,
        "true_negative_count": true_negatives,
    }


def _macro_f1(targets: np.ndarray, predictions: np.ndarray) -> float:
    actual = np.asarray(targets).reshape(-1)
    predicted = np.asarray(predictions).reshape(-1)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("targets and predictions must be non-empty equal vectors.")
    classes = np.union1d(actual, predicted)
    scores = []
    for class_value in classes:
        true_positive = np.count_nonzero(
            (predicted == class_value) & (actual == class_value)
        )
        false_positive = np.count_nonzero(
            (predicted == class_value) & (actual != class_value)
        )
        false_negative = np.count_nonzero(
            (predicted != class_value) & (actual == class_value)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def _expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int,
) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    actual = np.asarray(targets).reshape(-1)
    if values.ndim != 2 or values.shape[0] != actual.shape[0]:
        raise ValueError("probabilities and targets must have matching samples.")
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("ece_bins must be an integer of at least 2.")

    predicted = np.argmax(values, axis=1)
    confidence = np.max(values, axis=1)
    correctness = predicted == actual
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        selected = (
            (confidence >= lower) & (confidence <= upper)
            if index == 0
            else (confidence > lower) & (confidence <= upper)
        )
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(correctness[selected]))
                - float(np.mean(confidence[selected]))
            )
    return float(ece)


def _negative_log_likelihood(
    probabilities: np.ndarray, targets: np.ndarray
) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    actual = np.asarray(targets, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != actual.shape[0]:
        raise ValueError("probabilities and targets must have matching samples.")
    selected = values[np.arange(actual.shape[0]), actual]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def _detection_metrics(
    flags: np.ndarray,
    risk_scores: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    actual = np.asarray(labels, dtype=bool)
    metrics = _classification_metrics(flags, actual)
    metrics["prevalence"] = float(actual.mean()) if actual.size else 0.0
    metrics["evaluable"] = bool(np.any(actual) and np.any(~actual))
    if metrics["evaluable"]:
        metrics["non_evaluable_reason"] = None
        metrics["auroc"] = _binary_auroc(risk_scores, actual)
        metrics["average_precision"] = _average_precision(risk_scores, actual)
    else:
        metrics["non_evaluable_reason"] = (
            "no positive examples" if not np.any(actual) else "no negative examples"
        )
        metrics["auroc"] = None
        metrics["average_precision"] = None
    return metrics


def _flag_detection_metrics(
    flags: np.ndarray, labels: np.ndarray
) -> Dict[str, Any]:
    actual = np.asarray(labels, dtype=bool)
    metrics = _classification_metrics(flags, actual)
    metrics["prevalence"] = float(actual.mean()) if actual.size else 0.0
    metrics["evaluable"] = bool(np.any(actual) and np.any(~actual))
    metrics["non_evaluable_reason"] = (
        None
        if metrics["evaluable"]
        else (
            "no positive examples"
            if not np.any(actual)
            else "no negative examples"
        )
    )
    return metrics


@dataclass(frozen=True)
class MetaBEARSCalibration:
    """Thresholds and validation evidence used by an experiment run."""

    shortcut_threshold: float
    familiarity_threshold: float
    shortcut_policy: str
    shortcut_proxy: np.ndarray
    reference_distances: np.ndarray
    validation_report: MetaCognitiveReport
    shortcut_metrics: Mapping[str, Any]
    familiarity_validation_review_rate: float
    familiarity_target_review_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shortcut_threshold": float(self.shortcut_threshold),
            "familiarity_threshold": float(self.familiarity_threshold),
            "shortcut_policy": self.shortcut_policy,
            "shortcut_proxy_definition": (
                "task prediction correct and at least one concept prediction wrong"
            ),
            "shortcut_proxy_count": int(np.count_nonzero(self.shortcut_proxy)),
            "validation_samples": int(self.shortcut_proxy.shape[0]),
            "shortcut_metrics": dict(self.shortcut_metrics),
            "familiarity_target_review_rate": float(
                self.familiarity_target_review_rate
            ),
            "familiarity_validation_review_rate": float(
                self.familiarity_validation_review_rate
            ),
        }


def calibrate_metabears(
    validation_predictions: EnsemblePredictions,
    *,
    familiarity_validation_quantile: float = 0.05,
    shortcut_fallback_quantile: float = 0.95,
    perturbed_member_probabilities: Optional[np.ndarray] = None,
) -> MetaBEARSCalibration:
    """Calibrate both review paths using only an ID validation split.

    Shortcut risk uses the observable task-correct/concept-wrong mismatch as a
    validation proxy. If the validation split does not contain both proxy
    positives and negatives, a clearly reported upper-quantile fallback is
    used instead of pretending that supervised threshold selection succeeded.

    Familiarity has no labelled OOD validation samples in HalfMNIST. Its
    threshold is therefore the requested lower quantile of leave-one-out ID
    validation familiarity. The observed validation review rate is retained
    because ties can make it differ slightly from the requested quantile.
    """

    familiarity_quantile = _unit_interval(
        familiarity_validation_quantile,
        name="familiarity_validation_quantile",
    )
    shortcut_quantile = _unit_interval(
        shortcut_fallback_quantile,
        name="shortcut_fallback_quantile",
    )
    _require_experiment_fields(validation_predictions)

    reference_distances = ensemble_leave_one_out_reference_distances(
        validation_predictions.member_representations
    )
    provisional_report = build_meta_cognitive_report(
        validation_predictions.concept_member_probabilities,
        validation_predictions.label_member_probabilities,
        representation_distances=reference_distances,
        reference_distances=reference_distances,
        perturbed_member_probabilities=perturbed_member_probabilities,
    )
    proxy = shortcut_proxy_labels(validation_predictions)

    if np.any(proxy) and np.any(~proxy):
        selection = select_review_threshold(
            provisional_report.shortcut_risk,
            proxy,
        )
        shortcut_threshold = selection.threshold
        shortcut_policy = "validation_proxy_max_f1"
        shortcut_metrics: Mapping[str, Any] = selection.to_dict()
    else:
        shortcut_threshold = float(
            np.quantile(provisional_report.shortcut_risk, shortcut_quantile)
        )
        shortcut_policy = "validation_quantile_no_mixed_proxy_labels"
        shortcut_flags = provisional_report.shortcut_risk >= shortcut_threshold
        shortcut_metrics = _classification_metrics(shortcut_flags, proxy)

    validation_familiarity = familiarity_from_reference(
        reference_distances,
        reference_distances,
    )
    familiarity_threshold = float(
        np.quantile(validation_familiarity, familiarity_quantile)
    )
    validation_report = build_meta_cognitive_report(
        validation_predictions.concept_member_probabilities,
        validation_predictions.label_member_probabilities,
        representation_distances=reference_distances,
        reference_distances=reference_distances,
        perturbed_member_probabilities=perturbed_member_probabilities,
        review_threshold=shortcut_threshold,
        familiarity_threshold=familiarity_threshold,
    )

    return MetaBEARSCalibration(
        shortcut_threshold=shortcut_threshold,
        familiarity_threshold=familiarity_threshold,
        shortcut_policy=shortcut_policy,
        shortcut_proxy=proxy,
        reference_distances=reference_distances,
        validation_report=validation_report,
        shortcut_metrics=shortcut_metrics,
        familiarity_validation_review_rate=float(validation_report.ood_flag.mean()),
        familiarity_target_review_rate=familiarity_quantile,
    )


def build_calibrated_report(
    predictions: EnsemblePredictions,
    reference_predictions: EnsemblePredictions,
    calibration: MetaBEARSCalibration,
    *,
    perturbed_member_probabilities: Optional[np.ndarray] = None,
) -> MetaCognitiveReport:
    """Apply fixed validation-selected thresholds to a held-out split."""

    _require_experiment_fields(predictions)
    _require_experiment_fields(reference_predictions)
    distances = ensemble_nearest_reference_distances(
        predictions.member_representations,
        reference_predictions.member_representations,
    )
    return build_meta_cognitive_report(
        predictions.concept_member_probabilities,
        predictions.label_member_probabilities,
        representation_distances=distances,
        reference_distances=calibration.reference_distances,
        perturbed_member_probabilities=perturbed_member_probabilities,
        review_threshold=calibration.shortcut_threshold,
        familiarity_threshold=calibration.familiarity_threshold,
    )


def _prediction_metrics(
    predictions: EnsemblePredictions,
    report: MetaCognitiveReport,
    *,
    ece_bins: int,
) -> Dict[str, Any]:
    labels, concepts = _target_vectors(predictions)
    task_probabilities = predictions.label_member_probabilities.mean(axis=0)
    concept_probabilities = predictions.concept_member_probabilities.mean(axis=0)
    predicted_concepts = np.argmax(concept_probabilities, axis=-1).reshape(
        concepts.shape[0], -1
    )
    concept_probabilities = concept_probabilities.reshape(
        concepts.shape[0], concepts.shape[1], -1
    )
    valid = concepts >= 0
    concept_accuracy = (
        float(np.mean(predicted_concepts[valid] == concepts[valid]))
        if np.any(valid)
        else None
    )
    samples_with_concepts = np.any(valid, axis=1)
    exact_matches = np.all((predicted_concepts == concepts) | ~valid, axis=1)
    exact_concept_accuracy = (
        float(np.mean(exact_matches[samples_with_concepts]))
        if np.any(samples_with_concepts)
        else None
    )
    task_correct = report.predicted_label == labels
    accepted = ~report.review_flag
    reviewed = report.review_flag
    accepted_accuracy = (
        float(np.mean(task_correct[accepted])) if np.any(accepted) else None
    )
    reviewed_error_rate = (
        float(np.mean(~task_correct[reviewed])) if np.any(reviewed) else None
    )
    metrics = report.summary()
    metrics.update(
        {
            "task_accuracy": float(np.mean(task_correct)),
            "task_macro_f1": _macro_f1(labels, report.predicted_label),
            "task_ece": _expected_calibration_error(
                task_probabilities, labels, bins=ece_bins
            ),
            "task_nll": _negative_log_likelihood(task_probabilities, labels),
            "concept_accuracy": concept_accuracy,
            "concept_macro_f1": (
                _macro_f1(concepts[valid], predicted_concepts[valid])
                if np.any(valid)
                else None
            ),
            "concept_ece": (
                _expected_calibration_error(
                    concept_probabilities[valid], concepts[valid], bins=ece_bins
                )
                if np.any(valid)
                else None
            ),
            "concept_nll": (
                _negative_log_likelihood(
                    concept_probabilities[valid], concepts[valid]
                )
                if np.any(valid)
                else None
            ),
            "exact_concept_accuracy": exact_concept_accuracy,
            "coverage": float(np.mean(accepted)),
            "accepted_task_accuracy": accepted_accuracy,
            "accepted_task_risk": (
                1.0 - accepted_accuracy if accepted_accuracy is not None else None
            ),
            "reviewed_task_error_rate": reviewed_error_rate,
            "ece_bins": ece_bins,
        }
    )
    return metrics


def _binary_auroc(risk_scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(risk_scores, dtype=np.float64)
    actual = np.asarray(labels, dtype=bool)
    positives = int(np.count_nonzero(actual))
    negatives = actual.shape[0] - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires positive and negative samples.")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, scores.shape[0] + 1, dtype=np.float64)
    start = 0
    while start < sorted_scores.shape[0]:
        stop = start + 1
        while (
            stop < sorted_scores.shape[0]
            and sorted_scores[stop] == sorted_scores[start]
        ):
            stop += 1
        ranks[start:stop] = ranks[start:stop].mean()
        start = stop
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = original_ranks[actual].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _average_precision(risk_scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(risk_scores, dtype=np.float64)
    actual = np.asarray(labels, dtype=bool)
    positives = int(np.count_nonzero(actual))
    if positives == 0:
        raise ValueError("Average precision requires positive samples.")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = actual[order]
    average_precision = 0.0
    previous_recall = 0.0
    start = 0
    while start < sorted_scores.shape[0]:
        stop = start + 1
        while (
            stop < sorted_scores.shape[0]
            and sorted_scores[stop] == sorted_scores[start]
        ):
            stop += 1
        true_positives = int(np.count_nonzero(sorted_labels[:stop]))
        precision = true_positives / stop
        recall = true_positives / positives
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return float(average_precision)


def _bounded_loader(loader: Iterable[Any], max_batches: Optional[int]) -> Iterable[Any]:
    if max_batches is None:
        return loader
    if not isinstance(max_batches, int) or max_batches < 1:
        raise ValueError("max_batches must be a positive integer.")
    return islice(loader, max_batches)


def _write_predictions(path: Path, predictions: EnsemblePredictions) -> None:
    _require_experiment_fields(predictions)
    np.savez_compressed(
        path,
        concept_member_probabilities=predictions.concept_member_probabilities,
        label_member_probabilities=predictions.label_member_probabilities,
        member_representations=predictions.member_representations,
        labels=predictions.labels,
        concepts=predictions.concepts,
        batch_sizes=np.asarray(predictions.batch_sizes, dtype=np.int64),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _collect_intervention_predictions(
    models: Sequence[Any],
    loader: Iterable[Any],
    intervention: PredictionIntervention,
    *,
    apply_label_softmax: bool,
    representation_key: str,
    max_batches: Optional[int],
) -> EnsemblePredictions:
    transformed = collect_ensemble_predictions(
        models,
        _bounded_loader(loader, max_batches),
        apply_label_softmax=apply_label_softmax,
        representation_key=representation_key,
        image_transform=intervention.transform_images,
    )
    aligned_concepts = intervention.align_concept_probabilities(
        transformed.concept_member_probabilities
    )
    return EnsemblePredictions(
        concept_member_probabilities=aligned_concepts,
        label_member_probabilities=transformed.label_member_probabilities,
        member_representations=transformed.member_representations,
        labels=transformed.labels,
        concepts=transformed.concepts,
        batch_sizes=transformed.batch_sizes,
    )


def _as_perturbation_axis(predictions: EnsemblePredictions) -> np.ndarray:
    return predictions.concept_member_probabilities[:, :, np.newaxis, :, :]


def _paired_probability_js(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("Paired probabilities must have identical shapes.")
    midpoint = 0.5 * (left + right)
    safe_left = np.clip(left, 1e-12, 1.0)
    safe_right = np.clip(right, 1e-12, 1.0)
    safe_midpoint = np.clip(midpoint, 1e-12, 1.0)
    left_kl = np.sum(left * (np.log(safe_left) - np.log(safe_midpoint)), axis=-1)
    right_kl = np.sum(
        right * (np.log(safe_right) - np.log(safe_midpoint)), axis=-1
    )
    return np.clip(0.5 * (left_kl + right_kl) / np.log(2.0), 0.0, 1.0)


def _intervention_metrics(
    base: EnsemblePredictions,
    perturbed: EnsemblePredictions,
    report: MetaCognitiveReport,
) -> Dict[str, Any]:
    labels, concepts = _target_vectors(base)
    perturbed_labels, perturbed_targets = _target_vectors(perturbed)
    if not np.array_equal(labels, perturbed_labels) or not np.array_equal(
        concepts, perturbed_targets
    ):
        raise ValueError("Intervention and base predictions must share targets.")

    base_task_probabilities = base.label_member_probabilities.mean(axis=0)
    perturbed_task_probabilities = perturbed.label_member_probabilities.mean(axis=0)
    base_task = np.argmax(base_task_probabilities, axis=-1)
    perturbed_task = np.argmax(perturbed_task_probabilities, axis=-1)
    base_concepts = np.argmax(
        base.concept_member_probabilities.mean(axis=0), axis=-1
    ).reshape(concepts.shape)
    perturbed_concepts = np.argmax(
        perturbed.concept_member_probabilities.mean(axis=0), axis=-1
    ).reshape(concepts.shape)

    valid = concepts >= 0
    samples_with_concepts = np.any(valid, axis=1)
    base_exact = np.all((base_concepts == concepts) | ~valid, axis=1)
    perturbed_exact = np.all((perturbed_concepts == concepts) | ~valid, axis=1)
    base_exact &= samples_with_concepts
    perturbed_exact &= samples_with_concepts
    concept_predictions_match = np.all(
        (base_concepts == perturbed_concepts) | ~valid, axis=1
    )
    task_predictions_match = base_task == perturbed_task
    base_task_correct = base_task == labels
    perturbed_task_correct = perturbed_task == labels

    task_invariance_failure = base_task_correct & ~perturbed_task_correct
    concept_equivariance_break = base_exact & ~perturbed_exact
    semantic_instability_proxy = (
        base_task_correct
        & perturbed_task_correct
        & task_predictions_match
        & ~concept_predictions_match
    )

    return {
        "samples": int(labels.shape[0]),
        "task_prediction_invariance_rate": float(task_predictions_match.mean()),
        "base_task_accuracy": float(base_task_correct.mean()),
        "perturbed_task_accuracy": float(perturbed_task_correct.mean()),
        "task_invariance_failure_count": int(
            np.count_nonzero(task_invariance_failure)
        ),
        "mean_task_distribution_js": float(
            _paired_probability_js(
                base_task_probabilities, perturbed_task_probabilities
            ).mean()
        ),
        "concept_prediction_equivariance_rate": float(
            np.mean((base_concepts[valid] == perturbed_concepts[valid]))
        )
        if np.any(valid)
        else None,
        "exact_concept_prediction_equivariance_rate": float(
            concept_predictions_match[samples_with_concepts].mean()
        )
        if np.any(samples_with_concepts)
        else None,
        "base_exact_concept_accuracy": float(
            base_exact[samples_with_concepts].mean()
        )
        if np.any(samples_with_concepts)
        else None,
        "perturbed_exact_concept_accuracy": float(
            perturbed_exact[samples_with_concepts].mean()
        )
        if np.any(samples_with_concepts)
        else None,
        "concept_equivariance_break_count": int(
            np.count_nonzero(concept_equivariance_break)
        ),
        "semantic_instability_proxy_definition": (
            "clean and intervened task predictions are correct and unchanged, "
            "but at least one aligned concept prediction changes"
        ),
        "semantic_instability_proxy_count": int(
            np.count_nonzero(semantic_instability_proxy)
        ),
        "semantic_instability_detection": _detection_metrics(
            report.shortcut_flag,
            report.shortcut_risk,
            semantic_instability_proxy,
        ),
        "task_invariance_failure_shortcut_flags": _flag_detection_metrics(
            report.shortcut_flag, task_invariance_failure
        ),
        "task_invariance_failure_review_flags": _flag_detection_metrics(
            report.review_flag, task_invariance_failure
        ),
        "concept_equivariance_break_detection": _detection_metrics(
            report.shortcut_flag,
            report.shortcut_risk,
            concept_equivariance_break,
        ),
    }


def _attach_assignment_metrics(
    metrics: Dict[str, Any],
    intervention: PredictionIntervention,
    predictions: EnsemblePredictions,
) -> Dict[str, Any]:
    if intervention.assignment_metrics is None:
        return metrics
    if predictions.labels is None or predictions.batch_sizes is None:
        raise ValueError(
            "Intervention assignment metrics require labels and batch sizes."
        )
    metrics["input_assignment"] = dict(
        intervention.assignment_metrics(
            predictions.labels,
            predictions.batch_sizes,
        )
    )
    return metrics


@dataclass(frozen=True)
class MetaBEARSExperimentResult:
    """Artifacts returned after a validation/ID/OOD MetaBEARS run."""

    output_directory: Path
    calibration: MetaBEARSCalibration
    validation_report: MetaCognitiveReport
    id_test_report: MetaCognitiveReport
    ood_test_report: Optional[MetaCognitiveReport]
    summary: Mapping[str, Any]


def run_metabears_experiment(
    models: Sequence[Any],
    validation_loader: Iterable[Any],
    id_test_loader: Iterable[Any],
    *,
    ood_test_loader: Optional[Iterable[Any]] = None,
    output_directory: PathLike,
    familiarity_validation_quantile: float = 0.05,
    shortcut_fallback_quantile: float = 0.95,
    representation_key: str = "CS",
    apply_label_softmax: bool = False,
    max_batches: Optional[int] = None,
    ece_bins: int = 15,
    intervention: Optional[PredictionIntervention] = None,
    run_configuration: Optional[Mapping[str, Any]] = None,
    run_provenance: Optional[Mapping[str, Any]] = None,
) -> MetaBEARSExperimentResult:
    """Collect, calibrate, evaluate, and serialize a complete experiment."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    if not isinstance(ece_bins, int) or ece_bins < 2:
        raise ValueError("ece_bins must be an integer of at least 2.")

    validation_predictions = collect_ensemble_predictions(
        models,
        _bounded_loader(validation_loader, max_batches),
        apply_label_softmax=apply_label_softmax,
        representation_key=representation_key,
    )
    validation_intervention_predictions = None
    validation_perturbations = None
    if intervention is not None:
        validation_intervention_predictions = _collect_intervention_predictions(
            models,
            validation_loader,
            intervention,
            apply_label_softmax=apply_label_softmax,
            representation_key=representation_key,
            max_batches=max_batches,
        )
        validation_perturbations = _as_perturbation_axis(
            validation_intervention_predictions
        )
    calibration = calibrate_metabears(
        validation_predictions,
        familiarity_validation_quantile=familiarity_validation_quantile,
        shortcut_fallback_quantile=shortcut_fallback_quantile,
        perturbed_member_probabilities=validation_perturbations,
    )
    id_predictions = collect_ensemble_predictions(
        models,
        _bounded_loader(id_test_loader, max_batches),
        apply_label_softmax=apply_label_softmax,
        representation_key=representation_key,
    )
    id_intervention_predictions = None
    id_perturbations = None
    if intervention is not None:
        id_intervention_predictions = _collect_intervention_predictions(
            models,
            id_test_loader,
            intervention,
            apply_label_softmax=apply_label_softmax,
            representation_key=representation_key,
            max_batches=max_batches,
        )
        id_perturbations = _as_perturbation_axis(id_intervention_predictions)
    id_report = build_calibrated_report(
        id_predictions,
        validation_predictions,
        calibration,
        perturbed_member_probabilities=id_perturbations,
    )

    ood_predictions = None
    ood_report = None
    ood_intervention_predictions = None
    if ood_test_loader is not None:
        ood_predictions = collect_ensemble_predictions(
            models,
            _bounded_loader(ood_test_loader, max_batches),
            apply_label_softmax=apply_label_softmax,
            representation_key=representation_key,
        )
        ood_perturbations = None
        if intervention is not None:
            ood_intervention_predictions = _collect_intervention_predictions(
                models,
                ood_test_loader,
                intervention,
                apply_label_softmax=apply_label_softmax,
                representation_key=representation_key,
                max_batches=max_batches,
            )
            ood_perturbations = _as_perturbation_axis(
                ood_intervention_predictions
            )
        ood_report = build_calibrated_report(
            ood_predictions,
            validation_predictions,
            calibration,
            perturbed_member_probabilities=ood_perturbations,
        )

    calibration.validation_report.write_json(destination / "validation_report.json")
    calibration.validation_report.write_csv(destination / "validation_report.csv")
    np.savez_compressed(
        destination / "calibration.npz",
        shortcut_proxy=calibration.shortcut_proxy,
        reference_distances=calibration.reference_distances,
    )
    id_report.write_json(destination / "id_test_report.json")
    id_report.write_csv(destination / "id_test_report.csv")
    _write_predictions(
        destination / "validation_predictions.npz",
        validation_predictions,
    )
    _write_predictions(destination / "id_test_predictions.npz", id_predictions)
    if validation_intervention_predictions is not None:
        _write_predictions(
            destination / "validation_intervention_predictions.npz",
            validation_intervention_predictions,
        )
    if id_intervention_predictions is not None:
        _write_predictions(
            destination / "id_test_intervention_predictions.npz",
            id_intervention_predictions,
        )

    split_metrics: Dict[str, Any] = {
        "validation": _prediction_metrics(
            validation_predictions,
            calibration.validation_report,
            ece_bins=ece_bins,
        ),
        "id_test": _prediction_metrics(
            id_predictions, id_report, ece_bins=ece_bins
        ),
    }
    shortcut_detection = {
        "proxy_definition": (
            "task prediction correct and at least one concept prediction wrong"
        ),
        "validation": _detection_metrics(
            calibration.validation_report.shortcut_flag,
            calibration.validation_report.shortcut_risk,
            calibration.shortcut_proxy,
        ),
        "id_test": _detection_metrics(
            id_report.shortcut_flag,
            id_report.shortcut_risk,
            shortcut_proxy_labels(id_predictions),
        ),
    }

    intervention_summary = None
    if (
        intervention is not None
        and validation_intervention_predictions is not None
        and id_intervention_predictions is not None
    ):
        intervention_summary = {
            "name": intervention.name,
            "description": intervention.description,
            "concept_alignment": (
                "Transformed concept probabilities are mapped back to the "
                "original concept order before comparison."
            ),
            "validation": _attach_assignment_metrics(
                _intervention_metrics(
                    validation_predictions,
                    validation_intervention_predictions,
                    calibration.validation_report,
                ),
                intervention,
                validation_predictions,
            ),
            "id_test": _attach_assignment_metrics(
                _intervention_metrics(
                    id_predictions, id_intervention_predictions, id_report
                ),
                intervention,
                id_predictions,
            ),
        }

    ood_detection = None
    if ood_report is not None and ood_predictions is not None:
        ood_report.write_json(destination / "ood_test_report.json")
        ood_report.write_csv(destination / "ood_test_report.csv")
        _write_predictions(destination / "ood_test_predictions.npz", ood_predictions)
        split_metrics["ood_test"] = _prediction_metrics(
            ood_predictions, ood_report, ece_bins=ece_bins
        )
        if ood_intervention_predictions is not None:
            _write_predictions(
                destination / "ood_test_intervention_predictions.npz",
                ood_intervention_predictions,
            )
            if intervention_summary is not None:
                intervention_summary["ood_test"] = _attach_assignment_metrics(
                    _intervention_metrics(
                        ood_predictions,
                        ood_intervention_predictions,
                        ood_report,
                    ),
                    intervention,
                    ood_predictions,
                )

        combined_familiarity = np.concatenate(
            [id_report.neural_familiarity, ood_report.neural_familiarity]
        )
        ood_labels = np.concatenate(
            [
                np.zeros(id_report.predicted_label.shape[0], dtype=bool),
                np.ones(ood_report.predicted_label.shape[0], dtype=bool),
            ]
        )
        ood_flags = np.concatenate([id_report.ood_flag, ood_report.ood_flag])
        ood_detection = _classification_metrics(ood_flags, ood_labels)
        ood_detection.update(
            {
                "auroc": _binary_auroc(1.0 - combined_familiarity, ood_labels),
                "average_precision": _average_precision(
                    1.0 - combined_familiarity, ood_labels
                ),
            }
        )

    summary: Dict[str, Any] = {
        "configuration": _json_safe(dict(run_configuration or {})),
        "provenance": _json_safe(dict(run_provenance or {})),
        "calibration": calibration.to_dict(),
        "splits": split_metrics,
        "shortcut_detection": shortcut_detection,
        "ood_detection": ood_detection,
        "intervention": intervention_summary,
        "limitations": [
            (
                "Shortcut calibration uses a task-correct/concept-wrong proxy; "
                "it is not a causal shortcut label."
            ),
            (
                "Familiarity calibration uses only the lower tail of ID "
                "validation familiarity because HalfMNIST has no OOD "
                "validation split."
            ),
            (
                "The controlled intervention supplies behavioral failure "
                "labels; these strengthen causal analysis but do not by "
                "themselves prove the model's internal causal mechanism."
            )
            if intervention is not None
            else None,
        ],
    }
    summary["limitations"] = [
        item for item in summary["limitations"] if item is not None
    ]
    (destination / "run_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return MetaBEARSExperimentResult(
        output_directory=destination,
        calibration=calibration,
        validation_report=calibration.validation_report,
        id_test_report=id_report,
        ood_test_report=ood_report,
        summary=summary,
    )
