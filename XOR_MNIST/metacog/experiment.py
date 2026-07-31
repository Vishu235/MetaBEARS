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
        review_threshold=calibration.shortcut_threshold,
        familiarity_threshold=calibration.familiarity_threshold,
    )


def _prediction_metrics(
    predictions: EnsemblePredictions,
    report: MetaCognitiveReport,
) -> Dict[str, Any]:
    labels, concepts = _target_vectors(predictions)
    predicted_concepts = np.argmax(
        predictions.concept_member_probabilities.mean(axis=0), axis=-1
    ).reshape(concepts.shape[0], -1)
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
    metrics = report.summary()
    metrics.update(
        {
            "task_accuracy": float(np.mean(report.predicted_label == labels)),
            "concept_accuracy": concept_accuracy,
            "exact_concept_accuracy": exact_concept_accuracy,
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
    run_configuration: Optional[Mapping[str, Any]] = None,
) -> MetaBEARSExperimentResult:
    """Collect, calibrate, evaluate, and serialize a complete experiment."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    validation_predictions = collect_ensemble_predictions(
        models,
        _bounded_loader(validation_loader, max_batches),
        apply_label_softmax=apply_label_softmax,
        representation_key=representation_key,
    )
    calibration = calibrate_metabears(
        validation_predictions,
        familiarity_validation_quantile=familiarity_validation_quantile,
        shortcut_fallback_quantile=shortcut_fallback_quantile,
    )
    id_predictions = collect_ensemble_predictions(
        models,
        _bounded_loader(id_test_loader, max_batches),
        apply_label_softmax=apply_label_softmax,
        representation_key=representation_key,
    )
    id_report = build_calibrated_report(
        id_predictions,
        validation_predictions,
        calibration,
    )

    ood_predictions = None
    ood_report = None
    if ood_test_loader is not None:
        ood_predictions = collect_ensemble_predictions(
            models,
            _bounded_loader(ood_test_loader, max_batches),
            apply_label_softmax=apply_label_softmax,
            representation_key=representation_key,
        )
        ood_report = build_calibrated_report(
            ood_predictions,
            validation_predictions,
            calibration,
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

    split_metrics: Dict[str, Any] = {
        "validation": _prediction_metrics(
            validation_predictions, calibration.validation_report
        ),
        "id_test": _prediction_metrics(id_predictions, id_report),
    }
    ood_detection = None
    if ood_report is not None and ood_predictions is not None:
        ood_report.write_json(destination / "ood_test_report.json")
        ood_report.write_csv(destination / "ood_test_report.csv")
        _write_predictions(destination / "ood_test_predictions.npz", ood_predictions)
        split_metrics["ood_test"] = _prediction_metrics(ood_predictions, ood_report)

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
        "calibration": calibration.to_dict(),
        "splits": split_metrics,
        "ood_detection": ood_detection,
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
        ],
    }
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
