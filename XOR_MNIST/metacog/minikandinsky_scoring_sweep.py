"""Validation-only MiniKandinsky risk-scorer selection for MetaBEARS."""

import argparse
import json
from itertools import islice
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .experiment import (
    _average_precision,
    _binary_auroc,
    _fit_representation_normalizer,
    _normalize_prediction_representations,
)
from .integration import (
    EnsemblePredictions,
    collect_ensemble_predictions,
    ensemble_nearest_reference_distances,
)
from .minikandinsky import (
    ImageTransformLoader,
    MiniKandinskyTargetLoader,
    desaturate_minikandinsky_palette,
)
from .minikandinsky_representation_sweep import _validation_metrics
from .minikandinsky_runner import (
    _collect_provenance,
    _enable_legacy_imports,
    discover_checkpoint_paths,
    load_ensemble,
)
from .thresholds import select_review_threshold


SCORERS = (
    "nearest",
    "shrinkage_mahalanobis",
    "class_conditional_mahalanobis",
    "class_conditional_disagreement_fusion",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare predeclared MiniKandinsky risk scorers on ID and "
            "controlled OOD validation data only. No test split is evaluated."
        )
    )
    parser.add_argument("--dataset", default="minikandinsky")
    parser.add_argument("--model", default="minikanddpl")
    parser.add_argument("--task", default="mini_patterns_bombazza")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16)
    parser.add_argument("--finetuning", type=int, default=0)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--c-sup", "--c_sup", type=float, default=1.0)
    parser.add_argument("--w-c", "--w_c", type=float, default=10.0)
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--splitted", action="store_true")
    parser.add_argument("--ensemble-checkpoints", nargs="+", required=True)
    parser.add_argument("--checkpoint-seeds", nargs="+", type=int, default=[0, 10, 20])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--representation-key", choices=["CS", "pCS"], default="CS")
    parser.add_argument(
        "--normalization",
        choices=["none", "zscore", "l2", "zscore_l2"],
        default="zscore_l2",
    )
    parser.add_argument("--scorers", nargs="+", choices=SCORERS, default=list(SCORERS))
    parser.add_argument("--cross-fit-folds", type=int, default=5)
    parser.add_argument("--shrinkage", type=float, default=0.10)
    parser.add_argument("--max-false-review-rate", type=float, default=0.05)
    parser.add_argument("--minimum-auroc", type=float, default=0.70)
    parser.add_argument("--minimum-average-precision", type=float, default=0.70)
    parser.add_argument("--minimum-recall", type=float, default=0.50)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--provenance-cache", default=None)
    return parser


def _bounded_loader(
    loader: Iterable[Any], max_batches: Optional[int]
) -> Iterable[Any]:
    return loader if max_batches is None else islice(loader, max_batches)


def _fold_assignments(
    samples: int,
    folds: int,
    seed: int,
    *,
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    if folds < 2:
        raise ValueError("cross_fit_folds must be at least two.")
    if samples < 2 * folds:
        raise ValueError("At least two samples per cross-fitting fold are required.")
    assignments = np.empty(samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if labels is None:
        order = rng.permutation(samples)
        assignments[order] = np.arange(samples) % folds
        return assignments

    strata = np.asarray(labels, dtype=np.int64).reshape(-1)
    if strata.shape[0] != samples:
        raise ValueError("labels and fold assignments must contain equal samples.")
    for class_index in np.unique(strata):
        class_indices = np.flatnonzero(strata == class_index)
        if class_indices.shape[0] < 2 * folds:
            raise ValueError(
                f"Class {class_index} needs at least two samples per fold."
            )
        order = rng.permutation(class_indices)
        assignments[order] = np.arange(order.shape[0]) % folds
    return assignments


def _fit_shrinkage_statistics(
    representations: np.ndarray,
    *,
    labels: Optional[np.ndarray],
    shrinkage: float,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], ...]:
    values = np.asarray(representations, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("representations must be [members, samples, features].")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie within [0, 1].")
    if labels is None:
        target_labels = np.zeros(values.shape[1], dtype=np.int64)
    else:
        target_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if target_labels.shape[0] != values.shape[1]:
            raise ValueError("labels and representations must contain equal samples.")

    fitted = []
    for member_values in values:
        member_models: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for class_index in np.unique(target_labels):
            class_values = member_values[target_labels == class_index]
            if class_values.shape[0] < 2:
                raise ValueError(
                    f"Class {class_index} needs at least two reference samples."
                )
            mean = class_values.mean(axis=0)
            centered = class_values - mean
            covariance = centered.T @ centered / (class_values.shape[0] - 1)
            diagonal = np.diag(np.diag(covariance))
            covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
            average_variance = float(np.trace(covariance)) / covariance.shape[0]
            ridge = max(average_variance * 1e-6, 1e-8)
            eigenvalues, eigenvectors = np.linalg.eigh(
                covariance + ridge * np.eye(covariance.shape[0])
            )
            precision = (
                eigenvectors * (1.0 / np.maximum(eigenvalues, ridge))
            ) @ eigenvectors.T
            member_models[int(class_index)] = (mean, precision)
        fitted.append(member_models)
    return tuple(fitted)


def _mahalanobis_scores(
    representations: np.ndarray,
    models: Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], ...],
    *,
    predicted_classes: Optional[np.ndarray],
) -> np.ndarray:
    values = np.asarray(representations, dtype=np.float64)
    classes = (
        np.zeros(values.shape[1], dtype=np.int64)
        if predicted_classes is None
        else np.asarray(predicted_classes, dtype=np.int64).reshape(-1)
    )
    if classes.shape[0] != values.shape[1]:
        raise ValueError("predicted_classes and representations must align.")

    member_scores = np.empty(values.shape[:2], dtype=np.float64)
    for member_index, (member_values, member_models) in enumerate(
        zip(values, models)
    ):
        for class_index in np.unique(classes):
            if int(class_index) not in member_models:
                raise ValueError(f"No fitted statistics for class {class_index}.")
            mask = classes == class_index
            mean, precision = member_models[int(class_index)]
            difference = member_values[mask] - mean
            squared = np.einsum(
                "ni,ij,nj->n", difference, precision, difference, optimize=True
            )
            member_scores[member_index, mask] = np.sqrt(np.maximum(squared, 0.0))
    return member_scores.mean(axis=0)


def _predicted_labels(predictions: EnsemblePredictions) -> np.ndarray:
    return np.argmax(
        predictions.label_member_probabilities.mean(axis=0), axis=-1
    ).astype(np.int64)


def _cross_fitted_distances(
    validation_predictions: EnsemblePredictions,
    ood_predictions: EnsemblePredictions,
    *,
    scorer: str,
    folds: int,
    seed: int,
    shrinkage: float,
) -> Tuple[np.ndarray, np.ndarray]:
    id_values = np.asarray(
        validation_predictions.member_representations, dtype=np.float64
    )
    ood_values = np.asarray(ood_predictions.member_representations, dtype=np.float64)
    labels = np.asarray(validation_predictions.labels, dtype=np.int64).reshape(-1)
    id_predicted = _predicted_labels(validation_predictions)
    ood_predicted = _predicted_labels(ood_predictions)
    class_conditional = scorer.startswith("class_conditional")
    assignments = _fold_assignments(
        id_values.shape[1],
        folds,
        seed,
        labels=labels if class_conditional else None,
    )
    id_scores = np.empty(id_values.shape[1], dtype=np.float64)
    ood_scores = np.zeros(ood_values.shape[1], dtype=np.float64)

    for fold in range(folds):
        query_mask = assignments == fold
        reference_mask = ~query_mask
        references = id_values[:, reference_mask]
        if scorer == "nearest":
            id_scores[query_mask] = ensemble_nearest_reference_distances(
                id_values[:, query_mask], references
            )
            ood_scores += ensemble_nearest_reference_distances(
                ood_values, references
            ) / folds
            continue

        models = _fit_shrinkage_statistics(
            references,
            labels=labels[reference_mask] if class_conditional else None,
            shrinkage=shrinkage,
        )
        id_scores[query_mask] = _mahalanobis_scores(
            id_values[:, query_mask],
            models,
            predicted_classes=id_predicted[query_mask] if class_conditional else None,
        )
        ood_scores += _mahalanobis_scores(
            ood_values,
            models,
            predicted_classes=ood_predicted if class_conditional else None,
        ) / folds
    return id_scores, ood_scores


def _label_disagreement(predictions: EnsemblePredictions) -> np.ndarray:
    probabilities = np.clip(
        np.asarray(predictions.label_member_probabilities, dtype=np.float64),
        1e-12,
        1.0,
    )
    mean_probabilities = probabilities.mean(axis=0)
    entropy_of_mean = -np.sum(
        mean_probabilities * np.log(mean_probabilities), axis=-1
    )
    mean_entropy = -np.mean(
        np.sum(probabilities * np.log(probabilities), axis=-1), axis=0
    )
    normalizer = np.log(probabilities.shape[-1])
    return np.maximum(entropy_of_mean - mean_entropy, 0.0) / normalizer


def _empirical_percentiles(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    query = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.searchsorted(ordered, query, side="right") / ordered.shape[0]


def _candidate(
    name: str,
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    *,
    max_false_review_rate: float,
) -> Dict[str, Any]:
    risk = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate(
        [
            np.zeros(id_scores.shape[0], dtype=bool),
            np.ones(ood_scores.shape[0], dtype=bool),
        ]
    )
    threshold = select_review_threshold(
        risk,
        labels,
        higher_is_riskier=True,
        max_false_review_rate=max_false_review_rate,
    )
    return {
        "scorer": name,
        "id_samples": int(id_scores.shape[0]),
        "ood_samples": int(ood_scores.shape[0]),
        "mean_id_risk": float(np.mean(id_scores)),
        "mean_ood_risk": float(np.mean(ood_scores)),
        "auroc": _binary_auroc(risk, labels),
        "average_precision": _average_precision(risk, labels),
        "threshold_selection": threshold.to_dict(),
    }


def _accepted(candidate: Dict[str, Any], args: argparse.Namespace) -> bool:
    threshold = candidate["threshold_selection"]
    return bool(
        candidate["auroc"] >= args.minimum_auroc
        and candidate["average_precision"] >= args.minimum_average_precision
        and threshold["recall"] >= args.minimum_recall
        and threshold["false_review_rate"] <= args.max_false_review_rate
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for name in (
        "shrinkage",
        "max_false_review_rate",
        "minimum_auroc",
        "minimum_average_precision",
        "minimum_recall",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must lie within [0, 1].")
    if args.cross_fit_folds < 2:
        raise SystemExit("--cross-fit-folds must be at least two.")
    if args.max_batches is not None and args.max_batches < 1:
        raise SystemExit("--max-batches must be positive.")

    _enable_legacy_imports()
    try:
        from datasets import get_dataset
        from utils.conf import set_random_seed
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"The BEARS runtime dependency '{error.name}' is missing."
        ) from error

    checkpoint_paths = discover_checkpoint_paths(args)
    set_random_seed(args.seed)
    dataset = get_dataset(args)
    _, raw_validation_loader, _ = dataset.get_data_loaders()
    validation_loader = MiniKandinskyTargetLoader(raw_validation_loader)
    ood_validation_loader = ImageTransformLoader(
        validation_loader, desaturate_minikandinsky_palette
    )
    ensemble = load_ensemble(dataset, args, checkpoint_paths)
    validation_predictions = collect_ensemble_predictions(
        ensemble,
        _bounded_loader(validation_loader, args.max_batches),
        representation_key=args.representation_key,
    )
    ood_predictions = collect_ensemble_predictions(
        ensemble,
        _bounded_loader(ood_validation_loader, args.max_batches),
        representation_key=args.representation_key,
    )
    normalizer = _fit_representation_normalizer(
        validation_predictions, args.normalization
    )
    normalized_id = _normalize_prediction_representations(
        validation_predictions, normalizer
    )
    normalized_ood = _normalize_prediction_representations(
        ood_predictions, normalizer
    )

    score_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for scorer in args.scorers:
        distance_scorer = (
            "class_conditional_mahalanobis"
            if scorer == "class_conditional_disagreement_fusion"
            else scorer
        )
        if distance_scorer not in score_pairs:
            score_pairs[distance_scorer] = _cross_fitted_distances(
                normalized_id,
                normalized_ood,
                scorer=distance_scorer,
                folds=args.cross_fit_folds,
                seed=args.seed,
                shrinkage=args.shrinkage,
            )

    candidates = []
    for scorer in args.scorers:
        if scorer == "class_conditional_disagreement_fusion":
            id_distance, ood_distance = score_pairs[
                "class_conditional_mahalanobis"
            ]
            id_disagreement = _label_disagreement(validation_predictions)
            ood_disagreement = _label_disagreement(ood_predictions)
            id_scores = 0.5 * _empirical_percentiles(
                id_distance, id_distance
            ) + 0.5 * _empirical_percentiles(id_disagreement, id_disagreement)
            ood_scores = 0.5 * _empirical_percentiles(
                id_distance, ood_distance
            ) + 0.5 * _empirical_percentiles(id_disagreement, ood_disagreement)
        else:
            id_scores, ood_scores = score_pairs[scorer]
        candidate = _candidate(
            scorer,
            id_scores,
            ood_scores,
            max_false_review_rate=args.max_false_review_rate,
        )
        candidate["acceptance_criteria_satisfied"] = _accepted(candidate, args)
        candidates.append(candidate)

    ordered = sorted(
        candidates,
        key=lambda item: (
            item["auroc"],
            item["average_precision"],
            item["threshold_selection"]["recall"],
        ),
        reverse=True,
    )
    accepted = [item for item in ordered if item["acceptance_criteria_satisfied"]]
    destination = Path(args.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": {
            "name": "minikandinsky_fixed_representation_scoring_selection_v3",
            "selection_split": "ID validation plus desaturated-palette OOD validation",
            "test_split_evaluated": False,
            "representation_frozen_from_v2": {
                "key": args.representation_key,
                "normalization": args.normalization,
            },
            "cross_fitting": (
                "ID scores exclude their own fold; OOD scores average all fold models"
            ),
            "class_conditional_policy": (
                "fit with ID validation labels and select the class model with the "
                "ensemble-predicted task label"
            ),
            "fusion_policy": (
                "equal weight on ID-percentile class-conditional distance and "
                "label Jensen-Shannon disagreement"
            ),
            "selection_metric": "validation OOD AUROC, then average precision, then recall",
            "acceptance_criteria": {
                "minimum_auroc": args.minimum_auroc,
                "minimum_average_precision": args.minimum_average_precision,
                "minimum_recall_at_false_review_budget": args.minimum_recall,
                "max_false_review_rate": args.max_false_review_rate,
            },
        },
        "configuration": {
            "dataset": args.dataset,
            "model": args.model,
            "task": args.task,
            "training_concept_supervision": args.c_sup,
            "training_concept_weight": args.w_c,
            "ensemble_members": len(checkpoint_paths),
            "checkpoints": [str(path) for path in checkpoint_paths],
            "representation_key": args.representation_key,
            "normalization": args.normalization,
            "scorers": list(args.scorers),
            "cross_fit_folds": args.cross_fit_folds,
            "shrinkage": args.shrinkage,
            "max_batches": args.max_batches,
            "command_arguments": sys.argv[1:] if argv is None else list(argv),
        },
        "provenance": _collect_provenance(
            checkpoint_paths, args.provenance_cache
        ),
        "validation_prediction_metrics": _validation_metrics(
            validation_predictions
        ),
        "candidate_count": len(ordered),
        "candidates": ordered,
        "best_observed_candidate": ordered[0],
        "selected_candidate": accepted[0] if accepted else None,
        "selection_status": "accepted" if accepted else "no_usable_candidate",
        "limitations": [
            "This exploratory selection uses controlled OOD validation only.",
            "The desaturated OOD validation shift is reused from v2 development.",
            "No test loader is iterated and no held-out OOD result is reported.",
            "The representation is fixed from v2; only the predeclared scorer changes.",
            "A new OOD test transform must be frozen before final evaluation.",
        ],
    }
    output_path = destination / "scoring_selection.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Validation-only scoring selection: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
