"""Validation-only MiniKandinsky representation selection for OOD scoring."""

import argparse
import json
from itertools import islice
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from .experiment import (
    _average_precision,
    _binary_auroc,
    _fit_representation_normalizer,
    _normalize_prediction_representations,
)
from .familiarity import familiarity_from_reference
from .integration import (
    EnsemblePredictions,
    collect_ensemble_predictions,
    ensemble_leave_one_out_reference_distances,
    ensemble_nearest_reference_distances,
)
from .minikandinsky import (
    ImageTransformLoader,
    MiniKandinskyTargetLoader,
    desaturate_minikandinsky_palette,
)
from .minikandinsky_runner import (
    _collect_provenance,
    _enable_legacy_imports,
    discover_checkpoint_paths,
    load_ensemble,
)
from .thresholds import select_review_threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a MiniKandinsky OOD representation using ID and controlled "
            "OOD validation data only. No test split is evaluated."
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
    parser.add_argument("--training-entropy", action="store_true")
    parser.add_argument("--training-entropy-weight", type=float, default=0.0)
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--splitted", action="store_true")
    parser.add_argument(
        "--ensemble-checkpoints",
        nargs="+",
        required=True,
        help="Two or more checkpoint paths from one training condition.",
    )
    parser.add_argument("--checkpoint-seeds", nargs="+", type=int, default=[0, 10, 20])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--representation-keys",
        nargs="+",
        choices=["CS", "pCS"],
        default=["CS", "pCS"],
    )
    parser.add_argument(
        "--normalizations",
        nargs="+",
        choices=["none", "zscore", "l2", "zscore_l2"],
        default=["none", "zscore", "l2", "zscore_l2"],
    )
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


def _validation_metrics(predictions: EnsemblePredictions) -> Dict[str, Any]:
    labels = np.asarray(predictions.labels, dtype=np.int64).reshape(-1)
    concepts = np.asarray(predictions.concepts, dtype=np.int64).reshape(
        labels.shape[0], -1
    )
    task_predictions = np.argmax(
        predictions.label_member_probabilities.mean(axis=0), axis=-1
    )
    concept_predictions = np.argmax(
        predictions.concept_member_probabilities.mean(axis=0), axis=-1
    ).reshape(concepts.shape)
    valid = concepts >= 0
    exact = np.all((concept_predictions == concepts) | ~valid, axis=1)
    samples_with_concepts = np.any(valid, axis=1)
    return {
        "samples": int(labels.shape[0]),
        "task_accuracy": float(np.mean(task_predictions == labels)),
        "concept_accuracy": (
            float(np.mean(concept_predictions[valid] == concepts[valid]))
            if np.any(valid)
            else None
        ),
        "exact_concept_accuracy": (
            float(np.mean(exact[samples_with_concepts]))
            if np.any(samples_with_concepts)
            else None
        ),
    }


def _candidate_metrics(
    validation_predictions: EnsemblePredictions,
    ood_validation_predictions: EnsemblePredictions,
    *,
    representation_key: str,
    normalization: str,
    max_false_review_rate: float,
) -> Dict[str, Any]:
    normalizer = _fit_representation_normalizer(
        validation_predictions, normalization
    )
    normalized_id = _normalize_prediction_representations(
        validation_predictions, normalizer
    )
    normalized_ood = _normalize_prediction_representations(
        ood_validation_predictions, normalizer
    )
    reference_distances = ensemble_leave_one_out_reference_distances(
        normalized_id.member_representations
    )
    ood_distances = ensemble_nearest_reference_distances(
        normalized_ood.member_representations,
        normalized_id.member_representations,
    )
    id_familiarity = familiarity_from_reference(
        reference_distances, reference_distances
    )
    ood_familiarity = familiarity_from_reference(
        ood_distances, reference_distances
    )
    familiarity = np.concatenate([id_familiarity, ood_familiarity])
    ood_labels = np.concatenate(
        [
            np.zeros(id_familiarity.shape[0], dtype=bool),
            np.ones(ood_familiarity.shape[0], dtype=bool),
        ]
    )
    risk = 1.0 - familiarity
    threshold = select_review_threshold(
        familiarity,
        ood_labels,
        higher_is_riskier=False,
        max_false_review_rate=max_false_review_rate,
    )
    return {
        "representation_key": representation_key,
        "normalization": normalization,
        "id_samples": int(id_familiarity.shape[0]),
        "ood_samples": int(ood_familiarity.shape[0]),
        "mean_id_distance": float(np.mean(reference_distances)),
        "mean_ood_distance": float(np.mean(ood_distances)),
        "mean_id_familiarity": float(np.mean(id_familiarity)),
        "mean_ood_familiarity": float(np.mean(ood_familiarity)),
        "auroc": _binary_auroc(risk, ood_labels),
        "average_precision": _average_precision(risk, ood_labels),
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
        "max_false_review_rate",
        "minimum_auroc",
        "minimum_average_precision",
        "minimum_recall",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must lie within [0, 1].")
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

    candidates = []
    validation_metrics = None
    for representation_key in args.representation_keys:
        validation_predictions = collect_ensemble_predictions(
            ensemble,
            _bounded_loader(validation_loader, args.max_batches),
            representation_key=representation_key,
        )
        ood_validation_predictions = collect_ensemble_predictions(
            ensemble,
            _bounded_loader(ood_validation_loader, args.max_batches),
            representation_key=representation_key,
        )
        if validation_metrics is None:
            validation_metrics = _validation_metrics(validation_predictions)
        for normalization in args.normalizations:
            candidate = _candidate_metrics(
                validation_predictions,
                ood_validation_predictions,
                representation_key=representation_key,
                normalization=normalization,
                max_false_review_rate=args.max_false_review_rate,
            )
            candidate["acceptance_criteria_satisfied"] = _accepted(
                candidate, args
            )
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
    command_arguments = sys.argv[1:] if argv is None else list(argv)
    result = {
        "protocol": {
            "name": "minikandinsky_validation_only_representation_selection_v2",
            "selection_split": "ID validation plus desaturated-palette OOD validation",
            "test_split_evaluated": False,
            "risk_direction": "larger nearest-neighbour distance is riskier",
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
            "training_entropy": args.training_entropy,
            "training_entropy_weight": args.training_entropy_weight,
            "ensemble_members": len(checkpoint_paths),
            "checkpoints": [str(path) for path in checkpoint_paths],
            "representation_keys": list(args.representation_keys),
            "normalizations": list(args.normalizations),
            "max_batches": args.max_batches,
            "command_arguments": command_arguments,
        },
        "provenance": _collect_provenance(
            checkpoint_paths, args.provenance_cache
        ),
        "validation_prediction_metrics": validation_metrics,
        "candidate_count": len(candidates),
        "candidates": ordered,
        "best_observed_candidate": ordered[0],
        "selected_candidate": accepted[0] if accepted else None,
        "selection_status": "accepted" if accepted else "no_usable_candidate",
        "limitations": [
            "This exploratory selection uses controlled OOD validation only.",
            "No test loader is iterated and no held-out OOD result is reported.",
            "A new OOD test transform must be frozen after selection and before evaluation.",
        ],
    }
    output_path = destination / "representation_selection.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Validation-only representation selection: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
