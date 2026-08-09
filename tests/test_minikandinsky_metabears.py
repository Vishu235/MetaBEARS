"""Tests for the MiniKandinsky MetaBEARS adapter and controls."""

from argparse import Namespace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from XOR_MNIST.metacog import EnsemblePredictions, collect_ensemble_predictions
from XOR_MNIST.metacog.minikandinsky import (
    MiniKandinskyModelAdapter,
    MiniKandinskyTargetLoader,
    align_cycled_minikandinsky_colors,
    align_permuted_minikandinsky_concepts,
    cycle_minikandinsky_palette,
    desaturate_minikandinsky_palette,
    pastelize_minikandinsky_palette,
    permute_minikandinsky_figures,
)
from XOR_MNIST.metacog.minikandinsky_runner import _collect_provenance
from XOR_MNIST.metacog.minikandinsky_representation_sweep import (
    _accepted,
    _candidate_metrics,
    _validation_metrics,
)
from XOR_MNIST.metacog.minikandinsky_scoring_sweep import (
    _cross_fitted_distances,
    _fit_shrinkage_statistics,
    _confidence_deficit,
    _label_disagreement,
    _load_and_validate_frozen_candidate,
    _mahalanobis_scores,
    _predictive_entropy,
)
from XOR_MNIST.utils.losses import KAND_Classification


def _categorical_probabilities(batch_size: int) -> np.ndarray:
    probabilities = np.full((batch_size, 18, 3), 0.025, dtype=np.float64)
    for sample in range(batch_size):
        for concept in range(18):
            probabilities[sample, concept, (sample + concept) % 3] = 0.95
    return probabilities


def _sweep_predictions(representations: np.ndarray) -> EnsemblePredictions:
    values = np.asarray(representations, dtype=np.float64)
    members, samples, _ = values.shape
    concept_probabilities = np.tile(
        np.array([0.9, 0.1]), (members, samples, 1, 1)
    )
    label_probabilities = np.tile(
        np.array([0.9, 0.1]), (members, samples, 1)
    )
    return EnsemblePredictions(
        concept_member_probabilities=concept_probabilities,
        label_member_probabilities=label_probabilities,
        member_representations=values,
        labels=np.zeros(samples, dtype=np.int64),
        concepts=np.zeros((samples, 1), dtype=np.int64),
        batch_sizes=(samples,),
    )


def _scoring_predictions(
    representations: np.ndarray,
    labels: np.ndarray,
    label_probabilities: np.ndarray,
) -> EnsemblePredictions:
    values = np.asarray(representations, dtype=np.float64)
    members, samples, _ = values.shape
    concept_probabilities = np.tile(
        np.array([0.9, 0.1]), (members, samples, 1, 1)
    )
    return EnsemblePredictions(
        concept_member_probabilities=concept_probabilities,
        label_member_probabilities=np.asarray(label_probabilities),
        member_representations=values,
        labels=np.asarray(labels, dtype=np.int64),
        concepts=np.zeros((samples, 1), dtype=np.int64),
        batch_sizes=(samples,),
    )


class FakeRawMiniKandinskyModel:
    def __init__(self, member_index: int) -> None:
        self.member_index = member_index
        self.evaluation_mode = False

    def eval(self) -> None:
        self.evaluation_mode = True

    def __call__(self, images: np.ndarray) -> dict:
        batch_size = images.shape[0]
        concepts = _categorical_probabilities(batch_size).reshape(
            batch_size, 3, 18
        )
        labels = np.tile(np.array([0.1, 0.9]), (batch_size, 1))
        representations = np.arange(
            batch_size * 54, dtype=np.float64
        ).reshape(batch_size, 3, 18)
        return {"pCS": concepts, "YS": labels, "CS": representations}


class MiniKandinskyAdapterTests(unittest.TestCase):
    def test_frozen_v4_ablation_preserves_negative_result(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "minikandinsky_results_freeze_v4.json"
        )
        freeze = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(freeze["status"], "frozen_negative_ablation")
        self.assertFalse(freeze["validation_protocol"]["test_split_evaluated"])
        self.assertFalse(
            freeze["conclusion"]["fusion_outperforms_uncertainty_baselines"]
        )
        operating_points = freeze["frozen_operating_points"]
        self.assertGreater(
            operating_points["predictive_entropy"]["auroc"],
            operating_points["class_conditional_disagreement_fusion"]["auroc"],
        )
        self.assertEqual(
            operating_points["predictive_entropy"]["auroc"],
            operating_points["confidence_deficit"]["auroc"],
        )
        self.assertTrue(
            freeze["integrity_checks"]["v3_fusion_reproduction_delta_zero"]
        )

    def test_frozen_v3_candidate_records_reproducible_configuration(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "minikandinsky_results_freeze_v3.json"
        )
        freeze = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(freeze["status"], "frozen_validation_candidate")
        self.assertFalse(freeze["validation_protocol"]["test_split_evaluated"])
        self.assertEqual(
            freeze["frozen_configuration"]["scorer"],
            "class_conditional_disagreement_fusion",
        )
        self.assertEqual(freeze["frozen_configuration"]["threshold"], 0.8185)
        self.assertEqual(
            set(freeze["run_provenance"]["checkpoint_sha256"]),
            {"seed_0", "seed_10", "seed_20"},
        )
        self.assertTrue(
            all(
                len(artifact["sha256"]) == 64
                for artifact in freeze["source_artifacts"]
            )
        )

        args = Namespace(
            frozen_candidate_protocol=str(path),
            representation_key="CS",
            normalization="zscore_l2",
            cross_fit_folds=5,
            shrinkage=0.1,
        )
        provenance = {
            "checkpoints": [
                {"sha256": value}
                for value in freeze["run_provenance"][
                    "checkpoint_sha256"
                ].values()
            ],
            "dataset_artifacts": [
                {"sha256": freeze["run_provenance"]["dataset_sha256"]}
            ],
        }
        loaded = _load_and_validate_frozen_candidate(args, provenance)
        self.assertEqual(loaded["freeze_id"], freeze["freeze_id"])
        provenance["checkpoints"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Checkpoint fingerprints"):
            _load_and_validate_frozen_candidate(args, provenance)

    def test_cross_fitted_scoring_distinguishes_distant_ood_samples(self) -> None:
        member_one = np.array(
            [
                [-0.2, 0.0], [0.0, 0.2], [0.1, -0.1], [0.2, 0.1],
                [2.8, 3.0], [3.0, 3.2], [3.1, 2.9], [3.2, 3.1],
            ]
        )
        id_values = np.stack([member_one, member_one + 0.05])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        id_probabilities = np.tile(np.eye(2)[labels], (2, 1, 1))
        ood_values = np.array(
            [[[8.0, 8.0], [9.0, 9.0]], [[8.1, 8.1], [9.1, 9.1]]]
        )
        ood_labels = np.array([0, 1])
        ood_probabilities = np.tile(np.eye(2)[ood_labels], (2, 1, 1))
        id_predictions = _scoring_predictions(
            id_values, labels, id_probabilities
        )
        ood_predictions = _scoring_predictions(
            ood_values, ood_labels, ood_probabilities
        )

        id_scores, ood_scores = _cross_fitted_distances(
            id_predictions,
            ood_predictions,
            scorer="class_conditional_mahalanobis",
            folds=2,
            seed=0,
            shrinkage=0.1,
        )

        self.assertGreater(ood_scores.mean(), id_scores.mean())

    def test_shrinkage_scores_and_disagreement_are_member_preserving(self) -> None:
        references = np.array(
            [
                [[0.0, 0.0], [0.2, 0.1], [3.0, 3.0], [3.2, 3.1]],
                [[0.1, 0.0], [0.3, 0.1], [3.1, 3.0], [3.3, 3.1]],
            ]
        )
        labels = np.array([0, 0, 1, 1])
        models = _fit_shrinkage_statistics(
            references, labels=labels, shrinkage=0.1
        )
        far = np.array([[[8.0, 8.0]], [[8.1, 8.0]]])
        scores = _mahalanobis_scores(
            far, models, predicted_classes=np.array([1])
        )
        self.assertGreater(scores[0], 1.0)

        probabilities = np.array(
            [
                [[0.9, 0.1], [0.9, 0.1]],
                [[0.9, 0.1], [0.1, 0.9]],
            ]
        )
        predictions = _scoring_predictions(
            np.zeros((2, 2, 1)), np.zeros(2), probabilities
        )
        disagreement = _label_disagreement(predictions)
        predictive_entropy = _predictive_entropy(predictions)
        confidence_deficit = _confidence_deficit(predictions)
        self.assertAlmostEqual(disagreement[0], 0.0)
        self.assertGreater(disagreement[1], disagreement[0])
        self.assertGreater(predictive_entropy[1], predictive_entropy[0])
        self.assertGreater(confidence_deficit[1], confidence_deficit[0])

    def test_validation_only_sweep_separates_distant_ood_representations(self) -> None:
        id_values = np.array(
            [[[0.0], [1.0], [3.0], [6.0]], [[0.2], [1.2], [3.2], [6.2]]]
        )
        ood_values = np.array(
            [[[20.0], [22.0], [24.0], [26.0]], [[20.2], [22.2], [24.2], [26.2]]]
        )
        id_predictions = _sweep_predictions(id_values)
        ood_predictions = _sweep_predictions(ood_values)

        candidate = _candidate_metrics(
            id_predictions,
            ood_predictions,
            representation_key="CS",
            normalization="none",
            max_false_review_rate=0.30,
        )

        self.assertEqual(candidate["auroc"], 0.875)
        self.assertEqual(candidate["average_precision"], 0.8)
        self.assertEqual(candidate["threshold_selection"]["recall"], 1.0)
        self.assertTrue(
            _accepted(
                candidate,
                Namespace(
                    minimum_auroc=0.8,
                    minimum_average_precision=0.8,
                    minimum_recall=0.9,
                    max_false_review_rate=0.30,
                ),
            )
        )
        self.assertEqual(_validation_metrics(id_predictions)["task_accuracy"], 1.0)

    def test_minikandinsky_task_loss_is_differentiable_without_concept_loss(self) -> None:
        task_probabilities = torch.tensor(
            [[0.8, 0.2], [0.3, 0.7]],
            dtype=torch.float32,
            requires_grad=True,
        )
        labels = torch.tensor([[0, 1, 0], [1, 0, 1]])

        loss, _ = KAND_Classification(
            {"YS": task_probabilities, "LABELS": labels},
            Namespace(model="minikanddpl", task="mini_patterns_bombazza"),
        )
        loss.backward()

        self.assertTrue(loss.requires_grad)
        self.assertIsNotNone(task_probabilities.grad)

    def test_unfrozen_runner_passes_explicit_null_protocol(self) -> None:
        with patch(
            "XOR_MNIST.metacog.minikandinsky_runner.collect_run_provenance",
            return_value={"protocol": None},
        ) as collect:
            result = _collect_provenance([], None)

        self.assertEqual(result, {"protocol": None})
        self.assertIsNone(collect.call_args.kwargs["protocol"])

    def test_model_and_target_adapters_feed_generic_collector(self) -> None:
        labels = np.array([[2, 2, 2, 1], [0, 1, 2, 1]], dtype=np.int64)
        concepts = np.arange(36, dtype=np.int64).reshape(2, 3, 6) % 3
        loader = MiniKandinskyTargetLoader(
            [(np.zeros((2, 3, 28, 252)), labels, concepts)]
        )
        raw_models = [
            FakeRawMiniKandinskyModel(0),
            FakeRawMiniKandinskyModel(1),
        ]

        collected = collect_ensemble_predictions(
            [MiniKandinskyModelAdapter(model) for model in raw_models],
            loader,
        )

        self.assertEqual(
            collected.concept_member_probabilities.shape,
            (2, 2, 18, 3),
        )
        self.assertEqual(collected.member_representations.shape, (2, 2, 54))
        np.testing.assert_array_equal(collected.labels, np.array([1, 1]))
        self.assertTrue(all(model.evaluation_mode for model in raw_models))

    def test_model_adapter_rejects_non_categorical_feature_count(self) -> None:
        class InvalidModel:
            def __call__(self, images: np.ndarray) -> dict:
                return {
                    "pCS": np.zeros((1, 3, 17)),
                    "YS": np.array([[0.5, 0.5]]),
                    "CS": np.zeros((1, 3, 17)),
                }

        with self.assertRaisesRegex(ValueError, "groups of three"):
            MiniKandinskyModelAdapter(InvalidModel())(
                np.zeros((1, 3, 28, 252))
            )


class MiniKandinskyInterventionTests(unittest.TestCase):
    def test_figure_permutation_and_alignment_are_inverse(self) -> None:
        images = np.concatenate(
            [
                np.full((1, 3, 2, 4), figure, dtype=np.float32)
                for figure in range(3)
            ],
            axis=-1,
        )
        permuted = permute_minikandinsky_figures(images)
        self.assertTrue(np.all(permuted[..., :4] == 1))
        self.assertTrue(np.all(permuted[..., 4:8] == 2))
        self.assertTrue(np.all(permuted[..., 8:] == 0))

        base = _categorical_probabilities(2)[np.newaxis, ...]
        grouped = base.reshape(1, 2, 3, 6, 3)
        transformed = grouped[:, :, [1, 2, 0], :, :].reshape(base.shape)

        aligned = align_permuted_minikandinsky_concepts(transformed)

        np.testing.assert_allclose(aligned, base)

    def test_palette_cycle_changes_known_colors_and_alignment_is_inverse(self) -> None:
        images = np.array(
            [[[[1.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]]],
            dtype=np.float32,
        )
        cycled = cycle_minikandinsky_palette(images)
        expected = np.array(
            [[[[1.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(cycled, expected)

        base = _categorical_probabilities(2)[np.newaxis, ...]
        transformed = base.reshape(1, 2, 3, 6, 3).copy()
        colors = transformed[:, :, :, 3:, :].copy()
        transformed[:, :, :, 3:, :] = colors[..., [2, 0, 1]]
        aligned = align_cycled_minikandinsky_colors(
            transformed.reshape(base.shape)
        )
        np.testing.assert_allclose(aligned, base)

    def test_desaturation_preserves_background_and_source_identity(self) -> None:
        images = np.array(
            [[[[1.0, 1.0, 0.0, 1.0]], [[0.0, 1.0, 0.0, 1.0]], [[0.0, 0.0, 1.0, 1.0]]]],
            dtype=np.float32,
        )

        transformed = desaturate_minikandinsky_palette(images)

        np.testing.assert_allclose(transformed[0, :, 0, 0], 0.35)
        np.testing.assert_allclose(transformed[0, :, 0, 1], 0.70)
        np.testing.assert_allclose(transformed[0, :, 0, 2], 0.15)
        np.testing.assert_allclose(transformed[0, :, 0, 3], 1.0)

    def test_pastel_shift_preserves_background_and_color_identity(self) -> None:
        images = np.array(
            [[[[1.0, 1.0, 0.0, 1.0]], [[0.0, 1.0, 0.0, 1.0]], [[0.0, 0.0, 1.0, 1.0]]]],
            dtype=np.float32,
        )

        transformed = pastelize_minikandinsky_palette(images)

        np.testing.assert_allclose(
            transformed[0, :, 0, :],
            np.array(
                [
                    [1.0, 1.0, 0.45, 1.0],
                    [0.45, 1.0, 0.45, 1.0],
                    [0.45, 0.45, 1.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )


if __name__ == "__main__":
    unittest.main()
