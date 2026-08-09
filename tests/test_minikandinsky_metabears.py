"""Tests for the MiniKandinsky MetaBEARS adapter and controls."""

import unittest
from argparse import Namespace
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
