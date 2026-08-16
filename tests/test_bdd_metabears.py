"""Tests for the BDD-OIA MetaBEARS adapter."""

import unittest

import numpy as np
import torch

from XOR_MNIST.metacog import EnsemblePredictions, collect_ensemble_predictions
from XOR_MNIST.metacog.bdd import (
    ACTION_NAMES,
    BDDModelAdapter,
    BDDTargetLoader,
    _action_bits_to_index,
    _combine_action_pairs,
    decode_action_combination,
)


class ActionCombinationTests(unittest.TestCase):
    def test_action_names_order_is_locked(self) -> None:
        self.assertEqual(ACTION_NAMES, ("forward", "stop", "left", "right"))

    def test_decode_round_trips_bit_order(self) -> None:
        for index in range(16):
            forward, stop, left, right = decode_action_combination(index)
            expected = 8 * forward + 4 * stop + 2 * left + 1 * right
            self.assertEqual(expected, index)

    def test_decode_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            decode_action_combination(16)
        with self.assertRaises(ValueError):
            decode_action_combination(-1)

    def test_combine_action_pairs_matches_deterministic_bits(self) -> None:
        for index in range(16):
            forward, stop, left, right = decode_action_combination(index)
            pairs = np.zeros((1, 4, 2), dtype=np.float64)
            pairs[0, 0, int(forward)] = 1.0
            pairs[0, 1, int(stop)] = 1.0
            pairs[0, 2, int(left)] = 1.0
            pairs[0, 3, int(right)] = 1.0
            combination = _combine_action_pairs(pairs)
            self.assertEqual(combination.shape, (1, 16))
            expected = np.zeros(16)
            expected[index] = 1.0
            np.testing.assert_allclose(combination[0], expected)

    def test_combine_action_pairs_sums_to_one(self) -> None:
        rng = np.random.default_rng(0)
        positive = rng.uniform(0.1, 0.9, size=(5, 4))
        pairs = np.stack([1.0 - positive, positive], axis=-1)
        combination = _combine_action_pairs(pairs)
        self.assertEqual(combination.shape, (5, 16))
        np.testing.assert_allclose(
            combination.sum(axis=-1), np.ones(5), atol=1e-10
        )

    def test_combine_action_pairs_rejects_bad_shape(self) -> None:
        with self.assertRaises(ValueError):
            _combine_action_pairs(np.zeros((3, 5, 2)))
        with self.assertRaises(ValueError):
            _combine_action_pairs(np.zeros((3, 4, 3)))

    def test_action_bits_to_index_matches_decode(self) -> None:
        bits = np.array([[1, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]])
        indices = _action_bits_to_index(bits)
        np.testing.assert_array_equal(indices, [8 + 2 + 1, 0, 15])

    def test_action_bits_to_index_accepts_torch(self) -> None:
        bits = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        indices = _action_bits_to_index(bits)
        self.assertEqual(int(indices[0]), 8 + 2)

    def test_action_bits_to_index_rejects_short_vector(self) -> None:
        with self.assertRaises(ValueError):
            _action_bits_to_index(np.zeros((2, 3)))


class FakeRawBDDModel:
    """Mimics ``DPL_AUC.forward``'s side-effect interface for adapter tests."""

    def __init__(self, member_index: int) -> None:
        self.member_index = member_index
        self.evaluation_mode = False

    def eval(self) -> None:
        self.evaluation_mode = True

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        concept_positive = torch.full(
            (batch_size, 21), 0.5 + 0.05 * self.member_index, dtype=torch.float64
        )
        pair_concepts = torch.stack(
            [1.0 - concept_positive, concept_positive], dim=-1
        )
        self.pC = pair_concepts.reshape(batch_size, 42)
        self.concepts_labeled = concept_positive.reshape(batch_size, 21, 1)

        action_positive = torch.full((batch_size, 4), 0.5, dtype=torch.float64)
        action_pairs = torch.stack(
            [1.0 - action_positive, action_positive], dim=-1
        )
        return action_pairs.reshape(batch_size, 8)


class BDDModelAdapterTests(unittest.TestCase):
    def test_adapter_produces_expected_shapes(self) -> None:
        adapter = BDDModelAdapter(FakeRawBDDModel(member_index=0))
        outputs = adapter(torch.zeros((4, 2048)))
        self.assertEqual(tuple(outputs["pCS"].shape), (4, 21, 2))
        self.assertEqual(tuple(outputs["YS"].shape), (4, 16))
        self.assertEqual(tuple(outputs["CS"].shape), (4, 21))

    def test_adapter_pCS_and_YS_sum_to_one(self) -> None:
        adapter = BDDModelAdapter(FakeRawBDDModel(member_index=1))
        outputs = adapter(torch.zeros((3, 2048)))
        concept_totals = outputs["pCS"].sum(dim=-1)
        action_totals = outputs["YS"].sum(dim=-1)
        torch.testing.assert_close(
            concept_totals, torch.ones_like(concept_totals)
        )
        torch.testing.assert_close(action_totals, torch.ones_like(action_totals))

    def test_adapter_requires_pC_attribute(self) -> None:
        class BrokenModel:
            def __call__(self, images: torch.Tensor) -> torch.Tensor:
                return torch.zeros((images.shape[0], 8))

        adapter = BDDModelAdapter(BrokenModel())
        with self.assertRaises(ValueError):
            adapter(torch.zeros((2, 2048)))

    def test_adapter_requires_concepts_labeled_attribute(self) -> None:
        class PartialModel:
            def __call__(self, images: torch.Tensor) -> torch.Tensor:
                batch_size = images.shape[0]
                self.pC = torch.zeros((batch_size, 42))
                return torch.zeros((batch_size, 8))

        adapter = BDDModelAdapter(PartialModel())
        with self.assertRaises(ValueError):
            adapter(torch.zeros((2, 2048)))

    def test_adapter_getattr_passthrough(self) -> None:
        model = FakeRawBDDModel(member_index=0)
        model.device = "cpu"
        adapter = BDDModelAdapter(model)
        self.assertEqual(adapter.device, "cpu")

    def test_adapter_eval_forwards_to_model(self) -> None:
        model = FakeRawBDDModel(member_index=0)
        adapter = BDDModelAdapter(model)
        adapter.eval()
        self.assertTrue(model.evaluation_mode)


class BDDTargetLoaderTests(unittest.TestCase):
    def test_drops_padding_and_encodes_combined_index(self) -> None:
        actions = torch.tensor(
            [[1.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        concepts = torch.zeros((2, 21))
        images = torch.zeros((2, 2048))
        wrapped = list(BDDTargetLoader([(images, actions, concepts)]))
        self.assertEqual(len(wrapped), 1)
        wrapped_images, wrapped_actions, wrapped_concepts = wrapped[0]
        self.assertTrue(torch.equal(wrapped_images, images))
        self.assertTrue(torch.equal(wrapped_concepts, concepts))
        np.testing.assert_array_equal(wrapped_actions.numpy(), [8 + 2 + 1, 0])

    def test_rejects_malformed_batch(self) -> None:
        loader = [(torch.zeros((1, 2048)), torch.zeros((1, 4)))]
        with self.assertRaises(ValueError):
            list(BDDTargetLoader(loader))

    def test_rejects_short_action_vector(self) -> None:
        loader = [
            (
                torch.zeros((1, 2048)),
                torch.zeros((1, 2)),
                torch.zeros((1, 21)),
            )
        ]
        with self.assertRaises(ValueError):
            list(BDDTargetLoader(loader))

    def test_len_passthrough(self) -> None:
        class SizedLoader:
            def __len__(self) -> int:
                return 7

            def __iter__(self):
                return iter([])

        self.assertEqual(len(BDDTargetLoader(SizedLoader())), 7)


class BDDIntegrationTests(unittest.TestCase):
    """End-to-end check through the generic MetaBEARS collector."""

    def test_collect_ensemble_predictions_accepts_bdd_adapter(self) -> None:
        models = [BDDModelAdapter(FakeRawBDDModel(i)) for i in range(3)]
        images = torch.zeros((4, 2048))
        actions = torch.zeros((4, 5))
        actions[:, 0] = 1.0  # forward = 1 for every sample; combined index 8
        concepts = torch.zeros((4, 21))
        loader = BDDTargetLoader([(images, actions, concepts)])

        predictions = collect_ensemble_predictions(
            models, loader, representation_key="CS"
        )

        self.assertIsInstance(predictions, EnsemblePredictions)
        self.assertEqual(
            predictions.concept_member_probabilities.shape, (3, 4, 21, 2)
        )
        self.assertEqual(predictions.label_member_probabilities.shape, (3, 4, 16))
        self.assertEqual(predictions.member_representations.shape, (3, 4, 21))
        np.testing.assert_array_equal(predictions.labels, [8, 8, 8, 8])


if __name__ == "__main__":
    unittest.main()
