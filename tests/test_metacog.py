"""Focused behavioral tests for the Phase-II diagnostic layer."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from XOR_MNIST.metacog import (
    align_swapped_concept_probabilities,
    apply_halfmnist_label_patch,
    build_meta_cognitive_report,
    collect_ensemble_predictions,
    ensemble_leave_one_out_reference_distances,
    ensemble_nearest_reference_distances,
    familiarity_from_reference,
    probe_concept_consistency,
    select_review_threshold,
    shuffled_patch_assignment_metrics,
    swap_halfmnist_image_halves,
)
from XOR_MNIST.metacog.demo import run_demo


def peaked(class_index: int, class_count: int = 3, peak: float = 0.98) -> np.ndarray:
    probability = np.full(class_count, (1.0 - peak) / (class_count - 1))
    probability[class_index] = peak
    return probability


def stable_concepts(
    members: int = 5, samples: int = 2, concepts: int = 2
) -> np.ndarray:
    values = np.empty((members, samples, concepts, 3), dtype=np.float64)
    for member in range(members):
        for sample in range(samples):
            for concept in range(concepts):
                values[member, sample, concept] = peaked((sample + concept) % 3)
    return values


def stable_labels(members: int = 5, samples: int = 2) -> np.ndarray:
    values = np.empty((members, samples, 3), dtype=np.float64)
    for member in range(members):
        for sample in range(samples):
            values[member, sample] = peaked(sample % 3, peak=0.97)
    return values


def inject_shortcut_vote_split(concepts: np.ndarray, sample: int) -> None:
    """Make ensemble members disagree on concept semantics for one sample."""

    for member in range(concepts.shape[0]):
        for concept in range(concepts.shape[2]):
            concepts[member, sample, concept] = peaked((concept + member % 2) % 3)


class HalfMNISTInterventionTests(unittest.TestCase):
    def test_half_swap_exchanges_image_halves(self) -> None:
        images = np.arange(8).reshape(1, 1, 2, 4)

        swapped = swap_halfmnist_image_halves(images)

        np.testing.assert_array_equal(swapped[..., :2], images[..., 2:])
        np.testing.assert_array_equal(swapped[..., 2:], images[..., :2])

    def test_half_swap_alignment_reverses_concept_positions(self) -> None:
        probabilities = np.zeros((2, 1, 2, 3), dtype=np.float64)
        probabilities[:, :, 0, 0] = 1.0
        probabilities[:, :, 1, 2] = 1.0

        aligned = align_swapped_concept_probabilities(probabilities)

        np.testing.assert_array_equal(aligned[:, :, 0], probabilities[:, :, 1])
        np.testing.assert_array_equal(aligned[:, :, 1], probabilities[:, :, 0])

    def test_half_swap_rejects_odd_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "even"):
            swap_halfmnist_image_halves(np.zeros((1, 1, 3, 5)))

    def test_correlated_patch_encodes_canonical_pair(self) -> None:
        images = np.zeros((1, 1, 28, 56), dtype=np.float32)

        patched = apply_halfmnist_label_patch(
            images, np.array([3]), mode="correlated"
        )

        starts = [1, 5, 9, 13, 17]
        left_values = [patched[0, 0, 1:4, start : start + 3].mean() for start in starts]
        right_values = [
            patched[0, 0, 1:4, 28 + start : 28 + start + 3].mean()
            for start in starts
        ]
        np.testing.assert_array_equal(left_values, [0, 1, 0, 0, 0])
        np.testing.assert_array_equal(right_values, [0, 0, 1, 0, 0])

    def test_conflicting_patch_encodes_wrong_sum(self) -> None:
        images = np.zeros((1, 1, 28, 56), dtype=np.float32)

        patched = apply_halfmnist_label_patch(
            images, np.array([3]), mode="conflict"
        )

        self.assertEqual(patched[0, 0, 1:4, 9:12].mean(), 1.0)
        self.assertEqual(patched[0, 0, 1:4, 37:40].mean(), 1.0)

    def test_neutral_patch_removes_label_information(self) -> None:
        images = np.zeros((1, 1, 28, 56), dtype=np.float32)

        patched = apply_halfmnist_label_patch(
            images, np.array([8]), mode="neutral"
        )

        self.assertEqual(patched[0, 0, 1:4, 1:4].mean(), 0.5)
        self.assertEqual(patched[0, 0, 1:4, 29:32].mean(), 0.5)

    def test_removed_patch_restores_reserved_cells_to_background(self) -> None:
        images = np.ones((1, 1, 28, 56), dtype=np.float32)

        patched = apply_halfmnist_label_patch(
            images, np.array([8]), mode="removed"
        )

        for half_offset in (0, 28):
            for start in (1, 5, 9, 13, 17):
                np.testing.assert_array_equal(
                    patched[
                        0,
                        0,
                        1:4,
                        half_offset + start : half_offset + start + 3,
                    ],
                    0.0,
                )
        self.assertEqual(patched[0, 0, 10, 10], 1.0)

    def test_shuffled_patch_preserves_batch_labels_but_reassigns_them(self) -> None:
        images = np.zeros((3, 1, 28, 56), dtype=np.float32)

        patched = apply_halfmnist_label_patch(
            images, np.array([1, 4, 7]), mode="shuffled"
        )

        expected_bright_cells = ((13, 45), (1, 33), (9, 37))
        for sample_index, (left_start, right_start) in enumerate(
            expected_bright_cells
        ):
            self.assertEqual(
                patched[sample_index, 0, 1:4, left_start : left_start + 3].mean(),
                1.0,
            )
            self.assertEqual(
                patched[
                    sample_index,
                    0,
                    1:4,
                    right_start : right_start + 3,
                ].mean(),
                1.0,
            )

    def test_shuffled_patch_selects_rotation_with_fewest_matches(self) -> None:
        images = np.zeros((4, 1, 28, 56), dtype=np.float32)

        patched = apply_halfmnist_label_patch(
            images, np.array([1, 1, 4, 4]), mode="shuffled"
        )

        # A two-position rotation gives [4, 4, 1, 1] with no label matches.
        self.assertEqual(patched[1, 0, 1:4, 9:12].mean(), 1.0)
        self.assertEqual(patched[1, 0, 1:4, 37:40].mean(), 1.0)

    def test_shuffled_patch_reports_its_effective_mismatch_rate(self) -> None:
        metrics = shuffled_patch_assignment_metrics(
            np.array([1, 1, 4, 4, 2]),
            (4, 1),
        )

        self.assertEqual(metrics["changed_assignment_count"], 4)
        self.assertEqual(metrics["unchanged_assignment_count"], 1)
        self.assertEqual(metrics["effective_mismatch_rate"], 0.8)


class FakeBEARSModel:
    """Small NumPy model that follows the BEARS output dictionary contract."""

    def __init__(self, member_index: int) -> None:
        self.member_index = member_index
        self.evaluation_mode = False

    def eval(self) -> None:
        self.evaluation_mode = True

    def __call__(self, images: np.ndarray) -> dict:
        images = np.asarray(images, dtype=np.float64)
        batch_size = images.shape[0]
        concept_probabilities = np.empty((batch_size, 2, 3), dtype=np.float64)
        label_probabilities = np.empty((batch_size, 3), dtype=np.float64)

        for sample_index, sample in enumerate(images[:, 0].astype(int)):
            label_probabilities[sample_index] = peaked(
                (sample + self.member_index) % 3,
                peak=0.9,
            )
            for concept_index in range(2):
                concept_probabilities[sample_index, concept_index] = peaked(
                    (sample + concept_index + self.member_index) % 3,
                    peak=0.9,
                )

        concept_logits = np.repeat(images[:, :, np.newaxis], 2, axis=1)
        concept_logits = np.concatenate(
            [concept_logits, concept_logits + self.member_index], axis=2
        )
        return {
            "CS": concept_logits,
            "YS": label_probabilities,
            "pCS": concept_probabilities,
        }


class EnsembleIntegrationTests(unittest.TestCase):
    def test_collector_preserves_member_and_sample_axes(self) -> None:
        models = [FakeBEARSModel(0), FakeBEARSModel(1)]
        loader = [
            (
                np.array([[0.0], [1.0]]),
                np.array([0, 1]),
                np.array([[0, 1], [1, 2]]),
            ),
            (
                np.array([[2.0]]),
                np.array([2]),
                np.array([[2, 0]]),
            ),
        ]

        collected = collect_ensemble_predictions(models, loader)

        self.assertTrue(all(model.evaluation_mode for model in models))
        self.assertEqual(collected.concept_member_probabilities.shape, (2, 3, 2, 3))
        self.assertEqual(collected.label_member_probabilities.shape, (2, 3, 3))
        self.assertEqual(collected.member_representations.shape, (2, 3, 4))
        self.assertEqual(collected.batch_sizes, (2, 1))
        np.testing.assert_array_equal(collected.labels, np.array([0, 1, 2]))
        np.testing.assert_array_equal(
            collected.concepts,
            np.array([[0, 1], [1, 2], [2, 0]]),
        )
        self.assertEqual(
            np.argmax(collected.label_member_probabilities[1, 2]),
            0,
        )

    def test_collector_can_skip_representations(self) -> None:
        loader = [
            (
                np.array([[0.0], [1.0]]),
                np.array([0, 1]),
                np.array([[0, 1], [1, 2]]),
            )
        ]
        collected = collect_ensemble_predictions(
            [FakeBEARSModel(0), FakeBEARSModel(1)],
            loader,
            representation_key=None,
        )

        self.assertIsNone(collected.member_representations)

    def test_memberwise_reference_distances_are_averaged_after_lookup(self) -> None:
        references = np.array(
            [
                [[0.0], [4.0], [10.0]],
                [[0.0], [8.0], [20.0]],
            ]
        )
        queries = np.array(
            [
                [[1.0], [7.0]],
                [[2.0], [14.0]],
            ]
        )

        distances = ensemble_nearest_reference_distances(queries, references)

        np.testing.assert_allclose(distances, np.array([1.5, 4.5]))

    def test_leave_one_out_distances_do_not_collapse_to_zero(self) -> None:
        references = np.array(
            [
                [[0.0], [2.0], [5.0]],
                [[0.0], [4.0], [10.0]],
            ]
        )

        distances = ensemble_leave_one_out_reference_distances(references)

        np.testing.assert_allclose(distances, np.array([3.0, 3.0, 4.5]))

    def test_distance_outputs_feed_familiarity_contract(self) -> None:
        references = np.array(
            [
                [[0.0], [1.0], [3.0]],
                [[0.0], [2.0], [6.0]],
            ]
        )
        queries = np.array([[[0.1], [20.0]], [[0.2], [40.0]]])

        sample_distances = ensemble_nearest_reference_distances(queries, references)
        reference_distances = ensemble_leave_one_out_reference_distances(references)
        familiarity = familiarity_from_reference(sample_distances, reference_distances)

        self.assertGreater(familiarity[0], familiarity[1])
        self.assertEqual(familiarity[1], 0.0)

    def test_collected_outputs_feed_a_meta_cognitive_report(self) -> None:
        models = [FakeBEARSModel(0), FakeBEARSModel(1)]
        reference = collect_ensemble_predictions(
            models,
            [
                (
                    np.array([[0.0], [2.0], [4.0]]),
                    np.array([0, 2, 1]),
                    np.array([[0, 1], [2, 0], [1, 2]]),
                )
            ],
        )
        evaluated = collect_ensemble_predictions(
            models,
            [
                (
                    np.array([[1.0], [20.0]]),
                    np.array([1, 2]),
                    np.array([[1, 2], [2, 0]]),
                )
            ],
        )

        report = build_meta_cognitive_report(
            evaluated.concept_member_probabilities,
            evaluated.label_member_probabilities,
            representation_distances=ensemble_nearest_reference_distances(
                evaluated.member_representations,
                reference.member_representations,
            ),
            reference_distances=ensemble_leave_one_out_reference_distances(
                reference.member_representations
            ),
        )

        self.assertEqual(report.predicted_label.shape, (2,))
        self.assertTrue(report.ood_flag[1])


class ConceptConsistencyTests(unittest.TestCase):
    def test_identical_members_are_consistent(self) -> None:
        result = probe_concept_consistency(stable_concepts())
        np.testing.assert_allclose(result.score, np.ones(2), atol=1e-10)
        np.testing.assert_allclose(result.ensemble_js, np.zeros(2), atol=1e-10)
        np.testing.assert_allclose(
            result.vote_disagreement, np.zeros(2), atol=1e-10
        )

    def test_split_concept_votes_reduce_consistency(self) -> None:
        concepts = stable_concepts(samples=1)
        inject_shortcut_vote_split(concepts, sample=0)

        result = probe_concept_consistency(concepts)

        self.assertLess(result.score[0], 0.8)
        self.assertGreater(result.ensemble_js[0], 0.2)
        self.assertGreater(result.vote_disagreement[0], 0.1)

    def test_label_preserving_perturbation_is_measured_separately(self) -> None:
        concepts = stable_concepts(samples=1)
        perturbed = np.repeat(concepts[:, :, np.newaxis, :, :], 2, axis=2)
        perturbed[:, :, :, 0, :] = peaked(2)

        result = probe_concept_consistency(
            concepts, perturbed_member_probabilities=perturbed
        )

        self.assertAlmostEqual(result.ensemble_js[0], 0.0, places=10)
        self.assertGreater(result.perturbation_js[0], 0.3)
        self.assertLess(result.score[0], 0.9)

    def test_stable_perturbation_does_not_dilute_base_instability(self) -> None:
        concepts = stable_concepts(samples=1)
        inject_shortcut_vote_split(concepts, sample=0)
        perturbed = concepts[:, :, np.newaxis, :, :].copy()

        base_result = probe_concept_consistency(concepts)
        perturbed_result = probe_concept_consistency(
            concepts, perturbed_member_probabilities=perturbed
        )

        np.testing.assert_allclose(perturbed_result.score, base_result.score)

    def test_invalid_probability_distribution_is_rejected(self) -> None:
        concepts = stable_concepts()
        concepts[0, 0, 0] = np.array([0.6, 0.6, 0.0])
        with self.assertRaisesRegex(ValueError, "sum to one"):
            probe_concept_consistency(concepts)


class MetaCognitiveReportTests(unittest.TestCase):
    def test_shortcut_signature_is_flagged(self) -> None:
        concepts = stable_concepts(samples=2)
        inject_shortcut_vote_split(concepts, sample=1)

        report = build_meta_cognitive_report(
            concepts,
            stable_labels(samples=2),
            representation_distances=np.array([0.3, 0.3]),
            reference_distances=np.linspace(0.2, 0.8, 50),
            review_threshold=0.4,
        )

        self.assertFalse(report.review_flag[0])
        self.assertTrue(report.review_flag[1])
        self.assertGreater(report.shortcut_risk[1], report.shortcut_risk[0])
        self.assertTrue(report.shortcut_flag[1])
        self.assertFalse(report.ood_flag.any())

    def test_unfamiliarity_triggers_review_without_shortcut_risk(self) -> None:
        report = build_meta_cognitive_report(
            stable_concepts(samples=2),
            stable_labels(samples=2),
            representation_distances=np.array([0.2, 10.0]),
            reference_distances=np.linspace(0.1, 1.0, 50),
        )

        self.assertGreater(report.neural_familiarity[0], 0.5)
        self.assertEqual(report.neural_familiarity[1], 0.0)
        self.assertEqual(report.shortcut_risk[1], 0.0)
        self.assertFalse(report.shortcut_flag[1])
        self.assertTrue(report.ood_flag[1])
        self.assertTrue(report.review_flag[1])
        self.assertFalse(report.ood_flag[0])
        self.assertFalse(report.review_flag[0])

    def test_review_flag_is_union_of_flags(self) -> None:
        concepts = stable_concepts(samples=4)
        inject_shortcut_vote_split(concepts, sample=1)
        inject_shortcut_vote_split(concepts, sample=3)

        report = build_meta_cognitive_report(
            concepts,
            stable_labels(samples=4),
            representation_distances=np.array([0.3, 0.3, 10.0, 10.0]),
            reference_distances=np.linspace(0.2, 0.8, 50),
            review_threshold=0.4,
        )

        np.testing.assert_array_equal(
            report.review_flag, report.shortcut_flag | report.ood_flag
        )
        self.assertFalse(report.shortcut_flag[0])
        self.assertFalse(report.ood_flag[0])
        self.assertFalse(report.review_flag[0])

        self.assertTrue(report.shortcut_flag[1])
        self.assertFalse(report.ood_flag[1])
        self.assertTrue(report.review_flag[1])

        self.assertFalse(report.shortcut_flag[2])
        self.assertTrue(report.ood_flag[2])
        self.assertTrue(report.review_flag[2])

        self.assertTrue(report.shortcut_flag[3])
        self.assertTrue(report.ood_flag[3])
        self.assertTrue(report.review_flag[3])

    def test_familiarity_threshold_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "familiarity_threshold"):
            build_meta_cognitive_report(
                stable_concepts(samples=2),
                stable_labels(samples=2),
                representation_distances=np.array([0.2, 0.4]),
                reference_distances=np.linspace(0.1, 1.0, 50),
                familiarity_threshold=1.5,
            )

    def test_report_serialization_has_expected_schema(self) -> None:
        report = build_meta_cognitive_report(
            stable_concepts(samples=2),
            stable_labels(samples=2),
            representation_distances=np.array([0.2, 0.4]),
            reference_distances=np.linspace(0.1, 1.0, 50),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            json_path = report.write_json(directory / "report.json")
            csv_path = report.write_csv(directory / "report.csv")

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["samples"], 2)
            self.assertEqual(len(payload["samples"]), 2)
            self.assertIn("task_confidence", payload["samples"][0])
            self.assertIn("perturbation_js", payload["samples"][0])
            self.assertIn(
                "ensemble_concept_disagreement", payload["samples"][0]
            )
            self.assertIn("shortcut_flag", payload["samples"][0])
            self.assertIn("ood_flag", payload["samples"][0])
            self.assertNotIn("symbolic_confidence", payload["samples"][0])

            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("neural_familiarity", rows[0])
            self.assertIn("task_confidence", rows[0])

    def test_familiarity_uses_reference_distribution(self) -> None:
        scores = familiarity_from_reference(
            np.array([0.0, 2.0, 10.0]),
            np.array([0.0, 1.0, 2.0, 3.0]),
        )
        np.testing.assert_allclose(scores, np.array([1.0, 1.0 / 3.0, 0.0]))


class ThresholdSelectionTests(unittest.TestCase):
    def test_max_f1_separates_clean_scores(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        labels = np.array([False, False, False, True, True, True])

        selection = select_review_threshold(scores, labels)

        self.assertEqual(selection.policy, "max_f1")
        self.assertAlmostEqual(selection.f1, 1.0)
        self.assertAlmostEqual(selection.precision, 1.0)
        self.assertAlmostEqual(selection.recall, 1.0)
        self.assertAlmostEqual(selection.threshold, 0.8)
        self.assertTrue(selection.constraints_satisfied)

    def test_false_review_rate_budget_tightens_threshold(self) -> None:
        scores = np.array([0.1, 0.4, 0.5, 0.6, 0.8, 0.9])
        labels = np.array([False, False, True, False, True, True])

        unconstrained = select_review_threshold(scores, labels)
        constrained = select_review_threshold(
            scores, labels, max_false_review_rate=0.0
        )

        self.assertGreaterEqual(constrained.threshold, unconstrained.threshold)
        self.assertLessEqual(constrained.false_review_rate, 0.0)
        self.assertEqual(constrained.policy, "constrained")
        self.assertTrue(constrained.constraints_satisfied)

    def test_target_precision_constraint_is_respected(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        labels = np.array(
            [False, True, False, True, False, True, True, True]
        )

        selection = select_review_threshold(scores, labels, target_precision=0.8)

        self.assertGreaterEqual(selection.precision, 0.8)
        self.assertEqual(selection.policy, "constrained")

    def test_false_review_budget_can_select_no_reviews(self) -> None:
        scores = np.array([0.1, 0.2, 0.8])
        labels = np.array([True, False, False])

        selection = select_review_threshold(
            scores, labels, max_false_review_rate=0.0
        )

        self.assertEqual(selection.flagged_count, 0)
        self.assertEqual(selection.false_review_rate, 0.0)
        self.assertGreater(selection.threshold, scores.max())

    def test_infeasible_constraint_raises(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.array([True, False, True, False])

        with self.assertRaisesRegex(ValueError, "No threshold satisfies"):
            select_review_threshold(scores, labels, target_precision=0.99)

    def test_infeasible_constraint_falls_back(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.array([True, False, True, False])

        selection = select_review_threshold(
            scores, labels, target_precision=0.99, on_infeasible="max_f1"
        )

        self.assertEqual(selection.policy, "max_f1")
        self.assertFalse(selection.constraints_satisfied)

    def test_low_score_is_risky_direction(self) -> None:
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([False, False, True, True])

        selection = select_review_threshold(scores, labels, higher_is_riskier=False)

        flags = scores <= selection.threshold
        np.testing.assert_array_equal(flags, labels)

    def test_requires_at_least_one_failure(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one failure"):
            select_review_threshold(
                np.array([0.1, 0.2, 0.3]), np.array([False, False, False])
            )

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            select_review_threshold(np.array([0.1, 0.2]), np.array([True]))

    def test_selected_threshold_feeds_report(self) -> None:
        calibration_concepts = stable_concepts(samples=4)
        inject_shortcut_vote_split(calibration_concepts, sample=1)
        calibration_report = build_meta_cognitive_report(
            calibration_concepts,
            stable_labels(samples=4),
            representation_distances=np.array([0.3, 0.3, 0.3, 0.3]),
            reference_distances=np.linspace(0.2, 0.8, 50),
        )
        selection = select_review_threshold(
            calibration_report.shortcut_risk,
            np.array([False, True, False, False]),
        )

        evaluation_concepts = stable_concepts(samples=2)
        inject_shortcut_vote_split(evaluation_concepts, sample=1)
        report = build_meta_cognitive_report(
            evaluation_concepts,
            stable_labels(samples=2),
            representation_distances=np.array([0.3, 0.3]),
            reference_distances=np.linspace(0.2, 0.8, 50),
            review_threshold=selection.threshold,
        )

        self.assertFalse(report.shortcut_flag[0])
        self.assertTrue(report.shortcut_flag[1])


class DemoTests(unittest.TestCase):
    def test_demo_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = run_demo(Path(temporary_directory), seed=42)

        self.assertEqual(report.predicted_label.shape[0], 10)
        np.testing.assert_array_equal(
            report.review_flag, report.shortcut_flag | report.ood_flag
        )


if __name__ == "__main__":
    unittest.main()
