"""Focused behavioral tests for the Phase-II diagnostic layer."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from XOR_MNIST.metacog import (
    build_meta_cognitive_report,
    familiarity_from_reference,
    probe_concept_consistency,
    select_review_threshold,
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
