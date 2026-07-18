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
)


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
        for member in range(concepts.shape[0]):
            concepts[member, 0, 0] = peaked(member % 2)

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
        for member in range(concepts.shape[0]):
            for concept in range(concepts.shape[2]):
                concepts[member, 1, concept] = peaked((concept + member % 2) % 3)

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

    def test_unfamiliarity_does_not_imply_shortcut(self) -> None:
        report = build_meta_cognitive_report(
            stable_concepts(samples=2),
            stable_labels(samples=2),
            representation_distances=np.array([0.2, 10.0]),
            reference_distances=np.linspace(0.1, 1.0, 50),
        )

        self.assertGreater(report.neural_familiarity[0], 0.5)
        self.assertEqual(report.neural_familiarity[1], 0.0)
        self.assertEqual(report.shortcut_risk[1], 0.0)
        self.assertFalse(report.review_flag[1])

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
            self.assertIn("shortcut_risk", payload["samples"][0])

            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("neural_familiarity", rows[0])

    def test_familiarity_uses_reference_distribution(self) -> None:
        scores = familiarity_from_reference(
            np.array([0.0, 2.0, 10.0]),
            np.array([0.0, 1.0, 2.0, 3.0]),
        )
        np.testing.assert_allclose(scores, np.array([1.0, 1.0 / 3.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
