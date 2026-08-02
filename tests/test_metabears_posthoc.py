"""Tests for post-hoc detector baselines and selective-risk curves."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from aggregate_metabears_results import (
    evaluate_detector_matrix,
    evaluate_external_negative_control_fusion_matrix,
    evaluate_intervention_calibrated_fusion_matrix,
    evaluate_leave_one_intervention_out_fusion_matrix,
    load_analysis_protocol,
)
from XOR_MNIST.metacog.protocol import load_protocol
from XOR_MNIST.metacog.posthoc import (
    calibrate_fusion_references,
    calibrate_fusion_references_from_result_directory,
    detector_scores,
    detector_scores_and_targets,
    evaluate_detector_arrays,
    evaluate_fusion_arrays,
    evaluate_fusion_result_directory,
    evaluate_result_directory,
    fit_validation_fusion,
    fit_validation_fusion_from_result_directory,
    fit_leave_one_intervention_out_fusion,
    fit_leave_one_intervention_out_fusion_from_result_directories,
    precision_recall_curve,
    risk_coverage_curve,
)


def peaked(class_index: int, peak: float = 0.9) -> np.ndarray:
    probabilities = np.full(2, 1.0 - peak, dtype=np.float64)
    probabilities[class_index] = peak
    return probabilities


def prediction_pair() -> tuple[dict, dict]:
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    concepts = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.int64)
    base_labels = np.empty((2, 4, 2), dtype=np.float64)
    perturbed_labels = np.empty_like(base_labels)
    base_concepts = np.empty((2, 4, 2, 2), dtype=np.float64)
    perturbed_concepts = np.empty_like(base_concepts)
    for member in range(2):
        for sample, label in enumerate(labels):
            base_labels[member, sample] = peaked(int(label))
            perturbed_label = 1 if sample == 0 else int(label)
            perturbed_labels[member, sample] = peaked(perturbed_label)
            for concept_index, concept in enumerate(concepts[sample]):
                base_concepts[member, sample, concept_index] = peaked(int(concept))
                perturbed_concept = (
                    1 - int(concept)
                    if sample in {0, 1} and concept_index == 0
                    else int(concept)
                )
                perturbed_concepts[member, sample, concept_index] = peaked(
                    perturbed_concept
                )
    base = {
        "concept_member_probabilities": base_concepts,
        "label_member_probabilities": base_labels,
        "labels": labels,
        "concepts": concepts,
    }
    perturbed = {
        "concept_member_probabilities": perturbed_concepts,
        "label_member_probabilities": perturbed_labels,
        "labels": labels,
        "concepts": concepts,
    }
    return base, perturbed


class DetectorTargetTests(unittest.TestCase):
    def test_reconstructs_controlled_failure_targets(self) -> None:
        scores, targets = detector_scores_and_targets(*prediction_pair())

        np.testing.assert_array_equal(
            targets["task_invariance_failure"],
            np.array([True, False, False, False]),
        )
        np.testing.assert_array_equal(
            targets["semantic_instability"],
            np.array([False, True, False, False]),
        )
        self.assertIn("full_metabears", scores)
        self.assertIn("perturbation_js", scores)
        self.assertIn("task_distribution_js", scores)
        self.assertEqual(len(scores), 10)
        np.testing.assert_array_equal(
            targets["controlled_failure_union"],
            np.array([True, True, False, False]),
        )

    def test_unlabeled_score_calibration_does_not_require_targets(self) -> None:
        base, perturbed = prediction_pair()
        unlabeled_base = {
            name: value
            for name, value in base.items()
            if name not in {"labels", "concepts"}
        }
        unlabeled_perturbed = {
            name: value
            for name, value in perturbed.items()
            if name not in {"labels", "concepts"}
        }

        scores = detector_scores(unlabeled_base, unlabeled_perturbed)
        references = calibrate_fusion_references(
            unlabeled_base,
            unlabeled_perturbed,
            signal_names=("perturbation_js", "task_distribution_js"),
        )

        self.assertIn("task_distribution_js", scores)
        self.assertEqual(
            set(references),
            {"perturbation_js", "task_distribution_js"},
        )
        self.assertEqual(references["perturbation_js"].shape, (4,))

    def test_curves_have_expected_endpoints(self) -> None:
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([True, False, False, False])

        precision_recall = precision_recall_curve(scores, labels)
        risk_coverage = risk_coverage_curve(scores, labels)

        self.assertEqual(precision_recall[0]["review_rate"], 0.0)
        self.assertEqual(precision_recall[-1]["recall"], 1.0)
        self.assertEqual(risk_coverage[-1]["coverage"], 1.0)
        self.assertEqual(risk_coverage[-1]["selective_risk"], 0.25)

    def test_evaluates_each_detector_against_both_targets(self) -> None:
        analysis = evaluate_detector_arrays(*prediction_pair())

        self.assertEqual(len(analysis.metrics), 30)
        full_semantic = next(
            row
            for row in analysis.metrics
            if row["target"] == "semantic_instability"
            and row["detector"] == "full_metabears"
        )
        self.assertTrue(full_semantic["evaluable"])
        self.assertEqual(full_semantic["positive_count"], 1)
        self.assertIsNotNone(full_semantic["auroc"])
        self.assertIsNotNone(full_semantic["average_precision"])

    def test_loads_existing_run_artifact_layout(self) -> None:
        base, perturbed = prediction_pair()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            np.savez_compressed(output / "id_test_predictions.npz", **base)
            np.savez_compressed(
                output / "id_test_intervention_predictions.npz", **perturbed
            )

            analysis = evaluate_result_directory(
                output,
                seed=10,
                intervention="patch_shuffled",
            )

        self.assertTrue(analysis.metrics)
        self.assertTrue(
            all(row["seed"] == 10 for row in analysis.metrics)
        )
        self.assertTrue(
            all(
                row["intervention"] == "patch_shuffled"
                for row in analysis.metrics
            )
        )

    def test_matrix_evaluation_reproduces_frozen_semantic_metrics(self) -> None:
        base, perturbed = prediction_pair()
        expected = evaluate_detector_arrays(base, perturbed)
        full_semantic = next(
            row
            for row in expected.metrics
            if row["target"] == "semantic_instability"
            and row["detector"] == "full_metabears"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            np.savez_compressed(output / "id_test_predictions.npz", **base)
            np.savez_compressed(
                output / "id_test_intervention_predictions.npz", **perturbed
            )
            rows = [
                {
                    "seed": 10,
                    "intervention": "patch_shuffled",
                    "summary_path": str(output / "run_summary.json"),
                    "semantic_instability_auroc": full_semantic["auroc"],
                    "semantic_instability_average_precision": full_semantic[
                        "average_precision"
                    ],
                }
            ]

            metrics, precision_recall, risk_coverage = evaluate_detector_matrix(
                rows
            )

        self.assertEqual(len(metrics), 30)
        self.assertTrue(precision_recall)
        self.assertTrue(risk_coverage)

    def test_validation_fusion_fits_without_test_labels(self) -> None:
        base, perturbed = prediction_pair()

        model = fit_validation_fusion(
            base,
            perturbed,
            signal_names=(
                "perturbation_js",
                "task_distribution_js",
                "concept_instability_without_perturbation",
            ),
            cross_validation_folds=2,
            seed=7,
        )
        held_out = evaluate_fusion_arrays(model, base, perturbed)

        self.assertAlmostEqual(float(model.weights.sum()), 1.0)
        self.assertEqual(model.cross_validation_folds, 2)
        self.assertGreaterEqual(model.validation_recall, 0.95)
        self.assertEqual(len(held_out.analysis.metrics), 3)
        self.assertEqual(len(held_out.threshold_metrics), 3)

    def test_validation_fusion_uses_saved_validation_then_id_test(self) -> None:
        base, perturbed = prediction_pair()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            np.savez_compressed(output / "validation_predictions.npz", **base)
            np.savez_compressed(
                output / "validation_intervention_predictions.npz", **perturbed
            )
            np.savez_compressed(output / "id_test_predictions.npz", **base)
            np.savez_compressed(
                output / "id_test_intervention_predictions.npz", **perturbed
            )
            model = fit_validation_fusion_from_result_directory(
                output,
                signal_names=("perturbation_js", "task_distribution_js"),
                cross_validation_folds=2,
                seed=3,
            )
            result = evaluate_fusion_result_directory(
                model,
                output,
                seed=3,
                intervention="patch_shuffled",
            )

        self.assertEqual(len(result.analysis.metrics), 3)
        self.assertTrue(all(row["seed"] == 3 for row in result.analysis.metrics))

    def test_conditioned_references_load_without_validation_targets(self) -> None:
        base, perturbed = prediction_pair()
        unlabeled_base = {
            name: value
            for name, value in base.items()
            if name not in {"labels", "concepts"}
        }
        unlabeled_perturbed = {
            name: value
            for name, value in perturbed.items()
            if name not in {"labels", "concepts"}
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            np.savez_compressed(
                output / "validation_predictions.npz", **unlabeled_base
            )
            np.savez_compressed(
                output / "validation_intervention_predictions.npz",
                **unlabeled_perturbed,
            )

            references = calibrate_fusion_references_from_result_directory(
                output,
                signal_names=("perturbation_js", "task_distribution_js"),
            )

        self.assertEqual(references["task_distribution_js"].shape, (4,))

    def test_v3_matrix_calibrates_each_intervention_without_labels(self) -> None:
        base, perturbed = prediction_pair()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rows = []
            for intervention in ("patch_shuffled", "patch_removed"):
                output = root / intervention
                output.mkdir()
                for split in ("validation", "id_test"):
                    np.savez_compressed(
                        output / f"{split}_predictions.npz", **base
                    )
                    np.savez_compressed(
                        output / f"{split}_intervention_predictions.npz",
                        **perturbed,
                    )
                rows.append(
                    {
                        "seed": 0,
                        "intervention": intervention,
                        "summary_path": str(output / "run_summary.json"),
                    }
                )
            repo_root = Path(__file__).resolve().parents[1]
            base_protocol = load_protocol(repo_root / "experiment_protocol.json")
            analysis_protocol = load_analysis_protocol(
                repo_root / "analysis_protocol_v3.json", base_protocol
            )

            (
                metrics,
                _,
                _,
                models,
                references,
                threshold_results,
            ) = evaluate_intervention_calibrated_fusion_matrix(
                rows, analysis_protocol
            )

        self.assertEqual(len(metrics), 6)
        self.assertEqual(len(models), 1)
        self.assertEqual(len(references), 2)
        self.assertEqual(len(threshold_results), 6)
        self.assertTrue(all(not row["labels_used"] for row in references))
        self.assertTrue(
            all(
                row["detector"] == "intervention_calibrated_fusion_v3"
                for row in metrics
            )
        )

    def test_leave_one_intervention_out_fit_uses_blocked_folds(self) -> None:
        validation_pairs = {
            name: prediction_pair()
            for name in ("patch_shuffled", "patch_removed", "patch_conflict")
        }

        model = fit_leave_one_intervention_out_fusion(
            validation_pairs,
            signal_names=("perturbation_js", "task_distribution_js"),
        )

        self.assertEqual(model.cross_validation_folds, 3)
        self.assertAlmostEqual(float(model.weights.sum()), 1.0)
        self.assertGreaterEqual(model.validation_recall, 0.95)
        self.assertGreaterEqual(model.validation_min_group_recall, 0.95)

    def test_held_out_intervention_validation_is_not_loaded(self) -> None:
        base, perturbed = prediction_pair()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training_directories = {}
            for intervention in (
                "patch_shuffled",
                "patch_removed",
                "patch_conflict",
            ):
                output = root / intervention
                output.mkdir()
                np.savez_compressed(
                    output / "validation_predictions.npz", **base
                )
                np.savez_compressed(
                    output / "validation_intervention_predictions.npz",
                    **perturbed,
                )
                training_directories[intervention] = output
            held_out = root / "patch_neutral"
            held_out.mkdir()
            np.savez_compressed(held_out / "id_test_predictions.npz", **base)
            np.savez_compressed(
                held_out / "id_test_intervention_predictions.npz", **perturbed
            )

            model = (
                fit_leave_one_intervention_out_fusion_from_result_directories(
                    training_directories,
                    signal_names=("perturbation_js", "task_distribution_js"),
                )
            )
            result = evaluate_fusion_result_directory(
                model,
                held_out,
                seed=0,
                intervention="patch_neutral",
                detector_name="leave_one_intervention_out_fusion_v4",
            )

        self.assertEqual(len(result.analysis.metrics), 3)
        self.assertTrue(
            all(
                row["detector"] == "leave_one_intervention_out_fusion_v4"
                for row in result.analysis.metrics
            )
        )

    def test_v4_matrix_evaluates_each_intervention_once(self) -> None:
        base, perturbed = prediction_pair()
        interventions = (
            "patch_shuffled",
            "patch_removed",
            "patch_conflict",
            "patch_neutral",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rows = []
            for intervention in interventions:
                output = root / intervention
                output.mkdir()
                for split in ("validation", "id_test"):
                    np.savez_compressed(
                        output / f"{split}_predictions.npz", **base
                    )
                    np.savez_compressed(
                        output / f"{split}_intervention_predictions.npz",
                        **perturbed,
                    )
                rows.append(
                    {
                        "seed": 0,
                        "intervention": intervention,
                        "summary_path": str(output / "run_summary.json"),
                    }
                )
            repo_root = Path(__file__).resolve().parents[1]
            base_protocol = load_protocol(repo_root / "experiment_protocol.json")
            analysis_protocol = load_analysis_protocol(
                repo_root / "analysis_protocol_v4.json", base_protocol
            )

            metrics, _, _, models, thresholds = (
                evaluate_leave_one_intervention_out_fusion_matrix(
                    rows, analysis_protocol
                )
            )

        self.assertEqual(len(metrics), 12)
        self.assertEqual(len(models), 4)
        self.assertEqual(len(thresholds), 12)
        self.assertTrue(
            all(not row["held_out_validation_used"] for row in models)
        )
        self.assertEqual(
            {row["held_out_intervention"] for row in models},
            set(interventions),
        )

    def test_v5_fits_patch_interventions_without_loading_control_validation(
        self,
    ) -> None:
        base, perturbed = prediction_pair()
        training_interventions = (
            "patch_shuffled",
            "patch_removed",
            "patch_conflict",
            "patch_neutral",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rows = []
            for intervention in training_interventions:
                output = root / intervention
                output.mkdir()
                np.savez_compressed(
                    output / "validation_predictions.npz", **base
                )
                np.savez_compressed(
                    output / "validation_intervention_predictions.npz",
                    **perturbed,
                )
                rows.append(
                    {
                        "seed": 0,
                        "intervention": intervention,
                        "summary_path": str(output / "run_summary.json"),
                    }
                )
            control = root / "half_swap"
            control.mkdir()
            np.savez_compressed(control / "id_test_predictions.npz", **base)
            np.savez_compressed(
                control / "id_test_intervention_predictions.npz", **perturbed
            )
            rows.append(
                {
                    "seed": 0,
                    "intervention": "half_swap",
                    "summary_path": str(control / "run_summary.json"),
                }
            )
            repo_root = Path(__file__).resolve().parents[1]
            base_protocol = load_protocol(repo_root / "experiment_protocol.json")
            analysis_protocol = load_analysis_protocol(
                repo_root / "analysis_protocol_v5.json", base_protocol
            )

            metrics, _, _, models, thresholds = (
                evaluate_external_negative_control_fusion_matrix(
                    rows, analysis_protocol
                )
            )

        self.assertEqual(len(metrics), 3)
        self.assertEqual(len(models), 1)
        self.assertEqual(len(thresholds), 3)
        self.assertFalse(models[0]["negative_control_validation_used"])
        self.assertEqual(models[0]["cross_validation_folds"], 4)
        self.assertTrue(
            all(
                row["detector"] == "external_negative_control_fusion_v5"
                for row in metrics
            )
        )


if __name__ == "__main__":
    unittest.main()
