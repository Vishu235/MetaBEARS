"""End-to-end tests for validation-calibrated MetaBEARS experiments."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from XOR_MNIST.metacog import (
    PredictionIntervention,
    calibrate_metabears,
    collect_ensemble_predictions,
    run_metabears_experiment,
    shortcut_proxy_labels,
)
from XOR_MNIST.metacog.halfmnist_runner import _checkpoint_filename


def peaked(class_index: int, class_count: int, peak: float = 0.98) -> np.ndarray:
    probabilities = np.full(class_count, (1.0 - peak) / (class_count - 1))
    probabilities[class_index] = peak
    return probabilities


class FakeHalfMNISTMember:
    """NumPy-only member with one deliberate semantic mismatch at code 1."""

    def __init__(self, member_index: int) -> None:
        self.member_index = member_index
        self.evaluation_mode = False

    def eval(self) -> None:
        self.evaluation_mode = True

    def __call__(self, images: np.ndarray) -> dict:
        codes = np.asarray(images)[:, 0].astype(int)
        labels = np.empty((codes.shape[0], 5), dtype=np.float64)
        concepts = np.empty((codes.shape[0], 2, 3), dtype=np.float64)
        representations = np.empty((codes.shape[0], 2, 3), dtype=np.float64)

        for index, code in enumerate(codes):
            labels[index] = peaked(code % 5, 5)
            predicted_concept = code % 3
            for concept_index in range(2):
                concept_class = predicted_concept
                if code == 1 and concept_index == 0:
                    concept_class = 0 if self.member_index == 0 else 2
                concepts[index, concept_index] = peaked(concept_class, 3)

            scale = self.member_index + 1
            representations[index] = (
                np.arange(6, dtype=np.float64).reshape(2, 3) + scale * code
            )

        return {"CS": representations, "YS": labels, "pCS": concepts}


def loader(codes: list) -> list:
    code_array = np.asarray(codes, dtype=np.float64)
    labels = np.asarray([int(code) % 5 for code in codes], dtype=np.int64)
    concepts = np.asarray(
        [[int(code) % 3, int(code) % 3] for code in codes],
        dtype=np.int64,
    )
    return [(code_array[:, np.newaxis], labels, concepts)]


class CalibrationTests(unittest.TestCase):
    def test_proxy_marks_task_correct_concept_mismatch(self) -> None:
        predictions = collect_ensemble_predictions(
            [FakeHalfMNISTMember(0), FakeHalfMNISTMember(1)],
            loader([0, 1, 2, 3]),
        )

        proxy = shortcut_proxy_labels(predictions)

        np.testing.assert_array_equal(
            proxy,
            np.array([False, True, False, False]),
        )

    def test_mixed_proxy_labels_select_supervised_threshold(self) -> None:
        predictions = collect_ensemble_predictions(
            [FakeHalfMNISTMember(0), FakeHalfMNISTMember(1)],
            loader([0, 1, 2, 3]),
        )

        calibration = calibrate_metabears(predictions)

        self.assertEqual(calibration.shortcut_policy, "validation_proxy_max_f1")
        self.assertTrue(calibration.validation_report.shortcut_flag[1])
        self.assertEqual(calibration.shortcut_metrics["failure_count"], 1)

    def test_no_proxy_positive_uses_reported_quantile_fallback(self) -> None:
        predictions = collect_ensemble_predictions(
            [FakeHalfMNISTMember(0), FakeHalfMNISTMember(1)],
            loader([0, 2, 3]),
        )

        calibration = calibrate_metabears(
            predictions,
            shortcut_fallback_quantile=0.9,
        )

        self.assertEqual(
            calibration.shortcut_policy,
            "validation_quantile_no_mixed_proxy_labels",
        )
        self.assertEqual(np.count_nonzero(calibration.shortcut_proxy), 0)


class ExperimentRunnerTests(unittest.TestCase):
    def test_patch_checkpoint_filename_cannot_overwrite_standard_member(self) -> None:
        args = Namespace(
            dataset="halfmnist",
            ensemble_kind="bears",
            model="mnistdpl",
            joint=False,
            real_kl=True,
            shortcut_patch_training=True,
        )

        filename = _checkpoint_filename(args, seed=1)

        self.assertTrue(filename.endswith("-shortcut-patch-True.pt"))

    def test_experiment_writes_all_split_artifacts_and_ood_metrics(self) -> None:
        models = [FakeHalfMNISTMember(0), FakeHalfMNISTMember(1)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            result = run_metabears_experiment(
                models,
                loader([0, 1, 2, 3]),
                loader([0, 2]),
                ood_test_loader=loader([20, 21]),
                output_directory=output,
                run_configuration={"seed": 7, "source": "fake-checkpoints"},
            )

            expected_files = {
                "validation_report.json",
                "validation_report.csv",
                "validation_predictions.npz",
                "calibration.npz",
                "id_test_report.json",
                "id_test_report.csv",
                "id_test_predictions.npz",
                "ood_test_report.json",
                "ood_test_report.csv",
                "ood_test_predictions.npz",
                "run_summary.json",
            }
            observed_files = {path.name for path in output.iterdir()}
            self.assertTrue(expected_files.issubset(observed_files))
            summary = json.loads(
                (output / "run_summary.json").read_text(encoding="utf-8")
            )

        self.assertTrue(all(model.evaluation_mode for model in models))
        self.assertIsNotNone(result.ood_test_report)
        self.assertEqual(summary["configuration"]["seed"], 7)
        self.assertEqual(summary["splits"]["id_test"]["samples"], 2)
        self.assertEqual(summary["splits"]["ood_test"]["samples"], 2)
        self.assertEqual(summary["ood_detection"]["auroc"], 1.0)
        self.assertEqual(summary["ood_detection"]["average_precision"], 1.0)
        self.assertIn("task_macro_f1", summary["splits"]["id_test"])
        self.assertIn("task_ece", summary["splits"]["id_test"])
        self.assertIn("concept_macro_f1", summary["splits"]["id_test"])
        self.assertIn("accepted_task_risk", summary["splits"]["id_test"])
        self.assertIn("id_test", summary["shortcut_detection"])

    def test_experiment_runs_paired_intervention_and_writes_artifacts(self) -> None:
        identity = PredictionIntervention(
            name="identity_test",
            description="Identity transform used to test paired orchestration.",
            transform_images=lambda images, labels, concepts: np.asarray(
                images
            ).copy(),
            align_concept_probabilities=lambda probabilities: np.asarray(
                probabilities
            ).copy(),
            assignment_metrics=lambda labels, batch_sizes: {
                "effective_mismatch_rate": 0.5,
                "batch_sizes": list(batch_sizes),
            },
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            result = run_metabears_experiment(
                [FakeHalfMNISTMember(0), FakeHalfMNISTMember(1)],
                loader([0, 1, 2, 3]),
                loader([0, 1, 2, 3]),
                ood_test_loader=loader([20, 21]),
                output_directory=output,
                intervention=identity,
            )
            observed_files = {path.name for path in output.iterdir()}

        self.assertIn("validation_intervention_predictions.npz", observed_files)
        self.assertIn("id_test_intervention_predictions.npz", observed_files)
        self.assertIn("ood_test_intervention_predictions.npz", observed_files)
        self.assertEqual(result.summary["intervention"]["name"], "identity_test")
        self.assertEqual(
            result.summary["intervention"]["id_test"][
                "task_prediction_invariance_rate"
            ],
            1.0,
        )
        self.assertEqual(
            result.summary["splits"]["id_test"]["mean_perturbation_js"],
            0.0,
        )
        semantic_detection = result.summary["intervention"]["id_test"][
            "semantic_instability_detection"
        ]
        self.assertFalse(semantic_detection["evaluable"])
        self.assertEqual(
            semantic_detection["non_evaluable_reason"], "no positive examples"
        )
        self.assertEqual(
            result.summary["intervention"]["id_test"]["input_assignment"][
                "effective_mismatch_rate"
            ],
            0.5,
        )

    def test_experiment_can_run_without_an_ood_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_metabears_experiment(
                [FakeHalfMNISTMember(0), FakeHalfMNISTMember(1)],
                loader([0, 1, 2]),
                loader([0, 2]),
                output_directory=temporary_directory,
            )

        self.assertIsNone(result.ood_test_report)
        self.assertIsNone(result.summary["ood_detection"])


if __name__ == "__main__":
    unittest.main()
