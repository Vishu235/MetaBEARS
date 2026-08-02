"""Tests for frozen MetaBEARS protocol automation and aggregation."""

import tempfile
import unittest
from pathlib import Path

from aggregate_metabears_results import (
    METRIC_FIELDS,
    aggregate_rows,
    aggregate_detector_results,
    aggregate_unique_models,
    extract_run_row,
    paired_control_analysis,
    reporting_provenance,
)
from metabears_matrix import build_experiment_command, expected_member_seeds
from XOR_MNIST.metacog.protocol import (
    collect_run_provenance,
    load_protocol,
    validate_protocol_configuration,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "experiment_protocol.json"


def valid_configuration() -> dict:
    return {
        "dataset": "halfmnist",
        "task": "addition",
        "model": "mnistdpl",
        "ensemble_kind": "bears",
        "ensemble_members": 5,
        "n_epochs": 30,
        "batch_size": 64,
        "learning_rate": 0.0005,
        "exponential_decay": 0.95,
        "lambda_h": 0.8,
        "real_kl": True,
        "knowledge_aware_kl": False,
        "shortcut_patch_training": True,
        "shortcut_patch_size": 3,
        "max_batches": None,
        "familiarity_validation_quantile": 0.05,
        "shortcut_fallback_quantile": 0.95,
        "ece_bins": 15,
        "seed": 10,
        "intervention": "patch_shuffled",
    }


class FrozenProtocolTests(unittest.TestCase):
    def test_manifest_accepts_the_frozen_configuration(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)

        validate_protocol_configuration(protocol, valid_configuration())

        self.assertEqual(protocol.protocol_id, "metabears-halfmnist-v1")
        self.assertEqual(len(protocol.sha256), 64)

    def test_manifest_rejects_post_hoc_hyperparameter_changes(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        configuration = valid_configuration()
        configuration["learning_rate"] = 0.001
        configuration["seed"] = 42

        with self.assertRaisesRegex(ValueError, "learning_rate"):
            validate_protocol_configuration(protocol, configuration)

    def test_provenance_hashes_artifacts_and_reuses_a_cache(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "dataset.pt"
            checkpoint = root / "member.pt"
            cache = root / "hash_cache.json"
            dataset.write_bytes(b"frozen dataset")
            checkpoint.write_bytes(b"frozen checkpoint")

            provenance = collect_run_provenance(
                root,
                protocol=protocol,
                dataset_paths=[dataset],
                checkpoint_paths=[checkpoint],
                hash_cache_path=cache,
            )
            repeated = collect_run_provenance(
                root,
                protocol=protocol,
                dataset_paths=[dataset],
                checkpoint_paths=[checkpoint],
                hash_cache_path=cache,
            )

        self.assertTrue(cache.name.endswith(".json"))
        self.assertEqual(
            provenance["dataset_artifacts"][0]["sha256"],
            repeated["dataset_artifacts"][0]["sha256"],
        )
        self.assertEqual(provenance["protocol"]["id"], protocol.protocol_id)


class MatrixRunnerTests(unittest.TestCase):
    def test_frozen_command_contains_exact_training_settings(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        command = build_experiment_command(
            protocol,
            base_seed=10,
            intervention="patch_shuffled",
            output_directory=Path("result"),
            provenance_cache=Path("cache.json"),
        )

        self.assertEqual(expected_member_seeds(protocol, 10), [11, 12, 13, 14, 15])
        self.assertIn("--train-ensemble", command)
        self.assertEqual(command[command.index("--n_epochs") + 1], "30")
        self.assertEqual(command[command.index("--lr") + 1], "0.0005")
        self.assertEqual(command[command.index("--intervention") + 1], "patch_shuffled")


class AggregationTests(unittest.TestCase):
    def test_run_extraction_reports_prevalence_and_normalized_effect(self) -> None:
        detection = {
            "positive_count": 21,
            "prevalence": 0.05,
            "precision": 0.5,
            "recall": 1.0,
            "f1": 2.0 / 3.0,
        }
        summary = {
            "configuration": {"seed": 10},
            "intervention": {
                "name": "patch_shuffled",
                "id_test": {
                    "samples": 420,
                    "base_task_accuracy": 0.99,
                    "perturbed_task_accuracy": 0.91,
                    "input_assignment": {"effective_mismatch_rate": 0.4},
                    "task_invariance_failure_shortcut_flags": detection,
                    "semantic_instability_detection": {
                        **detection,
                        "auroc": 0.9,
                        "average_precision": 0.4,
                    },
                },
            },
            "provenance": {
                "git": {"commit": "abc"},
                "dataset_artifacts": [
                    {"exists": True, "sha256": "a" * 64}
                ],
                "checkpoints": [{"exists": True, "sha256": "b" * 64}],
            },
            "splits": {"id_test": {"review_rate": 0.6, "coverage": 0.4}},
            "ood_detection": {
                "auroc": 0.98,
                "average_precision": 0.99,
                "f1": 0.95,
            },
        }

        row = extract_run_row(summary, Path("run_summary.json"))

        self.assertEqual(row["id_samples"], 420)
        self.assertEqual(row["task_failure_count"], 21)
        self.assertEqual(row["semantic_instability_count"], 21)
        self.assertAlmostEqual(row["task_failure_prevalence"], 0.05)
        self.assertAlmostEqual(row["mismatch_normalized_accuracy_drop"], 0.2)

    def test_aggregation_reports_sample_standard_deviation(self) -> None:
        rows = []
        for seed, accuracy_drop in ((0, 0.1), (10, 0.2)):
            row = {
                "seed": seed,
                "intervention": "patch_shuffled",
                **{metric: 0.5 for metric in METRIC_FIELDS},
            }
            row["id_accuracy_drop"] = accuracy_drop
            rows.append(row)

        aggregates = aggregate_rows(rows)
        accuracy = next(
            row for row in aggregates if row["metric"] == "id_accuracy_drop"
        )

        self.assertEqual(accuracy["n"], 2)
        self.assertAlmostEqual(accuracy["mean"], 0.15)
        self.assertAlmostEqual(accuracy["sample_std"], 0.0707106781)
        self.assertAlmostEqual(accuracy["standard_error"], 0.05)
        self.assertAlmostEqual(accuracy["ci95_low"], -0.4853102368)
        self.assertAlmostEqual(accuracy["ci95_high"], 0.7853102368)
        self.assertEqual(
            accuracy["ci_method"],
            "two-sided Student-t over independent seeds",
        )

    def test_ood_aggregation_deduplicates_shared_control_checkpoints(self) -> None:
        rows = [
            {
                "seed": 0,
                "intervention": intervention,
                "git_commit": "abc",
                "dataset_fingerprint": "dataset",
                "checkpoint_fingerprint": "ensemble-0",
                "ood_auroc": 0.9,
                "ood_average_precision": 0.8,
                "ood_f1": 0.7,
            }
            for intervention in ("patch_shuffled", "patch_removed")
        ]

        model_rows, aggregates = aggregate_unique_models(rows)

        self.assertEqual(len(model_rows), 1)
        auroc = next(row for row in aggregates if row["metric"] == "ood_auroc")
        self.assertEqual(auroc["n"], 1)
        self.assertEqual(auroc["mean"], 0.9)
        self.assertIsNone(auroc["ci95_low"])

    def test_ood_aggregation_rejects_inconsistent_shared_results(self) -> None:
        rows = [
            {
                "seed": 0,
                "git_commit": "abc",
                "dataset_fingerprint": "dataset",
                "checkpoint_fingerprint": "ensemble-0",
                "ood_auroc": auroc,
                "ood_average_precision": 0.8,
                "ood_f1": 0.7,
            }
            for auroc in (0.9, 0.8)
        ]

        with self.assertRaisesRegex(ValueError, "inconsistent ood_auroc"):
            aggregate_unique_models(rows)

    def test_paired_controls_report_within_seed_differences(self) -> None:
        rows = []
        for seed, shuffled, removed in ((0, 0.14, 0.04), (10, 0.05, 0.03)):
            for intervention, accuracy_drop in (
                ("patch_shuffled", shuffled),
                ("patch_removed", removed),
            ):
                rows.append(
                    {
                        "seed": seed,
                        "intervention": intervention,
                        "checkpoint_fingerprint": f"ensemble-{seed}",
                        **{metric: 0.5 for metric in METRIC_FIELDS},
                        "id_accuracy_drop": accuracy_drop,
                    }
                )

        paired, aggregates = paired_control_analysis(
            rows,
            primary_intervention="patch_shuffled",
            comparator_interventions=["patch_removed"],
        )

        differences = [
            row["paired_difference"]
            for row in paired
            if row["metric"] == "id_accuracy_drop"
        ]
        self.assertEqual(differences, [0.1, 0.020000000000000004])
        accuracy = next(
            row for row in aggregates if row["metric"] == "id_accuracy_drop"
        )
        self.assertEqual(accuracy["n"], 2)
        self.assertAlmostEqual(accuracy["mean"], 0.06)

    def test_detector_aggregation_preserves_target_and_detector(self) -> None:
        rows = [
            {
                "seed": seed,
                "intervention": "patch_shuffled",
                "target": "task_invariance_failure",
                "detector": "full_metabears",
                "auroc": auroc,
                "average_precision": 0.4,
                "aurc": 0.1,
                "risk_at_80_coverage": 0.02,
                "review_rate_at_95_recall": 0.6,
            }
            for seed, auroc in ((0, 0.8), (10, 1.0))
        ]

        aggregates = aggregate_detector_results(rows)

        auroc = next(row for row in aggregates if row["metric"] == "auroc")
        self.assertEqual(auroc["target"], "task_invariance_failure")
        self.assertEqual(auroc["detector"], "full_metabears")
        self.assertAlmostEqual(auroc["mean"], 0.9)

    def test_reporting_provenance_hashes_analysis_sources(self) -> None:
        provenance = reporting_provenance(REPO_ROOT)

        self.assertEqual(len(provenance["source_files"]), 2)
        self.assertTrue(
            all(len(record["sha256"]) == 64 for record in provenance["source_files"])
        )


if __name__ == "__main__":
    unittest.main()
