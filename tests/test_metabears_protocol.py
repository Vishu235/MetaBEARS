"""Tests for frozen MetaBEARS protocol automation and aggregation."""

import tempfile
import unittest
from pathlib import Path

from aggregate_metabears_results import METRIC_FIELDS, aggregate_rows
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


if __name__ == "__main__":
    unittest.main()
