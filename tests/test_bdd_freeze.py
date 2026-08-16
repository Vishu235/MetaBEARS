"""Tests for the BDD-OIA MetaBEARS freeze manifest builder."""

import json
import tempfile
import unittest
from pathlib import Path

from bdd_freeze import build_freeze_manifest, hash_variant_directory, main


def _write_variant(root: Path, name: str, *, extra_files=()) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "configuration": {"dataset": "bdd_oia", "variant": name},
        "splits": {"id_test": {"task_accuracy": 0.5}},
    }
    (directory / "run_summary.json").write_text(json.dumps(summary))
    for filename, content in extra_files:
        (directory / filename).write_bytes(content)
    return directory


class HashVariantDirectoryTests(unittest.TestCase):
    def test_hashes_recognized_artifacts_and_embeds_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = _write_variant(
                root,
                "dpl_auc",
                extra_files=[("validation_predictions.npz", b"fake-npz-bytes")],
            )
            record = hash_variant_directory(directory)

        self.assertEqual(record["run_summary"]["configuration"]["variant"], "dpl_auc")
        self.assertIn("run_summary.json", record["artifact_hashes"])
        self.assertIn("validation_predictions.npz", record["artifact_hashes"])
        self.assertEqual(len(record["artifact_hashes"]["run_summary.json"]["sha256"]), 64)

    def test_missing_directory_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            hash_variant_directory(Path("/nonexistent/path/xyz"))

    def test_missing_run_summary_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            with self.assertRaises(FileNotFoundError):
                hash_variant_directory(empty)

    def test_hash_is_deterministic_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _write_variant(root, "a")
            second = _write_variant(root, "b")
            hash_a = hash_variant_directory(first)["artifact_hashes"]["run_summary.json"]["sha256"]
            hash_b = hash_variant_directory(second)["artifact_hashes"]["run_summary.json"]["sha256"]
            # Both summaries embed a different "variant" string, so hashes differ.
            self.assertNotEqual(hash_a, hash_b)


class BuildFreezeManifestTests(unittest.TestCase):
    def test_freezes_multiple_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = _write_variant(root, "dpl_auc")
            entropy_dir = _write_variant(root, "dpl_auc_entropy")

            manifest = build_freeze_manifest(
                freeze_id="bdd-oia-metabears-v1",
                scope="Test scope",
                repo_root=root,
                variants={"base": base_dir, "entropy": entropy_dir},
            )

        self.assertEqual(manifest["freeze_id"], "bdd-oia-metabears-v1")
        self.assertEqual(manifest["status"], "frozen")
        self.assertEqual(set(manifest["variants"]), {"base", "entropy"})
        self.assertIn("frozen_at_utc", manifest)
        self.assertIn("environment", manifest)

    def test_requires_at_least_one_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_freeze_manifest(
                    freeze_id="x", scope="x", repo_root=Path(tmp), variants={}
                )


class MainCliTests(unittest.TestCase):
    def test_end_to_end_writes_manifest_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = _write_variant(root, "dpl_auc")
            output_path = root / "freeze.json"

            exit_code = main(
                [
                    "--freeze-id", "bdd-oia-metabears-v1",
                    "--scope", "First-pass baseline freeze.",
                    "--variant", "base", str(base_dir),
                    "--output", str(output_path),
                    "--repo-root", str(root),
                ]
            )

            self.assertEqual(exit_code, 0)
            manifest = json.loads(output_path.read_text())
            self.assertEqual(manifest["freeze_id"], "bdd-oia-metabears-v1")
            self.assertIn("base", manifest["variants"])


if __name__ == "__main__":
    unittest.main()
