"""Tests for the BDD-OIA compositional OOD split."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

from XOR_MNIST.metacog.bdd import decode_action_combination
from XOR_MNIST.metacog.bdd_ood import (
    combination_frequencies,
    main as bdd_ood_main,
    select_rare_combinations,
    split_records_by_combination,
    write_compositional_split,
)


def _record(action_bits, stem="sample"):
    padded = list(action_bits) + [0.0]
    return {
        "img_path": f"{stem}.jpg",
        "class_label": [float(value) for value in padded],
        "attribute_label": [0.0] * 21,
        "uncertain_attribute_label": [0.0] * 21,
    }


class CombinationFrequencyTests(unittest.TestCase):
    def test_counts_every_combined_index(self) -> None:
        records = [
            _record((1, 0, 0, 0), "a"),
            _record((1, 0, 0, 0), "b"),
            _record((0, 0, 0, 0), "c"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_BDD_OIA.pkl"
            with path.open("wb") as handle:
                pickle.dump(records, handle)
            frequencies = combination_frequencies(path)

        self.assertEqual(set(frequencies), set(range(16)))
        self.assertEqual(frequencies[8], 2)  # forward=1 -> index 8
        self.assertEqual(frequencies[0], 1)  # all zero -> index 0
        self.assertEqual(sum(frequencies.values()), 3)

    def test_matches_decode_action_combination_bit_order(self) -> None:
        # forward + right -> 8 + 1 = 9
        records = [_record((1, 0, 0, 1), "a")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_BDD_OIA.pkl"
            with path.open("wb") as handle:
                pickle.dump(records, handle)
            frequencies = combination_frequencies(path)
        self.assertEqual(frequencies[9], 1)
        self.assertEqual(decode_action_combination(9), (True, False, False, True))


class SelectRareCombinationsTests(unittest.TestCase):
    def test_absent_combinations_are_always_rare(self) -> None:
        frequencies = {index: 100 for index in range(16)}
        frequencies[7] = 0
        rare = select_rare_combinations(frequencies, max_fraction=0.01)
        self.assertIn(7, rare)

    def test_respects_cumulative_budget(self) -> None:
        frequencies = {index: 10 for index in range(16)}
        frequencies[0] = 1
        frequencies[1] = 2
        frequencies[2] = 100
        # total = 10*13 + 1 + 2 + 100 = 233; 10% budget = 23.3
        rare = select_rare_combinations(frequencies, max_fraction=0.10)
        self.assertIn(0, rare)
        self.assertIn(1, rare)
        self.assertNotIn(2, rare)
        rare_total = sum(frequencies[index] for index in rare)
        self.assertLessEqual(rare_total, 0.10 * sum(frequencies.values()))

    def test_rejects_invalid_fraction(self) -> None:
        frequencies = {index: 1 for index in range(16)}
        with self.assertRaises(ValueError):
            select_rare_combinations(frequencies, max_fraction=0.0)
        with self.assertRaises(ValueError):
            select_rare_combinations(frequencies, max_fraction=1.0)

    def test_rejects_incomplete_frequency_table(self) -> None:
        with self.assertRaises(ValueError):
            select_rare_combinations({0: 1, 1: 1}, max_fraction=0.1)

    def test_rejects_empty_training_split(self) -> None:
        frequencies = {index: 0 for index in range(16)}
        with self.assertRaises(ValueError):
            select_rare_combinations(frequencies, max_fraction=0.1)


class SplitRecordsTests(unittest.TestCase):
    def test_partitions_by_membership(self) -> None:
        records = [
            _record((1, 0, 0, 0), "a"),  # index 8
            _record((0, 0, 0, 0), "b"),  # index 0
            _record((1, 1, 1, 1), "c"),  # index 15
        ]
        common, rare = split_records_by_combination(records, {0, 15})
        self.assertEqual([r["img_path"] for r in common], ["a.jpg"])
        self.assertEqual(
            sorted(r["img_path"] for r in rare), ["b.jpg", "c.jpg"]
        )


class WriteCompositionalSplitTests(unittest.TestCase):
    def test_writes_two_pkl_files_referencing_same_stems(self) -> None:
        records = [
            _record((1, 0, 0, 0), "a"),  # common, index 8
            _record((1, 0, 0, 0), "b"),  # common, index 8
            _record((0, 0, 0, 0), "c"),  # rare, index 0
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "test_BDD_OIA.pkl"
            with source.open("wb") as handle:
                pickle.dump(records, handle)

            id_path = root / "test_id_BDD_OIA.pkl"
            ood_path = root / "test_ood_BDD_OIA.pkl"
            id_count, ood_count = write_compositional_split(
                source, {0}, id_output_path=id_path, ood_output_path=ood_path
            )

            self.assertEqual((id_count, ood_count), (2, 1))
            with id_path.open("rb") as handle:
                id_records = pickle.load(handle)
            with ood_path.open("rb") as handle:
                ood_records = pickle.load(handle)
            self.assertEqual(
                sorted(r["img_path"] for r in id_records), ["a.jpg", "b.jpg"]
            )
            self.assertEqual([r["img_path"] for r in ood_records], ["c.jpg"])

    def test_rejects_filenames_without_recognized_split_name(self) -> None:
        records = [_record((0, 0, 0, 0), "a")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "test_BDD_OIA.pkl"
            with source.open("wb") as handle:
                pickle.dump(records, handle)
            with self.assertRaises(ValueError):
                write_compositional_split(
                    source,
                    {0},
                    id_output_path=root / "bad_name.pkl",
                    ood_output_path=root / "test_ood_BDD_OIA.pkl",
                )

    def test_rejects_empty_ood_split(self) -> None:
        records = [_record((1, 0, 0, 0), "a")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "test_BDD_OIA.pkl"
            with source.open("wb") as handle:
                pickle.dump(records, handle)
            with self.assertRaises(ValueError):
                write_compositional_split(
                    source,
                    {15},
                    id_output_path=root / "test_id_BDD_OIA.pkl",
                    ood_output_path=root / "test_ood_BDD_OIA.pkl",
                )


class MainCliTests(unittest.TestCase):
    def test_end_to_end_generates_id_and_ood_splits_for_val_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            # 9 forward-only (index 8) + 1 all-zero (index 0) training samples,
            # so index 0 (absent-ish, 10%) is rare under a 15% budget.
            train_records = [_record((1, 0, 0, 0), f"t{i}") for i in range(9)]
            train_records.append(_record((0, 0, 0, 0), "t9"))
            with (data_dir / "train_BDD_OIA.pkl").open("wb") as handle:
                pickle.dump(train_records, handle)

            for split in ("val", "test"):
                records = [
                    _record((1, 0, 0, 0), f"{split}_common"),
                    _record((0, 0, 0, 0), f"{split}_rare"),
                ]
                with (data_dir / f"{split}_BDD_OIA.pkl").open("wb") as handle:
                    pickle.dump(records, handle)

            summary_path = data_dir / "summary.json"
            exit_code = bdd_ood_main(
                [
                    "--bdd-data-dir", str(data_dir),
                    "--max-fraction", "0.15",
                    "--output-summary", str(summary_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads(summary_path.read_text())
            self.assertIn(0, summary["rare_combined_action_indices"])
            self.assertEqual(summary["split_counts"]["val"], {"id": 1, "ood": 1})
            self.assertEqual(summary["split_counts"]["test"], {"id": 1, "ood": 1})
            self.assertTrue((data_dir / "val_id_BDD_OIA.pkl").is_file())
            self.assertTrue((data_dir / "val_ood_BDD_OIA.pkl").is_file())
            self.assertTrue((data_dir / "test_id_BDD_OIA.pkl").is_file())
            self.assertTrue((data_dir / "test_ood_BDD_OIA.pkl").is_file())


if __name__ == "__main__":
    unittest.main()
