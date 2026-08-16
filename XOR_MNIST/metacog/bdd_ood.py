"""Compositional out-of-distribution split for BDD-OIA.

BDD-OIA's preprocessing pipeline (``BDD_OIA/preprocess_lastframe.py``) only
caches precomputed 2048-d ResNet50 features, not raw images, so a
pixel-level controlled transform in the style of MiniKandinsky's palette
desaturation is not available without a new preprocessing pass. This module
instead follows the same design HalfMNIST already uses for its own OOD
split: samples whose label combination is rare or absent in the training
distribution form a compositional out-of-distribution set drawn from the
existing validation/test data, requiring no new preprocessing and no
retraining of the already-frozen checkpoints.

The combined-action encoding matches :mod:`metacog.bdd` exactly (forward is
the most significant bit, right the least), so a sample's OOD membership is
determined by the same 16-way index the diagnostic layer itself reports.

The rarity rule is fixed before looking at any evaluation result: a
combination is rare if it lies among the lowest-frequency combinations in
the *training* split whose counts sum to at most ``max_fraction`` of the
training set. This budget is a property of the training distribution alone,
never of validation/test accuracy or any MetaBEARS score.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .bdd import _action_bits_to_index


ACTION_DIM = 4


def _load_records(pkl_path: Any) -> List[Dict[str, Any]]:
    with open(pkl_path, "rb") as handle:
        return pickle.load(handle)


def _combined_index(record: Dict[str, Any]) -> int:
    bits = np.asarray(record["class_label"], dtype=np.float64)[:ACTION_DIM]
    if bits.shape[0] != ACTION_DIM:
        raise ValueError(
            f"Expected at least {ACTION_DIM} action labels; got {bits.shape[0]}."
        )
    return int(_action_bits_to_index(bits))


def combination_frequencies(train_pkl_path: Any) -> Dict[int, int]:
    """Count each of the 16 combined-action indices in the training split."""

    records = _load_records(train_pkl_path)
    counts: Dict[int, int] = {index: 0 for index in range(16)}
    for record in records:
        counts[_combined_index(record)] += 1
    return counts


def select_rare_combinations(
    frequencies: Dict[int, int], *, max_fraction: float = 0.1
) -> Set[int]:
    """Freeze which combined-action indices count as rare.

    Combinations are added in ascending training-frequency order (absent
    combinations first) until their cumulative training count would exceed
    ``max_fraction`` of the total training sample count. This selection
    depends only on training-split statistics, never on validation, test, or
    evaluation results.
    """

    if not 0.0 < max_fraction < 1.0:
        raise ValueError("max_fraction must lie strictly within (0, 1).")
    if set(frequencies) != set(range(16)):
        raise ValueError("frequencies must contain exactly the 16 combined-action indices.")

    total = sum(frequencies.values())
    if total <= 0:
        raise ValueError("frequencies must describe a non-empty training split.")
    budget = max_fraction * total

    ordered = sorted(frequencies.items(), key=lambda item: (item[1], item[0]))
    rare: Set[int] = set()
    cumulative = 0
    for index, count in ordered:
        if count == 0:
            rare.add(index)
            continue
        if cumulative + count > budget:
            break
        rare.add(index)
        cumulative += count
    return rare


def split_records_by_combination(
    records: Sequence[Dict[str, Any]], rare_combinations: Set[int]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition already-loaded records into (common, rare) lists."""

    common: List[Dict[str, Any]] = []
    rare: List[Dict[str, Any]] = []
    for record in records:
        (rare if _combined_index(record) in rare_combinations else common).append(
            record
        )
    return common, rare


def write_compositional_split(
    source_pkl_path: Any,
    rare_combinations: Set[int],
    *,
    id_output_path: Any,
    ood_output_path: Any,
) -> Tuple[int, int]:
    """Read an existing split's pkl and write ID/OOD pkl files beside it.

    The written files reference the same underlying per-sample ``.pt``
    feature/label/concept files as the source split (matched by
    ``img_path``), so no feature data is duplicated or regenerated; only the
    index files differ. Both output filenames must embed a recognized split
    name (``val`` or ``test``) so ``BDD.dataset.BDDDataset`` accepts them.
    """

    id_path = Path(id_output_path)
    ood_path = Path(ood_output_path)
    for path, label in ((id_path, "id_output_path"), (ood_path, "ood_output_path")):
        if not ("val" in path.name or "test" in path.name):
            raise ValueError(
                f"{label} must contain 'val' or 'test' in its filename; got {path.name}."
            )

    records = _load_records(source_pkl_path)
    common, rare = split_records_by_combination(records, rare_combinations)
    if not common:
        raise ValueError("The common (ID) split is empty; check rare_combinations.")
    if not rare:
        raise ValueError(
            "The rare (OOD) split is empty; no samples in this source split "
            "match the frozen rare-combination set."
        )

    id_path.parent.mkdir(parents=True, exist_ok=True)
    ood_path.parent.mkdir(parents=True, exist_ok=True)
    with id_path.open("wb") as handle:
        pickle.dump(common, handle)
    with ood_path.open("wb") as handle:
        pickle.dump(rare, handle)
    return len(common), len(rare)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a rare-combination set from BDD-OIA training data and "
            "split the validation/test pkls into ID/OOD files by it."
        )
    )
    parser.add_argument(
        "--bdd-data-dir",
        required=True,
        help="Preprocessed BDD-OIA feature directory (e.g. data/bdd2048_resnet).",
    )
    parser.add_argument(
        "--max-fraction",
        type=float,
        default=0.1,
        help=(
            "Maximum cumulative fraction of the training split's samples "
            "that the selected rare combinations may cover."
        ),
    )
    parser.add_argument(
        "--output-summary",
        default=None,
        help=(
            "Optional path for a JSON summary of the frozen rare-combination "
            "set and resulting split sizes."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    data_dir = Path(args.bdd_data_dir).expanduser().resolve()
    train_pkl = data_dir / "train_BDD_OIA.pkl"
    if not train_pkl.is_file():
        parser.error(f"Missing training split: {train_pkl}")

    frequencies = combination_frequencies(train_pkl)
    rare_combinations = select_rare_combinations(
        frequencies, max_fraction=args.max_fraction
    )
    print(
        f"Frozen {len(rare_combinations)} rare combined-action indices "
        f"(max_fraction={args.max_fraction}): {sorted(rare_combinations)}"
    )

    split_counts: Dict[str, Dict[str, int]] = {}
    for split in ("val", "test"):
        source = data_dir / f"{split}_BDD_OIA.pkl"
        if not source.is_file():
            parser.error(f"Missing {split} split: {source}")
        id_path = data_dir / f"{split}_id_BDD_OIA.pkl"
        ood_path = data_dir / f"{split}_ood_BDD_OIA.pkl"
        id_count, ood_count = write_compositional_split(
            source, rare_combinations, id_output_path=id_path, ood_output_path=ood_path
        )
        split_counts[split] = {"id": id_count, "ood": ood_count}
        print(
            f"{split}: {id_count} ID samples -> {id_path.name}, "
            f"{ood_count} OOD samples -> {ood_path.name}"
        )

    summary = {
        "max_fraction": args.max_fraction,
        "rare_combined_action_indices": sorted(rare_combinations),
        "training_combination_frequencies": {
            str(index): count for index, count in frequencies.items()
        },
        "split_counts": split_counts,
    }
    if args.output_summary:
        summary_path = Path(args.output_summary).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote split summary: {summary_path}")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
