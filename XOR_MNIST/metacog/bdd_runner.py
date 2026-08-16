"""Command-line runner for a real BDD-OIA MetaBEARS experiment.

This evaluates a frozen ensemble of trained ``dpl_auc`` BDD-OIA checkpoints
(same training variant, different seeds) with the MetaBEARS diagnostic
layer, via :mod:`metacog.bdd`'s adapter. Train the checkpoints first with the
existing ``BDD_OIA/main_bdd.py`` pipeline (see ``colab/MetaBEARS_BDD_OIA.ipynb``);
this runner only evaluates already-trained members, it does not train them.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, List, Optional, Sequence

from .bdd import BDDModelAdapter, BDDTargetLoader
from .experiment import run_metabears_experiment
from .protocol import collect_run_provenance


def _bdd_oia_root() -> Path:
    return Path(__file__).resolve().parents[2] / "BDD_OIA"


def _xor_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _enable_legacy_imports() -> None:
    bdd_root = str(_bdd_oia_root())
    if bdd_root not in sys.path:
        sys.path.insert(0, bdd_root)


def build_parser() -> argparse.ArgumentParser:
    """Build the BDD-OIA evaluation CLI without importing PyTorch."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained BDD-OIA dpl_auc ensemble with "
            "validation-calibrated MetaBEARS diagnostics."
        )
    )
    parser.add_argument(
        "--bdd-data-dir",
        required=True,
        help=(
            "Directory containing train/val/test_BDD_OIA.pkl and the "
            "inputs/labels/concepts split folders (e.g. "
            "BDD_OIA/data/bdd2048_resnet)."
        ),
    )
    parser.add_argument(
        "--ensemble-checkpoints",
        nargs="+",
        required=True,
        help=(
            "Two or more model_best-<seed>.pth.tar checkpoint paths from "
            "the SAME training variant (e.g. all three dpl_auc-<seed> "
            "base-variant checkpoints, or all three dpl_auc_entropy-<seed> "
            "checkpoints). Do not mix variants in one ensemble."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=64)
    parser.add_argument("--nconcepts", type=int, default=30)
    parser.add_argument("--nconcepts-labeled", type=int, default=21)
    parser.add_argument("--concept-dim", type=int, default=1)
    parser.add_argument("--h-sparsity", type=int, default=7)
    parser.add_argument(
        "--nclasses",
        type=int,
        default=5,
        help="Must match the value main_bdd.py used to train the checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory; defaults to colab_outputs/metabears_bdd/<time>.",
    )
    parser.add_argument(
        "--representation-key",
        default="CS",
        help="Model output used as the learned representation.",
    )
    parser.add_argument(
        "--familiarity-validation-quantile",
        type=float,
        default=0.05,
        help="Lower ID-validation familiarity quantile used for OOD review.",
    )
    parser.add_argument(
        "--shortcut-fallback-quantile",
        type=float,
        default=0.95,
        help="Fallback risk quantile when validation proxy labels are degenerate.",
    )
    parser.add_argument(
        "--shortcut-max-false-review-rate",
        type=float,
        default=None,
        help="Optional maximum ID false-review rate for shortcut calibration.",
    )
    parser.add_argument(
        "--familiarity-max-false-review-rate",
        type=float,
        default=0.05,
        help="Maximum ID false-review rate for familiarity calibration.",
    )
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional per-split batch limit for a fast integration check.",
    )
    parser.add_argument(
        "--provenance-cache",
        default=None,
        help="Optional JSON cache for dataset and checkpoint SHA-256 values.",
    )
    parser.add_argument(
        "--compositional-ood",
        action="store_true",
        help=(
            "Use the frozen compositional OOD split (metacog.bdd_ood) instead "
            "of the plain val/test splits: <split>_id_BDD_OIA.pkl for ID "
            "validation/test and <split>_ood_BDD_OIA.pkl for OOD "
            "validation/test. Generate these first with "
            "'python -m metacog.bdd_ood'."
        ),
    )
    return parser


def _resolve_paths(paths: Sequence[str]) -> List[Path]:
    resolved = []
    for raw_path in paths:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = _bdd_oia_root() / candidate
        resolved.append(candidate.resolve())
    return resolved


def discover_checkpoint_paths(args: argparse.Namespace) -> List[Path]:
    paths = _resolve_paths(args.ensemble_checkpoints)
    if len(paths) < 2:
        raise ValueError("MetaBEARS requires at least two ensemble checkpoints.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing BDD-OIA checkpoints:\n{formatted}")
    return paths


def _new_model(args: argparse.Namespace, device: Any) -> Any:
    from aggregators_BDD import CBM_aggregator
    from conceptizers_BDD import image_fcc_conceptizer
    from DPL.dpl_auc import DPL_AUC
    from parametrizers import dfc_parametrizer

    conceptizer = image_fcc_conceptizer(
        2048,
        args.nconcepts,
        args.nconcepts_labeled,
        args.concept_dim,
        args.h_sparsity,
        False,
    )
    parametrizer = dfc_parametrizer(
        2048, 1024, 512, 256, 128, args.nconcepts, args.nclasses, layers=4
    )
    aggregator = CBM_aggregator(
        args.concept_dim, args.nclasses, args.nconcepts_labeled
    )
    model = DPL_AUC(conceptizer, parametrizer, aggregator, True, False, device)
    model.to(device)
    return model


def load_ensemble(
    args: argparse.Namespace,
    checkpoint_paths: Sequence[Path],
    device: Any,
) -> List[Any]:
    """Load and freeze BDD-OIA checkpoint members, wrapped for MetaBEARS."""

    import torch

    ensemble = []
    for path in checkpoint_paths:
        model = _new_model(args, device)
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=device)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        model.load_state_dict(state)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        ensemble.append(BDDModelAdapter(model))
        print(f"Loaded BDD-OIA ensemble member: {path}")
    return ensemble


def _default_output_directory() -> Path:
    import time

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return _xor_root().parent / "colab_outputs" / "metabears_bdd" / timestamp


def _configuration(
    args: argparse.Namespace,
    checkpoint_paths: Sequence[Path],
    command_arguments: Sequence[str],
) -> dict:
    return {
        "dataset": "bdd_oia",
        "model": "dpl_auc",
        "bdd_data_dir": args.bdd_data_dir,
        "compositional_ood": args.compositional_ood,
        "ood_definition": (
            (
                "Validation/test samples whose combined 16-way action index "
                "falls in the training set's frozen rare-combination set "
                "(metacog.bdd_ood); rarity is fixed from training-split "
                "frequency alone, never from evaluation results."
            )
            if args.compositional_ood
            else None
        ),
        "seed": args.seed,
        "ensemble_members": len(checkpoint_paths),
        "ensemble_source": "checkpoints",
        "checkpoints": [str(path) for path in checkpoint_paths],
        "representation_key": args.representation_key,
        "batch_size": args.batch_size,
        "nconcepts": args.nconcepts,
        "nconcepts_labeled": args.nconcepts_labeled,
        "concept_dim": args.concept_dim,
        "h_sparsity": args.h_sparsity,
        "nclasses": args.nclasses,
        "task_definition": (
            "16-way combined categorical over the four independent action "
            "pairs (forward, stop, left, right); see metacog/bdd.py. Task "
            "accuracy therefore requires all four actions to match "
            "simultaneously (equivalent to this project's existing "
            "action_exact_match baseline metric), which is stricter than "
            "per-action correctness."
        ),
        "max_batches": args.max_batches,
        "ece_bins": args.ece_bins,
        "familiarity_validation_quantile": args.familiarity_validation_quantile,
        "shortcut_fallback_quantile": args.shortcut_fallback_quantile,
        "shortcut_max_false_review_rate": args.shortcut_max_false_review_rate,
        "familiarity_max_false_review_rate": args.familiarity_max_false_review_rate,
        "command_arguments": list(command_arguments),
    }


def _collect_provenance(
    bdd_data_dir: Path,
    checkpoint_paths: Sequence[Path],
    provenance_cache: Optional[str],
    *,
    compositional_ood: bool = False,
) -> dict:
    if compositional_ood:
        dataset_paths = [
            bdd_data_dir / f"{split}_{suffix}_BDD_OIA.pkl"
            for split in ("val", "test")
            for suffix in ("id", "ood")
        ]
        dataset_paths.append(bdd_data_dir / "train_BDD_OIA.pkl")
    else:
        dataset_paths = [
            bdd_data_dir / f"{split}_BDD_OIA.pkl"
            for split in ("train", "val", "test")
        ]
    return collect_run_provenance(
        _xor_root().parent,
        protocol=None,
        dataset_paths=dataset_paths,
        checkpoint_paths=checkpoint_paths,
        hash_cache_path=(
            Path(provenance_cache).expanduser().resolve()
            if provenance_cache
            else None
        ),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.ece_bins < 2:
        parser.error("--ece-bins must be at least 2.")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive.")
    for option_name in (
        "shortcut_max_false_review_rate",
        "familiarity_max_false_review_rate",
    ):
        value = getattr(args, option_name)
        if value is not None and not 0.0 <= value <= 1.0:
            option = option_name.replace("_", "-")
            parser.error(f"--{option} must lie within [0, 1].")

    try:
        checkpoint_paths = discover_checkpoint_paths(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    bdd_data_dir = Path(args.bdd_data_dir).expanduser().resolve()
    if not bdd_data_dir.is_dir():
        parser.error(f"--bdd-data-dir does not exist: {bdd_data_dir}")

    _enable_legacy_imports()
    try:
        import torch

        from BDD.dataset import load_data
    except ModuleNotFoundError as error:
        parser.error(
            f"The BDD-OIA runtime dependency '{error.name}' is missing. "
            "Install requirements.txt/requirements.colab.txt before "
            "running a real experiment."
        )

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir = str(bdd_data_dir) + "/"

    def _loader(pkl_name: str, split_image_dir: str):
        pkl_path = bdd_data_dir / pkl_name
        if not pkl_path.is_file():
            parser.error(
                f"Missing {pkl_path}. Generate it first with "
                "'python -m metacog.bdd_ood --bdd-data-dir "
                f"{bdd_data_dir}'."
            )
        return BDDTargetLoader(
            load_data(
                [str(pkl_path)],
                True,
                False,
                args.batch_size,
                uncertain_label=False,
                n_class_attr=2,
                image_dir=split_image_dir,
                resampling=False,
            )
        )

    ood_validation_loader = None
    ood_test_loader = None
    if args.compositional_ood:
        validation_loader = _loader("val_id_BDD_OIA.pkl", image_dir + "val")
        id_test_loader = _loader("test_id_BDD_OIA.pkl", image_dir + "test")
        ood_validation_loader = _loader("val_ood_BDD_OIA.pkl", image_dir + "val")
        ood_test_loader = _loader("test_ood_BDD_OIA.pkl", image_dir + "test")
    else:
        validation_loader = _loader("val_BDD_OIA.pkl", image_dir + "val")
        id_test_loader = _loader("test_BDD_OIA.pkl", image_dir + "test")

    ensemble = load_ensemble(args, checkpoint_paths, device)
    output_directory = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_directory()
    )
    command_arguments = sys.argv[1:] if argv is None else list(argv)
    configuration = _configuration(args, checkpoint_paths, command_arguments)
    provenance = _collect_provenance(
        bdd_data_dir,
        checkpoint_paths,
        args.provenance_cache,
        compositional_ood=args.compositional_ood,
    )

    result = run_metabears_experiment(
        ensemble,
        validation_loader,
        id_test_loader,
        ood_validation_loader=ood_validation_loader,
        ood_test_loader=ood_test_loader,
        output_directory=output_directory,
        familiarity_validation_quantile=args.familiarity_validation_quantile,
        shortcut_fallback_quantile=args.shortcut_fallback_quantile,
        shortcut_max_false_review_rate=args.shortcut_max_false_review_rate,
        familiarity_max_false_review_rate=args.familiarity_max_false_review_rate,
        representation_key=args.representation_key,
        max_batches=args.max_batches,
        ece_bins=args.ece_bins,
        run_configuration=configuration,
        run_provenance=provenance,
    )

    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(f"BDD-OIA MetaBEARS artifacts: {result.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
