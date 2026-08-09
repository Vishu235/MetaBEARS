"""Command-line runner for checkpoint-based MiniKandinsky MetaBEARS runs."""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, List, Optional, Sequence

from .experiment import run_metabears_experiment
from .minikandinsky import (
    ImageTransformLoader,
    MiniKandinskyModelAdapter,
    MiniKandinskyTargetLoader,
    desaturate_minikandinsky_palette,
    get_minikandinsky_intervention,
)
from .protocol import collect_run_provenance


def _xor_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _enable_legacy_imports() -> None:
    xor_root = str(_xor_root())
    if xor_root not in sys.path:
        sys.path.insert(0, xor_root)


def build_parser() -> argparse.ArgumentParser:
    """Build the MiniKandinsky evaluation CLI without importing PyTorch."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained MiniKandinsky ensemble with validation-calibrated "
            "MetaBEARS diagnostics."
        )
    )
    parser.add_argument("--dataset", default="minikandinsky")
    parser.add_argument("--model", default="minikanddpl")
    parser.add_argument("--task", default="mini_patterns_bombazza")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16)
    parser.add_argument("--finetuning", type=int, default=0)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--c-sup", "--c_sup", type=float, default=1.0)
    parser.add_argument("--w-c", "--w_c", type=float, default=10.0)
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--splitted", action="store_true")
    parser.add_argument(
        "--ensemble-checkpoints",
        nargs="+",
        default=None,
        help=(
            "Two or more trained MiniKandinsky checkpoint paths. Relative paths "
            "are resolved from XOR_MNIST."
        ),
    )
    parser.add_argument(
        "--checkpoint-seeds",
        nargs="+",
        type=int,
        default=[0, 10, 20],
        help="Seeds used to discover default checkpoint filenames.",
    )
    parser.add_argument(
        "--intervention",
        choices=["none", "figure_permute", "palette_cycle"],
        default="figure_permute",
    )
    parser.add_argument(
        "--ood-transform",
        choices=["none", "palette_desaturate"],
        default="palette_desaturate",
        help="Controlled test-only distribution shift used for OOD evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Artifact directory; defaults to "
            "colab_outputs/metabears_minikandinsky/<time>."
        ),
    )
    parser.add_argument(
        "--representation-key",
        default="CS",
        help="Model output flattened as the learned representation.",
    )
    parser.add_argument(
        "--familiarity-validation-quantile",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--shortcut-fallback-quantile",
        type=float,
        default=0.95,
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
        help="Optional JSON cache for dataset and checkpoint hashes.",
    )
    return parser


def _resolve_paths(paths: Sequence[str]) -> List[Path]:
    resolved = []
    for raw_path in paths:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = _xor_root() / candidate
        resolved.append(candidate.resolve())
    return resolved


def discover_checkpoint_paths(args: argparse.Namespace) -> List[Path]:
    """Resolve explicit checkpoints or the baseline notebook filenames."""

    if args.ensemble_checkpoints:
        paths = _resolve_paths(args.ensemble_checkpoints)
    else:
        paths = [
            (
                _xor_root()
                / "data"
                / "ckpts"
                / f"minikandinsky-minikanddpl-dis-{seed}-end.pt"
            ).resolve()
            for seed in args.checkpoint_seeds
        ]
    if len(paths) < 2:
        raise ValueError("MetaBEARS requires at least two ensemble checkpoints.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing MiniKandinsky checkpoints:\n"
            f"{formatted}\nPass their Google Drive paths with "
            "--ensemble-checkpoints."
        )
    return paths


def _new_model(dataset: Any, args: argparse.Namespace) -> Any:
    from models import get_model

    encoder, decoder = dataset.get_backbone()
    n_images, c_split = dataset.get_split()
    model = get_model(args, encoder, decoder, n_images, c_split)
    model.to(model.device)
    return model


def load_ensemble(
    dataset: Any,
    args: argparse.Namespace,
    checkpoint_paths: Sequence[Path],
) -> List[Any]:
    """Load and freeze MiniKandinsky checkpoint members."""

    import torch

    ensemble = []
    for path in checkpoint_paths:
        model = _new_model(dataset, args)
        try:
            state = torch.load(path, map_location=model.device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=model.device)
        if isinstance(state, dict):
            for key in ("state_dict", "model_state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        model.load_state_dict(state)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        ensemble.append(MiniKandinskyModelAdapter(model))
        print(f"Loaded MiniKandinsky member: {path}")
    return ensemble


def _default_output_directory() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return (
        _xor_root().parent
        / "colab_outputs"
        / "metabears_minikandinsky"
        / timestamp
    )


def _configuration(
    args: argparse.Namespace,
    checkpoint_paths: Sequence[Path],
    command_arguments: Sequence[str],
) -> dict:
    return {
        "dataset": args.dataset,
        "model": args.model,
        "task": args.task,
        "seed": args.seed,
        "ensemble_members": len(checkpoint_paths),
        "ensemble_source": "checkpoints",
        "checkpoints": [str(path) for path in checkpoint_paths],
        "checkpoint_seeds": list(args.checkpoint_seeds),
        "training_concept_supervision": args.c_sup,
        "training_concept_weight": args.w_c,
        "representation_key": args.representation_key,
        "intervention": args.intervention,
        "ood_transform": args.ood_transform,
        "ood_definition": (
            "The ID test images are deterministically recolored from the known "
            "red/yellow/blue palette to three distinct grayscale levels."
            if args.ood_transform == "palette_desaturate"
            else None
        ),
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "ece_bins": args.ece_bins,
        "familiarity_validation_quantile": (
            args.familiarity_validation_quantile
        ),
        "shortcut_fallback_quantile": args.shortcut_fallback_quantile,
        "command_arguments": list(command_arguments),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dataset != "minikandinsky" or args.model != "minikanddpl":
        parser.error(
            "This runner supports --dataset minikandinsky --model minikanddpl."
        )
    if args.task != "mini_patterns_bombazza":
        parser.error("This runner supports --task mini_patterns_bombazza.")
    if args.ece_bins < 2:
        parser.error("--ece-bins must be at least 2.")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive.")
    try:
        checkpoint_paths = discover_checkpoint_paths(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    _enable_legacy_imports()
    try:
        from datasets import get_dataset
        from utils.conf import set_random_seed
    except ModuleNotFoundError as error:
        parser.error(
            f"The BEARS runtime dependency '{error.name}' is missing. "
            "Install requirements.txt before running the experiment."
        )

    set_random_seed(args.seed)
    dataset = get_dataset(args)
    _, raw_validation_loader, raw_test_loader = dataset.get_data_loaders()
    validation_loader = MiniKandinskyTargetLoader(raw_validation_loader)
    id_test_loader = MiniKandinskyTargetLoader(raw_test_loader)
    if args.ood_transform == "palette_desaturate":
        ood_test_loader = ImageTransformLoader(
            id_test_loader,
            desaturate_minikandinsky_palette,
        )
    else:
        ood_test_loader = None

    ensemble = load_ensemble(dataset, args, checkpoint_paths)
    intervention = (
        None
        if args.intervention == "none"
        else get_minikandinsky_intervention(args.intervention)
    )
    output_directory = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_directory()
    )
    command_arguments = sys.argv[1:] if argv is None else list(argv)
    configuration = _configuration(args, checkpoint_paths, command_arguments)
    repo_root = _xor_root().parent
    provenance = collect_run_provenance(
        repo_root,
        dataset_paths=[_xor_root() / "data" / "kand-3k.zip"],
        checkpoint_paths=checkpoint_paths,
        hash_cache_path=(
            Path(args.provenance_cache).expanduser().resolve()
            if args.provenance_cache
            else None
        ),
    )
    result = run_metabears_experiment(
        ensemble,
        validation_loader,
        id_test_loader,
        ood_test_loader=ood_test_loader,
        output_directory=output_directory,
        familiarity_validation_quantile=args.familiarity_validation_quantile,
        shortcut_fallback_quantile=args.shortcut_fallback_quantile,
        representation_key=args.representation_key,
        max_batches=args.max_batches,
        ece_bins=args.ece_bins,
        intervention=intervention,
        run_configuration=configuration,
        run_provenance=provenance,
    )

    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(f"MiniKandinsky MetaBEARS artifacts: {result.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
