"""Command-line runner for a real HalfMNIST MetaBEARS experiment."""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, List, Optional, Sequence

from .experiment import run_metabears_experiment
from .interventions import get_intervention
from .protocol import (
    collect_run_provenance,
    load_protocol,
    validate_protocol_configuration,
    validate_protocol_repository,
)


def _xor_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _enable_legacy_imports() -> None:
    xor_root = str(_xor_root())
    if xor_root not in sys.path:
        sys.path.insert(0, xor_root)


def build_parser() -> argparse.ArgumentParser:
    """Create a parser without importing the dependency-heavy BEARS runtime."""

    parser = argparse.ArgumentParser(
        description="Run validation-calibrated MetaBEARS on HalfMNIST.",
    )
    parser.add_argument("--dataset", default="halfmnist")
    parser.add_argument("--model", default="mnistdpl")
    parser.add_argument("--task", default="addition")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c_sup", type=float, default=0.0)
    parser.add_argument("--which_c", type=int, nargs="+", default=[-1])
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--splitted", action="store_true")
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--w_sl", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--w_rec", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--w_h", type=float, default=1.0)
    parser.add_argument("--w_c", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--exp_decay", type=float, default=0.99)
    parser.add_argument("--n_epochs", "--n-epochs", type=int, default=50)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=64)
    parser.add_argument("--n_ensembles", "--n-ensembles", type=int, default=5)
    parser.add_argument("--wandb", default=None)
    parser.add_argument("--checkout", action="store_true")
    parser.add_argument("--non_verbose", "--non-verbose", action="store_true")
    parser.add_argument("--lambda_h", "--lambda-h", type=float, default=1.0)
    parser.add_argument("--real-kl", action="store_true")
    parser.add_argument("--knowledge-aware-kl", action="store_true")
    parser.add_argument(
        "--ensemble-checkpoints",
        nargs="+",
        default=None,
        help=(
            "Two or more ensemble checkpoint paths. Relative paths are "
            "resolved from XOR_MNIST. If omitted, standard BEARS filenames "
            "are discovered from data/ckpts."
        ),
    )
    parser.add_argument(
        "--train-ensemble",
        action="store_true",
        help="Train a fresh ensemble before running MetaBEARS.",
    )
    parser.add_argument(
        "--ensemble-kind",
        choices=["bears", "ensemble"],
        default="bears",
        help="Use BEARS KL separation or a standard deep ensemble.",
    )
    parser.add_argument(
        "--load-best-args",
        action="store_true",
        help="Apply the repository's HalfMNIST hyperparameter preset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Artifact directory; defaults to "
            "colab_outputs/metabears_halfmnist/<time>."
        ),
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
        "--representation-key",
        default="CS",
        help="Model output used as the learned representation.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional smoke-test limit applied independently to every split.",
    )
    parser.add_argument(
        "--intervention",
        choices=[
            "none",
            "half_swap",
            "patch_neutral",
            "patch_conflict",
            "patch_removed",
            "patch_shuffled",
        ],
        default="none",
        help=(
            "Optional label-preserving controlled intervention. half_swap "
            "exchanges the two digits; patch_* controls replace, remove, or "
            "shuffle the task-correlated patch and require "
            "--shortcut-patch-training."
        ),
    )
    parser.add_argument(
        "--shortcut-patch-training",
        action="store_true",
        help=(
            "Inject a task-correlated canonical-pair patch into HalfMNIST "
            "training, validation, ID-test, and OOD-test inputs."
        ),
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=15,
        help="Number of fixed-width bins for task and concept ECE.",
    )
    parser.add_argument(
        "--protocol-manifest",
        default=None,
        help="Optional frozen protocol JSON; incompatible runs are rejected.",
    )
    parser.add_argument(
        "--provenance-cache",
        default=None,
        help="Optional JSON cache for dataset and checkpoint SHA-256 values.",
    )
    return parser


def _checkpoint_filename(args: argparse.Namespace, seed: int) -> str:
    separated = args.ensemble_kind == "bears"
    stem = (
        f"deepens_dset-{args.dataset}-bears-{separated}-model-{args.model}-"
        f"seed-ensmember-{seed}-joint-{args.joint}"
    )
    if args.real_kl:
        stem += f"-real-kl-{args.real_kl}"
    if getattr(args, "shortcut_patch_training", False):
        stem += "-shortcut-patch-True"
    return stem + ".pt"


def _resolve_explicit_paths(paths: Sequence[str]) -> List[Path]:
    resolved = []
    for raw_path in paths:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = _xor_root() / candidate
        resolved.append(candidate.resolve())
    return resolved


def discover_checkpoint_paths(args: argparse.Namespace) -> List[Path]:
    """Resolve explicit paths or the exact names written by ``deep_ensemble``."""

    if args.ensemble_checkpoints:
        paths = _resolve_explicit_paths(args.ensemble_checkpoints)
    else:
        base_seed = args.seed if args.seed is not None else 0
        paths = [
            _xor_root()
            / "data"
            / "ckpts"
            / _checkpoint_filename(args, base_seed + member + 1)
            for member in range(args.n_ensembles)
        ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing ensemble checkpoints:\n"
            f"{formatted}\nPass --ensemble-checkpoints or use --train-ensemble."
        )
    if len(paths) < 2:
        raise ValueError("MetaBEARS requires at least two ensemble checkpoints.")
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
    """Load raw or conventionally wrapped PyTorch state dictionaries."""

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
        ensemble.append(model)
        print(f"Loaded ensemble member: {path}")
    return ensemble


def train_ensemble(
    dataset: Any,
    validation_loader: Any,
    args: argparse.Namespace,
) -> List[Any]:
    from utils.bayes import deep_ensemble

    (_xor_root() / "data" / "ckpts").mkdir(parents=True, exist_ok=True)
    base_seed = args.seed if args.seed is not None else 0
    return deep_ensemble(
        seeds=[base_seed + member + 1 for member in range(args.n_ensembles)],
        dataset=dataset,
        num_epochs=args.n_epochs,
        args=args,
        val_loader=validation_loader,
        separate_from_others=args.ensemble_kind == "bears",
        lambda_h=args.lambda_h,
        use_wandb=args.wandb,
        n_facts=5,
        knowledge_aware_kl=args.knowledge_aware_kl,
        real_kl=args.real_kl,
    )


def _default_output_directory() -> Path:
    repo_root = _xor_root().parent
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return repo_root / "colab_outputs" / "metabears_halfmnist" / timestamp


def _run_configuration(
    args: argparse.Namespace,
    *,
    ensemble_members: int,
    ensemble_source: str,
    checkpoint_paths: Sequence[Path],
    command_arguments: Sequence[str],
) -> dict:
    return {
        "dataset": args.dataset,
        "model": args.model,
        "task": args.task,
        "seed": args.seed,
        "ensemble_kind": args.ensemble_kind,
        "ensemble_members": ensemble_members,
        "ensemble_source": ensemble_source,
        "checkpoints": [str(path) for path in checkpoint_paths],
        "representation_key": args.representation_key,
        "max_batches": args.max_batches,
        "command_arguments": list(command_arguments),
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "exponential_decay": args.exp_decay,
        "real_kl": args.real_kl,
        "knowledge_aware_kl": args.knowledge_aware_kl,
        "lambda_h": args.lambda_h,
        "intervention": args.intervention,
        "ece_bins": args.ece_bins,
        "shortcut_patch_training": args.shortcut_patch_training,
        "shortcut_patch_size": 3 if args.shortcut_patch_training else None,
        "familiarity_validation_quantile": (
            args.familiarity_validation_quantile
        ),
        "shortcut_fallback_quantile": args.shortcut_fallback_quantile,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dataset != "halfmnist" or args.model != "mnistdpl":
        parser.error(
            "This runner currently supports --dataset halfmnist "
            "--model mnistdpl."
        )
    if args.task != "addition":
        parser.error("This first runner currently supports --task addition.")
    if args.n_ensembles < 2:
        parser.error("--n_ensembles must be at least 2.")
    if args.train_ensemble and args.ensemble_checkpoints:
        parser.error("Use either --train-ensemble or --ensemble-checkpoints, not both.")
    if args.ece_bins < 2:
        parser.error("--ece-bins must be at least 2.")
    if args.intervention.startswith("patch_") and not args.shortcut_patch_training:
        parser.error(
            "Patch interventions require --shortcut-patch-training so the "
            "base inputs and checkpoints use the correlated patch."
        )

    checkpoint_paths: Sequence[Path] = []
    if not args.train_ensemble:
        try:
            checkpoint_paths = discover_checkpoint_paths(args)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))

    _enable_legacy_imports()
    try:
        from datasets import get_dataset
        from exp_best_args import set_best_args_halfmnist
        from utils.conf import set_random_seed
    except ModuleNotFoundError as error:
        parser.error(
            f"The BEARS runtime dependency '{error.name}' is missing. "
            "Install requirements.txt before executing a real experiment."
        )

    if args.load_best_args:
        args = set_best_args_halfmnist(args)
    protocol = load_protocol(args.protocol_manifest) if args.protocol_manifest else None
    if protocol is not None:
        preliminary_configuration = _run_configuration(
            args,
            ensemble_members=args.n_ensembles,
            ensemble_source="pending",
            checkpoint_paths=checkpoint_paths,
            command_arguments=sys.argv[1:] if argv is None else argv,
        )
        try:
            validate_protocol_configuration(protocol, preliminary_configuration)
            validate_protocol_repository(protocol, _xor_root().parent)
        except ValueError as error:
            parser.error(str(error))
    set_random_seed(args.seed if args.seed is not None else 0)

    dataset = get_dataset(args)
    _, validation_loader, id_test_loader = dataset.get_data_loaders()
    if args.train_ensemble:
        ensemble = train_ensemble(dataset, validation_loader, args)
        ensemble_source = "trained"
        base_seed = args.seed if args.seed is not None else 0
        checkpoint_paths = [
            _xor_root()
            / "data"
            / "ckpts"
            / _checkpoint_filename(args, base_seed + member + 1)
            for member in range(args.n_ensembles)
        ]
    else:
        ensemble = load_ensemble(dataset, args, checkpoint_paths)
        ensemble_source = "checkpoints"

    output_directory = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_directory()
    )
    intervention = (
        None if args.intervention == "none" else get_intervention(args.intervention)
    )
    configuration = _run_configuration(
        args,
        ensemble_members=len(ensemble),
        ensemble_source=ensemble_source,
        checkpoint_paths=checkpoint_paths,
        command_arguments=sys.argv[1:] if argv is None else argv,
    )
    if protocol is not None:
        configuration["protocol_id"] = protocol.protocol_id
        configuration["protocol_sha256"] = protocol.sha256

    repo_root = _xor_root().parent
    if protocol is not None:
        dataset_paths = [
            repo_root / relative_path
            for relative_path in protocol.data["dataset"]["artifact_paths"]
        ]
    else:
        dataset_paths = [
            _xor_root()
            / "datasets"
            / "utils"
            / "2mnist_10digits"
            / "2mnist_10digits.pt",
            _xor_root() / "data" / "rn.npy",
        ]
    provenance = collect_run_provenance(
        repo_root,
        protocol=protocol,
        dataset_paths=dataset_paths,
        checkpoint_paths=checkpoint_paths,
        hash_cache_path=args.provenance_cache,
    )
    result = run_metabears_experiment(
        ensemble,
        validation_loader,
        id_test_loader,
        ood_test_loader=dataset.ood_loader,
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
    print(f"MetaBEARS artifacts: {result.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
