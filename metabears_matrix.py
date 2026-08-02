"""Run the frozen MetaBEARS seed/intervention matrix with durable outputs."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

from XOR_MNIST.metacog.protocol import FrozenProtocol, load_protocol


def _checkpoint_filename(protocol: FrozenProtocol, member_seed: int) -> str:
    data = protocol.data
    separated = data["ensemble"]["kind"] == "bears"
    stem = (
        f"deepens_dset-{data['dataset']['name']}-bears-{separated}-"
        f"model-{data['dataset']['model']}-seed-ensmember-{member_seed}-"
        "joint-False"
    )
    if data["training"]["real_kl"]:
        stem += "-real-kl-True"
    stem += "-shortcut-patch-True"
    return stem + ".pt"


def expected_member_seeds(protocol: FrozenProtocol, base_seed: int) -> List[int]:
    ensemble = protocol.data["ensemble"]
    start = int(ensemble["member_seed_offset_start"])
    return [
        base_seed + start + index for index in range(int(ensemble["members"]))
    ]


def build_experiment_command(
    protocol: FrozenProtocol,
    *,
    base_seed: int,
    intervention: str,
    output_directory: Path,
    provenance_cache: Path,
    checkpoint_paths: Optional[Sequence[Path]] = None,
) -> List[str]:
    """Construct an exact frozen-protocol HalfMNIST runner command."""

    data = protocol.data
    training = data["training"]
    calibration = data["calibration"]
    command = [
        sys.executable,
        "-m",
        "metacog.halfmnist_runner",
        "--dataset",
        str(data["dataset"]["name"]),
        "--model",
        str(data["dataset"]["model"]),
        "--task",
        str(data["dataset"]["task"]),
        "--output-dir",
        str(output_directory),
        "--seed",
        str(base_seed),
        "--n_ensembles",
        str(data["ensemble"]["members"]),
        "--n_epochs",
        str(training["epochs"]),
        "--batch_size",
        str(training["batch_size"]),
        "--lr",
        str(training["learning_rate"]),
        "--exp_decay",
        str(training["exponential_decay"]),
        "--ensemble-kind",
        str(data["ensemble"]["kind"]),
        "--lambda_h",
        str(training["lambda_h"]),
        "--familiarity-validation-quantile",
        str(calibration["familiarity_validation_quantile"]),
        "--shortcut-fallback-quantile",
        str(calibration["shortcut_fallback_quantile"]),
        "--intervention",
        intervention,
        "--ece-bins",
        str(calibration["ece_bins"]),
        "--shortcut-patch-training",
        "--protocol-manifest",
        str(protocol.path),
        "--provenance-cache",
        str(provenance_cache),
        "--non_verbose",
    ]
    if training["real_kl"]:
        command.append("--real-kl")
    if training["knowledge_aware_kl"]:
        command.append("--knowledge-aware-kl")
    if checkpoint_paths is None:
        command.extend(["--train-ensemble", "--checkout"])
    else:
        command.append("--ensemble-checkpoints")
        command.extend(str(path) for path in checkpoint_paths)
    return command


def _validate_seed(protocol: FrozenProtocol, seed: int) -> None:
    allowed = list(protocol.data["ensemble"]["base_seeds"])
    if seed not in allowed:
        raise ValueError(f"Seed {seed} is not frozen in protocol seeds {allowed}.")


def _validate_interventions(
    protocol: FrozenProtocol, interventions: Sequence[str]
) -> None:
    evaluation = protocol.data["evaluation"]
    allowed = {
        evaluation["primary_intervention"],
        *evaluation.get("secondary_interventions", []),
        *evaluation.get("supplementary_interventions", []),
        evaluation["negative_control"],
    }
    unsupported = sorted(set(interventions).difference(allowed))
    if unsupported:
        raise ValueError(f"Interventions are not frozen: {unsupported}")


def _copy_checkpoint_source(
    protocol: FrozenProtocol,
    base_seed: int,
    source: Path,
    destination: Path,
) -> List[Path]:
    expected_seeds = expected_member_seeds(protocol, base_seed)
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for member_seed in expected_seeds:
        source_path = source / _checkpoint_filename(protocol, member_seed)
        if not source_path.is_file():
            raise ValueError(
                f"Checkpoint source is missing frozen member seed {member_seed}: "
                f"{source_path}"
            )
        target = destination / source_path.name
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        copied.append(target.resolve())
    return copied


def _local_checkpoint_paths(
    protocol: FrozenProtocol, base_seed: int, xor_root: Path
) -> List[Path]:
    paths = [
        xor_root
        / "data"
        / "ckpts"
        / _checkpoint_filename(protocol, member_seed)
        for member_seed in expected_member_seeds(protocol, base_seed)
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Training did not create checkpoints:\n{formatted}")
    return paths


def _run_command(command: Sequence[str], cwd: Path) -> None:
    print("\nRunning:", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _write_matrix_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def run_seed(
    protocol: FrozenProtocol,
    *,
    repo_root: Path,
    output_root: Path,
    base_seed: int,
    interventions: Sequence[str],
    checkpoint_source: Optional[Path],
    reuse_checkpoints: bool,
    skip_completed: bool,
) -> Mapping[str, Any]:
    _validate_seed(protocol, base_seed)
    _validate_interventions(protocol, interventions)
    xor_root = repo_root / "XOR_MNIST"
    seed_root = output_root / f"seed_{base_seed}"
    checkpoint_destination = seed_root / "checkpoints"
    provenance_cache = output_root / ".provenance_hash_cache.json"
    primary = str(protocol.data["evaluation"]["primary_intervention"])
    trained_primary = False

    if checkpoint_source is not None:
        checkpoint_paths = _copy_checkpoint_source(
            protocol,
            base_seed,
            checkpoint_source,
            checkpoint_destination,
        )
    elif reuse_checkpoints:
        checkpoint_paths = _copy_checkpoint_source(
            protocol,
            base_seed,
            checkpoint_destination,
            checkpoint_destination,
        )
    else:
        primary_output = seed_root / primary
        primary_summary = primary_output / "run_summary.json"
        if primary_summary.is_file() and skip_completed:
            print(f"Skipping completed training evaluation: {primary_summary}")
        else:
            if primary_output.exists() and any(primary_output.iterdir()):
                raise FileExistsError(
                    f"Refusing to overwrite non-empty run: {primary_output}"
                )
            command = build_experiment_command(
                protocol,
                base_seed=base_seed,
                intervention=primary,
                output_directory=primary_output,
                provenance_cache=provenance_cache,
            )
            _run_command(command, xor_root)
            trained_primary = True
        local_paths = _local_checkpoint_paths(protocol, base_seed, xor_root)
        checkpoint_destination.mkdir(parents=True, exist_ok=True)
        checkpoint_paths = []
        for local_path in local_paths:
            target = checkpoint_destination / local_path.name
            shutil.copy2(local_path, target)
            checkpoint_paths.append(target.resolve())

    completed_runs: Dict[str, str] = {}
    for intervention in interventions:
        output_directory = seed_root / intervention
        summary_path = output_directory / "run_summary.json"
        if trained_primary and intervention == primary:
            if not summary_path.is_file():
                raise FileNotFoundError(f"Run did not create {summary_path}")
            completed_runs[intervention] = str(summary_path)
            continue
        if summary_path.is_file() and skip_completed:
            print(f"Skipping completed evaluation: {summary_path}")
            completed_runs[intervention] = str(summary_path)
            continue
        if output_directory.exists() and any(output_directory.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty run: {output_directory}"
            )
        command = build_experiment_command(
            protocol,
            base_seed=base_seed,
            intervention=intervention,
            output_directory=output_directory,
            provenance_cache=provenance_cache,
            checkpoint_paths=checkpoint_paths,
        )
        _run_command(command, xor_root)
        if not summary_path.is_file():
            raise FileNotFoundError(f"Run did not create {summary_path}")
        completed_runs[intervention] = str(summary_path)

    seed_manifest = {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "base_seed": base_seed,
        "member_seeds": expected_member_seeds(protocol, base_seed),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "runs": completed_runs,
    }
    _write_matrix_manifest(seed_root / "matrix_manifest.json", seed_manifest)
    return seed_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen MetaBEARS seeds and controls with durable outputs."
    )
    parser.add_argument("--protocol", default="experiment_protocol.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--interventions",
        nargs="+",
        default=None,
        help="Defaults to the primary and frozen secondary interventions.",
    )
    parser.add_argument(
        "--checkpoint-source",
        default=None,
        help="Existing checkpoint directory; valid only with one seed.",
    )
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    protocol = load_protocol(repo_root / args.protocol)
    output_root = Path(args.output_root).expanduser().resolve()
    evaluation = protocol.data["evaluation"]
    interventions = args.interventions or [
        evaluation["primary_intervention"],
        *evaluation.get("secondary_interventions", []),
    ]
    checkpoint_source = (
        Path(args.checkpoint_source).expanduser().resolve()
        if args.checkpoint_source
        else None
    )
    if checkpoint_source is not None and len(args.seeds) != 1:
        raise ValueError("--checkpoint-source can only be used with one seed.")

    manifests = []
    for seed in args.seeds:
        manifests.append(
            run_seed(
                protocol,
                repo_root=repo_root,
                output_root=output_root,
                base_seed=seed,
                interventions=interventions,
                checkpoint_source=checkpoint_source,
                reuse_checkpoints=args.reuse_checkpoints,
                skip_completed=args.skip_completed,
            )
        )
    _write_matrix_manifest(
        output_root / "matrix_manifest.json",
        {
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.sha256,
            "seeds": manifests,
        },
    )
    print(f"Completed frozen matrix under: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
