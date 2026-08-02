"""Frozen protocol validation and reproducibility metadata for MetaBEARS."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Union


PathLike = Union[str, Path]


@dataclass(frozen=True)
class FrozenProtocol:
    """Parsed protocol manifest together with its content digest."""

    path: Path
    data: Mapping[str, Any]
    sha256: str

    @property
    def protocol_id(self) -> str:
        return str(self.data["protocol_id"])


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_protocol(path: PathLike) -> FrozenProtocol:
    """Load and minimally validate a frozen experiment protocol."""

    resolved = Path(path).expanduser().resolve()
    content = resolved.read_bytes()
    data = json.loads(content.decode("utf-8"))
    required = {
        "protocol_id",
        "frozen_core_commit",
        "dataset",
        "ensemble",
        "training",
        "patch",
        "evaluation",
        "calibration",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(
            "Protocol manifest is missing required fields: " + ", ".join(missing)
        )
    return FrozenProtocol(
        path=resolved,
        data=data,
        sha256=_sha256_bytes(content),
    )


def _expected_configuration(protocol: FrozenProtocol) -> Dict[str, Any]:
    data = protocol.data
    return {
        "dataset": data["dataset"]["name"],
        "task": data["dataset"]["task"],
        "model": data["dataset"]["model"],
        "ensemble_kind": data["ensemble"]["kind"],
        "ensemble_members": data["ensemble"]["members"],
        "n_epochs": data["training"]["epochs"],
        "batch_size": data["training"]["batch_size"],
        "learning_rate": data["training"]["learning_rate"],
        "exponential_decay": data["training"]["exponential_decay"],
        "lambda_h": data["training"]["lambda_h"],
        "real_kl": data["training"]["real_kl"],
        "knowledge_aware_kl": data["training"]["knowledge_aware_kl"],
        "shortcut_patch_training": data["patch"]["training_enabled"],
        "shortcut_patch_size": data["patch"]["size"],
        "max_batches": data["evaluation"]["max_batches"],
        "familiarity_validation_quantile": data["calibration"][
            "familiarity_validation_quantile"
        ],
        "shortcut_fallback_quantile": data["calibration"][
            "shortcut_fallback_quantile"
        ],
        "ece_bins": data["calibration"]["ece_bins"],
    }


def validate_protocol_configuration(
    protocol: FrozenProtocol,
    configuration: Mapping[str, Any],
) -> None:
    """Reject a run that differs from the frozen scientific protocol."""

    mismatches = []
    for key, expected in _expected_configuration(protocol).items():
        observed = configuration.get(key)
        if observed != expected:
            mismatches.append(f"{key}: expected {expected!r}, observed {observed!r}")

    seed = configuration.get("seed")
    allowed_seeds = list(protocol.data["ensemble"]["base_seeds"])
    if seed not in allowed_seeds:
        mismatches.append(f"seed: expected one of {allowed_seeds!r}, observed {seed!r}")

    evaluation = protocol.data["evaluation"]
    allowed_interventions = {
        evaluation["primary_intervention"],
        *evaluation.get("secondary_interventions", []),
        *evaluation.get("supplementary_interventions", []),
        evaluation["negative_control"],
    }
    intervention = configuration.get("intervention")
    if intervention not in allowed_interventions:
        mismatches.append(
            "intervention: expected a frozen intervention in "
            f"{sorted(allowed_interventions)!r}, observed {intervention!r}"
        )

    if mismatches:
        raise ValueError(
            f"Run violates protocol {protocol.protocol_id}:\n  - "
            + "\n  - ".join(mismatches)
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hash_cache(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _file_record(path: Path, cache: Dict[str, Any]) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    record: Dict[str, Any] = {"path": str(resolved), "exists": resolved.is_file()}
    if not record["exists"]:
        return record

    stat = resolved.stat()
    key = str(resolved)
    cached = cache.get(key, {})
    if (
        cached.get("size_bytes") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(cached.get("sha256"), str)
    ):
        sha256 = cached["sha256"]
    else:
        sha256 = _file_sha256(resolved)
    cache[key] = {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256,
    }
    record.update(cache[key])
    return record


def _git_metadata(repo_root: Path) -> Dict[str, Any]:
    metadata_values: Dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        metadata_values.update({"commit": commit, "dirty": bool(status.strip())})
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata_values


def validate_protocol_repository(
    protocol: FrozenProtocol,
    repo_root: PathLike,
) -> None:
    """Require a clean worktree descended from the frozen core commit."""

    root = Path(repo_root).expanduser().resolve()
    git = _git_metadata(root)
    if git["commit"] is None:
        raise ValueError("A frozen protocol run requires a Git worktree.")
    if git["dirty"]:
        raise ValueError("A frozen protocol run requires a clean Git worktree.")
    frozen_commit = str(protocol.data["frozen_core_commit"])
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", frozen_commit, git["commit"]],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"Current commit is not descended from frozen core {frozen_commit}."
        ) from error


def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for package_name in ("numpy", "torch", "torchvision"):
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def collect_run_provenance(
    repo_root: PathLike,
    *,
    protocol: Optional[FrozenProtocol],
    dataset_paths: Sequence[PathLike],
    checkpoint_paths: Sequence[PathLike],
    hash_cache_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Collect deterministic source, environment, dataset, and model metadata."""

    root = Path(repo_root).expanduser().resolve()
    cache_path = (
        Path(hash_cache_path).expanduser().resolve()
        if hash_cache_path is not None
        else None
    )
    cache = _load_hash_cache(cache_path)
    dataset_records = [_file_record(Path(path), cache) for path in dataset_paths]
    checkpoint_records = [
        _file_record(Path(path), cache) for path in checkpoint_paths
    ]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(root),
        "protocol": (
            {
                "id": protocol.protocol_id,
                "path": str(protocol.path),
                "sha256": protocol.sha256,
                "frozen_core_commit": protocol.data["frozen_core_commit"],
            }
            if protocol is not None
            else None
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "dataset_artifacts": dataset_records,
        "checkpoints": checkpoint_records,
    }
