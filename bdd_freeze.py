"""Freeze BDD-OIA MetaBEARS evaluation artifacts into a hash-verified manifest.

Unlike the HalfMNIST v1--v5 protocols, this freezes a first-pass evaluation
of a single already-trained checkpoint ensemble per training variant (no
retraining, no multiple independent ensembles per variant), so it is
explicitly scoped as single-run evidence rather than a multi-trial matrix.
Run once per new evaluation pass (e.g. once for the plain ID/test baseline,
again for the compositional-OOD pass), each with its own ``--freeze-id``.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Dict, Optional, Sequence


_ARTIFACT_NAMES = (
    "run_summary.json",
    "validation_report.json",
    "validation_report.csv",
    "id_test_report.json",
    "id_test_report.csv",
    "ood_test_report.json",
    "ood_test_report.csv",
    "validation_predictions.npz",
    "id_test_predictions.npz",
    "ood_test_predictions.npz",
    "ood_validation_predictions.npz",
    "calibration.npz",
    "representation_normalization.npz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(repo_root: Path) -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        )
        return {"commit": commit, "dirty": bool(status.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"commit": None, "dirty": None}


def _environment() -> Dict[str, Optional[str]]:
    info: Dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "torch": None,
        "numpy": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except ImportError:
        pass
    return info


def hash_variant_directory(variant_dir: Path) -> Dict[str, Any]:
    """Hash every recognized artifact present in one variant's output directory."""

    if not variant_dir.is_dir():
        raise FileNotFoundError(f"Missing variant directory: {variant_dir}")
    summary_path = variant_dir / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing run_summary.json in {variant_dir}")

    artifact_hashes: Dict[str, Dict[str, Any]] = {}
    for name in _ARTIFACT_NAMES:
        path = variant_dir / name
        if path.is_file():
            artifact_hashes[name] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "directory": str(variant_dir),
        "artifact_hashes": artifact_hashes,
        "run_summary": summary,
    }


def build_freeze_manifest(
    *,
    freeze_id: str,
    scope: str,
    repo_root: Path,
    variants: Dict[str, Path],
) -> Dict[str, Any]:
    if not variants:
        raise ValueError("At least one variant directory is required.")
    variant_records = {
        name: hash_variant_directory(directory) for name, directory in variants.items()
    }
    return {
        "freeze_id": freeze_id,
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "git": _git_state(repo_root),
        "environment": _environment(),
        "variants": variant_records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze BDD-OIA MetaBEARS evaluation artifacts with SHA-256 hashes."
    )
    parser.add_argument("--freeze-id", required=True)
    parser.add_argument(
        "--scope",
        required=True,
        help="One-sentence description of exactly what this freeze covers.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        nargs=2,
        metavar=("NAME", "DIRECTORY"),
        required=True,
        help=(
            "A variant name and its MetaBEARS output directory (containing "
            "run_summary.json). Repeat --variant for multiple variants."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    variants = {
        name: Path(directory).expanduser().resolve() for name, directory in args.variant
    }
    try:
        manifest = build_freeze_manifest(
            freeze_id=args.freeze_id,
            scope=args.scope,
            repo_root=Path(args.repo_root).expanduser().resolve(),
            variants=variants,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote freeze manifest: {output_path}")
    for name, record in manifest["variants"].items():
        print(f"  {name}: {len(record['artifact_hashes'])} artifacts hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
