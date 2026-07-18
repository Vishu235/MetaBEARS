"""Colab entry points for BEARS experiments.

This module keeps the Colab notebook small while preserving the repo's normal
command-line entry points. Run it from the repository root, for example:

    python colab_runner.py --job halfmnist_smoke
    python colab_runner.py --job bdd_preprocess_smoke --lastframe-zip /content/drive/MyDrive/bears_data/lastframe.zip
"""

import argparse
import os
import shutil
import shlex
import subprocess
import sys
import time
import zipfile
from pathlib import Path


HALFMNIST_CKPT = "data/ckpts/halfmnist-mnistdpl-dis-None-end.pt"


def _quote_command(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def _log_path_for_command(command, cwd):
    repo_root = Path(os.environ.get("BEARS_REPO_ROOT", cwd))
    log_dir = repo_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    command_name = Path(str(command[1] if len(command) > 1 else command[0])).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return log_dir / f"{timestamp}_{cwd.name}_{command_name}.log"


def _run(command, cwd):
    cwd = Path(cwd)
    log_path = _log_path_for_command(command, cwd)
    command_text = _quote_command(command)
    print(f"\n[{cwd}]$ {command_text}", flush=True)
    print(f"Logging to {log_path}", flush=True)

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"[{cwd}]$ {command_text}\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        returncode = process.wait()

    if returncode != 0:
        raise SystemExit(
            f"Command failed with exit code {returncode}: {command_text}\n"
            f"See log: {log_path}"
        )


def _resolve_path(path, repo_root):
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def diagnostics(args):
    print(f"Python: {sys.version}")
    print(f"Repo root: {args.repo_root}")
    print(f"WANDB_MODE: {os.environ.get('WANDB_MODE')}")
    try:
        import torch
        import torchvision

        print(f"torch: {torch.__version__}")
        print(f"torchvision: {torchvision.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"Could not import torch/torchvision: {exc}")

    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi"], check=False)
    else:
        print("nvidia-smi not found; this runtime is probably CPU-only.")


def ensure_kandinsky_data(repo_root):
    data_dir = repo_root / "XOR_MNIST" / "data"
    extracted_dir = data_dir / "kand-3k"
    zip_path = data_dir / "kand-3k.zip"

    if extracted_dir.exists():
        print(f"MiniKandinsky data already extracted at {extracted_dir}")
        return

    if not zip_path.exists():
        raise SystemExit(
            "Missing XOR_MNIST/data/kand-3k.zip. Upload it or keep it tracked "
            "in the repository before running MiniKandinsky jobs."
        )

    print(f"Extracting {zip_path} to {data_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)


def xor_smoke(args):
    _run([sys.executable, "-m", "example.xor_main"], args.repo_root / "XOR_MNIST")


def halfmnist_smoke(args):
    _run(
        [
            sys.executable,
            "main.py",
            "--model",
            "mnistdpl",
            "--dataset",
            "halfmnist",
            "--task",
            "addition",
            "--n_epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--non_verbose",
        ],
        args.repo_root / "XOR_MNIST",
    )


def halfmnist_eval(args):
    ckpt = Path(args.halfmnist_ckpt or HALFMNIST_CKPT)
    ckpt_path = ckpt if ckpt.is_absolute() else args.repo_root / "XOR_MNIST" / ckpt
    if not ckpt_path.exists():
        raise SystemExit(
            f"Missing HalfMNIST checkpoint: {ckpt_path}. Upload it before "
            "running halfmnist_eval."
        )

    command = [
        sys.executable,
        "main.py",
        "--model",
        "mnistdpl",
        "--dataset",
        "halfmnist",
        "--task",
        "addition",
        "--posthoc",
        "--type",
        args.eval_type,
        "--checkin",
        str(ckpt),
        "--non_verbose",
    ]
    if args.halfmnist_preset == "repo-best":
        command.extend(["--load_best_args", "--seed", str(args.seed)])
    elif args.halfmnist_preset == "paper":
        command.extend(
            [
                "--seed",
                "0",
                "--n_epochs",
                "30",
                "--batch_size",
                "64",
                "--lr",
                "0.0005",
                "--exp_decay",
                "0.95",
                "--lambda_h",
                "0.8",
                "--real-kl",
            ]
        )
    if args.use_ood:
        command.append("--use_ood")
    _run(command, args.repo_root / "XOR_MNIST")


def minikand_smoke(args):
    ensure_kandinsky_data(args.repo_root)
    _run(
        [
            sys.executable,
            "main.py",
            "--model",
            "minikanddpl",
            "--dataset",
            "minikandinsky",
            "--task",
            "mini_patterns_bombazza",
            "--n_epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--c_sup",
            "1",
            "--w_c",
            "10",
            "--non_verbose",
        ],
        args.repo_root / "XOR_MNIST",
    )


def bdd_preprocess(args, full=False):
    lastframe_zip = _resolve_path(args.lastframe_zip, args.repo_root)
    if not lastframe_zip.exists():
        raise SystemExit(
            f"Missing lastframe.zip at {lastframe_zip}. Put it in Google Drive "
            "or upload it to Colab before running BDD preprocessing."
        )

    output = args.bdd_output
    if output is None:
        output = "data/bdd2048_resnet" if full else "data/bdd2048_colab_smoke"

    command = [
        sys.executable,
        "preprocess_lastframe.py",
        "--zip",
        str(lastframe_zip),
        "--output",
        output,
        "--force",
        "--feature-mode",
        "resnet50",
        "--feature-weights",
        args.feature_weights,
        "--feature-batch-size",
        str(args.feature_batch_size),
    ]
    if not full:
        command.extend(["--limit-per-split", str(args.limit_per_split)])

    _run(command, args.repo_root / "BDD_OIA")


def bdd_train(args, full=False):
    data_dir = args.bdd_output
    if data_dir is None:
        data_dir = "data/bdd2048_resnet" if full else "data/bdd2048_colab_smoke"

    epochs = args.epochs if full else min(args.epochs, 1)
    batch_size = args.bdd_batch_size if full else min(args.bdd_batch_size, 4)
    w_entropy = args.w_entropy if full else 0
    h_labeled_param = args.h_labeled_param if full else 0

    _run(
        [
            sys.executable,
            "main_bdd.py",
            "--train",
            "--bdd_data_dir",
            data_dir,
            "--h_type",
            "fcc",
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--nconcepts",
            "30",
            "--nconcepts_labeled",
            "21",
            "--h_sparsity",
            "7",
            "--opt",
            "adam",
            "--lr",
            "0.005",
            "--weight_decay",
            "0.00004",
            "--theta_reg_lambda",
            "0.001",
            "--objective",
            "bce",
            "--model_name",
            args.bdd_model_name,
            "--h_labeled_param",
            str(h_labeled_param),
            "--w_entropy",
            str(w_entropy),
            "--seed",
            str(args.seed),
        ],
        args.repo_root / "BDD_OIA",
    )


def archive_results(args):
    output_dir = args.repo_root / "colab_outputs"
    output_dir.mkdir(exist_ok=True)
    archive_path = output_dir / f"bears_results_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    candidates = [
        args.repo_root / "XOR_MNIST" / "dumps",
        args.repo_root / "XOR_MNIST" / "plots",
        args.repo_root / "BDD_OIA" / "dumps",
        args.repo_root / "BDD_OIA" / "plots",
        args.repo_root / "BDD_OIA" / "out",
        args.repo_root / "BDD_OIA" / "models" / "bdd",
        args.repo_root / "summary_tables",
        args.repo_root / "logs",
    ]

    print(f"Writing {archive_path}")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for candidate in candidates:
            if not candidate.exists():
                continue
            for path in candidate.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(args.repo_root))

    print(f"Created archive: {archive_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BEARS Colab setup checks and experiment presets."
    )
    parser.add_argument(
        "--job",
        required=True,
        choices=[
            "diagnostics",
            "xor_smoke",
            "halfmnist_smoke",
            "halfmnist_eval",
            "minikand_smoke",
            "bdd_preprocess_smoke",
            "bdd_train_smoke",
            "bdd_preprocess_full",
            "bdd_train_full",
            "archive_results",
        ],
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--lastframe-zip",
        default="/content/drive/MyDrive/bears_data/lastframe.zip",
        help="Path to the official BDD-OIA lastframe.zip.",
    )
    parser.add_argument("--bdd-output", default=None, help="BDD output data dir.")
    parser.add_argument("--halfmnist-ckpt", default=None, help="HalfMNIST checkpoint path.")
    parser.add_argument(
        "--eval-type",
        default="frequentist",
        choices=["frequentist", "mcdropout", "laplace", "bears", "deepensembles"],
    )
    parser.add_argument("--use-ood", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bdd-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-split", type=int, default=8)
    parser.add_argument(
        "--feature-weights",
        default="imagenet",
        choices=["imagenet", "none"],
        help="Use 'none' for very fast BDD smoke tests; use 'imagenet' for practical runs.",
    )
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--bdd-model-name", default="dpl_auc")
    parser.add_argument(
        "--h-labeled-param",
        dest="h_labeled_param",
        type=float,
        default=0.0,
        help="BDD concept-supervision loss weight for full BDD training.",
    )
    parser.add_argument(
        "--halfmnist-preset",
        default="default",
        choices=["default", "repo-best", "paper"],
        help=(
            "Extra HalfMNIST evaluation hyperparameters. Use 'paper' for "
            "DPL+BEARS reproduction settings from the paper/repo analysis "
            "notebook, and 'repo-best' for exp_best_args.py presets."
        ),
    )
    parser.add_argument(
        "--w-entropy",
        dest="w_entropy",
        type=float,
        default=1.0,
        help="BDD entropy loss weight for full BDD training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.repo_root = _resolve_path(args.repo_root, Path.cwd())
    os.environ["BEARS_REPO_ROOT"] = str(args.repo_root)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("MPLBACKEND", "Agg")

    jobs = {
        "diagnostics": diagnostics,
        "xor_smoke": xor_smoke,
        "halfmnist_smoke": halfmnist_smoke,
        "halfmnist_eval": halfmnist_eval,
        "minikand_smoke": minikand_smoke,
        "bdd_preprocess_smoke": lambda parsed: bdd_preprocess(parsed, full=False),
        "bdd_train_smoke": lambda parsed: bdd_train(parsed, full=False),
        "bdd_preprocess_full": lambda parsed: bdd_preprocess(parsed, full=True),
        "bdd_train_full": lambda parsed: bdd_train(parsed, full=True),
        "archive_results": archive_results,
    }
    jobs[args.job](args)


if __name__ == "__main__":
    main()
