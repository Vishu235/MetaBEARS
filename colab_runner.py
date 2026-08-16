"""Colab entry points for BEARS experiments.

This module keeps the Colab notebook small while preserving the repo's normal
command-line entry points. Run it from the repository root, for example:

    python colab_runner.py --job halfmnist_smoke
    python colab_runner.py --job minikand_train --seed 0 --epochs 30
    python colab_runner.py --job metabears_minikandinsky --minikand-checkpoints ...
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
        # Force the child Python process to be unbuffered. Without this,
        # print() in a subprocess whose stdout is a pipe (not a TTY) is
        # block-buffered by default, so nothing appears here until the
        # child's internal buffer fills or the process exits — a long
        # training run can look completely silent for a long time even
        # while it is actively progressing.
        child_env = dict(os.environ, PYTHONUNBUFFERED="1")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
            env=child_env,
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
        "--seed",
        str(args.seed),
        "--non_verbose",
    ]
    if args.halfmnist_preset == "repo-best":
        command.append("--load_best_args")
    elif args.halfmnist_preset == "paper":
        command.extend(
            [
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


def metabears_demo(args):
    output_dir = args.repo_root / "colab_outputs" / "metabears_demo"
    _run(
        [
            sys.executable,
            "-m",
            "XOR_MNIST.metacog.demo",
            "--output-dir",
            str(output_dir),
            "--seed",
            str(args.seed),
        ],
        args.repo_root,
    )


def metabears_halfmnist(args):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.repo_root
        / "colab_outputs"
        / "metabears_halfmnist"
        / timestamp
    )
    command = [
        sys.executable,
        "-m",
        "metacog.halfmnist_runner",
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--n_ensembles",
        str(args.n_ensembles),
        "--n_epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--ensemble-kind",
        args.metabears_ensemble_kind,
        "--lambda_h",
        str(args.lambda_h),
        "--familiarity-validation-quantile",
        str(args.familiarity_validation_quantile),
        "--shortcut-fallback-quantile",
        str(args.shortcut_fallback_quantile),
        "--intervention",
        args.metabears_intervention,
        "--ece-bins",
        str(args.ece_bins),
        "--non_verbose",
    ]
    if args.metabears_checkpoints:
        command.append("--ensemble-checkpoints")
        command.extend(args.metabears_checkpoints)
    if args.metabears_train_ensemble:
        command.append("--train-ensemble")
    if args.metabears_real_kl:
        command.append("--real-kl")
    if args.metabears_shortcut_patch_training:
        command.append("--shortcut-patch-training")
    if args.metabears_max_batches is not None:
        command.extend(["--max-batches", str(args.metabears_max_batches)])
    if args.halfmnist_preset == "repo-best":
        command.append("--load-best-args")
    elif args.halfmnist_preset == "paper":
        command.extend(
            [
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
    _run(command, args.repo_root / "XOR_MNIST")


def _run_minikand(args, *, epochs, save_checkpoint):
    ensure_kandinsky_data(args.repo_root)
    command = [
        sys.executable,
        "main.py",
        "--model",
        "minikanddpl",
        "--dataset",
        "minikandinsky",
        "--task",
        "mini_patterns_bombazza",
        "--n_epochs",
        str(epochs),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--c_sup",
        str(args.minikand_c_sup),
        "--w_c",
        str(args.minikand_w_c),
        "--w_h",
        str(args.minikand_w_h),
        "--non_verbose",
    ]
    if args.minikand_entropy:
        command.append("--entropy")
    if save_checkpoint:
        command.append("--checkout")
    _run(command, args.repo_root / "XOR_MNIST")
    if save_checkpoint and (
        args.minikand_checkpoint_tag is not None
        or not (args.minikand_c_sup == 1.0 and args.minikand_w_c == 10.0)
    ):
        source = (
            args.repo_root
            / "XOR_MNIST"
            / "data"
            / "ckpts"
            / f"minikandinsky-minikanddpl-dis-{args.seed}-end.pt"
        )
        supervision = format(args.minikand_c_sup, "g").replace(".", "p")
        weight = format(args.minikand_w_c, "g").replace(".", "p")
        tag = (
            f"{args.minikand_checkpoint_tag}-"
            if args.minikand_checkpoint_tag is not None
            else ""
        )
        destination = source.with_name(
            "minikandinsky-minikanddpl-"
            f"{tag}csup-{supervision}-wc-{weight}-seed-{args.seed}-end.pt"
        )
        if not source.is_file():
            raise SystemExit(
                f"Expected trained checkpoint was not created: {source}"
            )
        os.replace(source, destination)
        print(f"Saved supervision-control checkpoint: {destination}")


def minikand_smoke(args):
    _run_minikand(args, epochs=min(args.epochs, 1), save_checkpoint=False)


def minikand_train(args):
    _run_minikand(args, epochs=args.epochs, save_checkpoint=True)


def metabears_minikandinsky(args):
    ensure_kandinsky_data(args.repo_root)
    if not args.minikand_checkpoints or len(args.minikand_checkpoints) < 2:
        raise SystemExit(
            "Pass at least two trained members with --minikand-checkpoints."
        )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.minikand_output_dir
    if output_dir is None:
        output_dir = (
            args.repo_root
            / "colab_outputs"
            / "metabears_minikandinsky"
            / timestamp
        )
    command = [
        sys.executable,
        "-m",
        "metacog.minikandinsky_runner",
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--intervention",
        args.minikand_intervention,
        "--c-sup",
        str(args.minikand_c_sup),
        "--w-c",
        str(args.minikand_w_c),
        "--representation-key",
        args.minikand_representation_key,
        "--representation-normalization",
        args.minikand_representation_normalization,
        "--ood-validation-transform",
        args.minikand_ood_validation_transform,
        "--ood-transform",
        args.minikand_ood_transform,
        "--familiarity-validation-quantile",
        str(args.familiarity_validation_quantile),
        "--shortcut-fallback-quantile",
        str(args.shortcut_fallback_quantile),
        "--ece-bins",
        str(args.ece_bins),
        "--ensemble-checkpoints",
        *args.minikand_checkpoints,
    ]
    if args.minikand_shortcut_max_false_review_rate is not None:
        command.extend(
            [
                "--shortcut-max-false-review-rate",
                str(args.minikand_shortcut_max_false_review_rate),
            ]
        )
    command.extend(
        [
            "--familiarity-max-false-review-rate",
            str(args.minikand_familiarity_max_false_review_rate),
        ]
    )
    if args.metabears_max_batches is not None:
        command.extend(["--max-batches", str(args.metabears_max_batches)])
    _run(command, args.repo_root / "XOR_MNIST")


def metabears_bdd(args):
    if not args.bdd_metabears_data_dir:
        raise SystemExit("Pass the preprocessed feature directory with --bdd-metabears-data-dir.")
    if not args.bdd_metabears_checkpoints or len(args.bdd_metabears_checkpoints) < 2:
        raise SystemExit(
            "Pass at least two same-variant checkpoints with "
            "--bdd-metabears-checkpoints."
        )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.bdd_metabears_output_dir
    if output_dir is None:
        output_dir = (
            args.repo_root / "colab_outputs" / "metabears_bdd" / timestamp
        )
    command = [
        sys.executable,
        "-m",
        "metacog.bdd_runner",
        "--bdd-data-dir",
        args.bdd_metabears_data_dir,
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.bdd_metabears_batch_size),
        "--familiarity-validation-quantile",
        str(args.familiarity_validation_quantile),
        "--shortcut-fallback-quantile",
        str(args.shortcut_fallback_quantile),
        "--ece-bins",
        str(args.ece_bins),
        "--ensemble-checkpoints",
        *args.bdd_metabears_checkpoints,
    ]
    if args.bdd_metabears_compositional_ood:
        command.append("--compositional-ood")
    if args.metabears_max_batches is not None:
        command.extend(["--max-batches", str(args.metabears_max_batches)])
    _run(command, args.repo_root / "XOR_MNIST")


def bdd_ood_split(args):
    if not args.bdd_metabears_data_dir:
        raise SystemExit(
            "Pass the preprocessed feature directory with --bdd-metabears-data-dir."
        )
    command = [
        sys.executable,
        "-m",
        "metacog.bdd_ood",
        "--bdd-data-dir",
        args.bdd_metabears_data_dir,
        "--max-fraction",
        str(args.bdd_ood_max_fraction),
    ]
    if args.bdd_ood_output_summary:
        command.extend(["--output-summary", args.bdd_ood_output_summary])
    _run(command, args.repo_root / "XOR_MNIST")


def bdd_freeze(args):
    for name, value in (
        ("--bdd-freeze-results-root", args.bdd_freeze_results_root),
        ("--bdd-freeze-id", args.bdd_freeze_id),
        ("--bdd-freeze-scope", args.bdd_freeze_scope),
        ("--bdd-freeze-output", args.bdd_freeze_output),
    ):
        if not value:
            raise SystemExit(f"Pass {name}.")

    results_root = Path(args.bdd_freeze_results_root)
    variant_dirs = {
        "base": results_root / "dpl_auc",
        "entropy": results_root / "dpl_auc_entropy",
        "csup": results_root / "dpl_auc_csup",
    }
    command = [
        sys.executable,
        "bdd_freeze.py",
        "--freeze-id",
        args.bdd_freeze_id,
        "--scope",
        args.bdd_freeze_scope,
        "--output",
        args.bdd_freeze_output,
    ]
    found_any = False
    for name, directory in variant_dirs.items():
        if (directory / "run_summary.json").is_file():
            command.extend(["--variant", name, str(directory)])
            found_any = True
    if not found_any:
        raise SystemExit(
            f"No run_summary.json found under dpl_auc/dpl_auc_entropy/dpl_auc_csup "
            f"in {results_root}."
        )
    _run(command, args.repo_root)


def minikand_representation_sweep(args):
    ensure_kandinsky_data(args.repo_root)
    if not args.minikand_checkpoints or len(args.minikand_checkpoints) < 2:
        raise SystemExit(
            "Pass at least two trained members with --minikand-checkpoints."
        )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.minikand_output_dir
    if output_dir is None:
        output_dir = (
            args.repo_root
            / "colab_outputs"
            / "minikandinsky_representation_sweep"
            / timestamp
        )
    command = [
        sys.executable,
        "-m",
        "metacog.minikandinsky_representation_sweep",
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--c-sup",
        str(args.minikand_c_sup),
        "--w-c",
        str(args.minikand_w_c),
        "--max-false-review-rate",
        str(args.minikand_familiarity_max_false_review_rate),
        "--minimum-auroc",
        str(args.minikand_sweep_minimum_auroc),
        "--minimum-average-precision",
        str(args.minikand_sweep_minimum_average_precision),
        "--minimum-recall",
        str(args.minikand_sweep_minimum_recall),
        "--ensemble-checkpoints",
        *args.minikand_checkpoints,
    ]
    if args.minikand_entropy:
        command.extend(
            [
                "--training-entropy",
                "--training-entropy-weight",
                str(args.minikand_w_h),
            ]
        )
    if args.metabears_max_batches is not None:
        command.extend(["--max-batches", str(args.metabears_max_batches)])
    _run(command, args.repo_root / "XOR_MNIST")


def minikand_scoring_sweep(args, *, uncertainty_ablation=False):
    ensure_kandinsky_data(args.repo_root)
    if not args.minikand_checkpoints or len(args.minikand_checkpoints) < 2:
        raise SystemExit(
            "Pass at least two trained members with --minikand-checkpoints."
        )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.minikand_output_dir
    if output_dir is None:
        output_dir = (
            args.repo_root
            / "colab_outputs"
            / (
                "minikandinsky_uncertainty_ablation"
                if uncertainty_ablation
                else "minikandinsky_scoring_sweep"
            )
            / timestamp
        )
    command = [
        sys.executable,
        "-m",
        "metacog.minikandinsky_scoring_sweep",
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--c-sup",
        str(args.minikand_c_sup),
        "--w-c",
        str(args.minikand_w_c),
        "--representation-key",
        args.minikand_representation_key,
        "--normalization",
        args.minikand_representation_normalization,
        "--cross-fit-folds",
        str(args.minikand_cross_fit_folds),
        "--shrinkage",
        str(args.minikand_shrinkage),
        "--max-false-review-rate",
        str(args.minikand_familiarity_max_false_review_rate),
        "--minimum-auroc",
        str(args.minikand_sweep_minimum_auroc),
        "--minimum-average-precision",
        str(args.minikand_sweep_minimum_average_precision),
        "--minimum-recall",
        str(args.minikand_sweep_minimum_recall),
        "--ensemble-checkpoints",
        *args.minikand_checkpoints,
    ]
    if uncertainty_ablation:
        command.extend(
            [
                "--analysis-mode",
                "uncertainty_ablation",
                "--scorers",
                "class_conditional_disagreement_fusion",
                "label_disagreement",
                "predictive_entropy",
                "confidence_deficit",
            ]
        )
    if args.metabears_max_batches is not None:
        command.extend(["--max-batches", str(args.metabears_max_batches)])
    _run(command, args.repo_root / "XOR_MNIST")


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
        "--feature-workers",
        str(args.feature_workers),
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
            "metabears_demo",
            "metabears_halfmnist",
            "metabears_bdd",
            "bdd_ood_split",
            "bdd_freeze",
            "minikand_smoke",
            "minikand_train",
            "metabears_minikandinsky",
            "minikand_representation_sweep",
            "minikand_scoring_sweep",
            "minikand_uncertainty_ablation",
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
        "--metabears-checkpoints",
        nargs="*",
        default=None,
        help="Explicit HalfMNIST ensemble checkpoints for MetaBEARS.",
    )
    parser.add_argument(
        "--metabears-train-ensemble",
        action="store_true",
        help="Train the HalfMNIST ensemble before the MetaBEARS run.",
    )
    parser.add_argument(
        "--metabears-ensemble-kind",
        choices=["bears", "ensemble"],
        default="bears",
    )
    parser.add_argument("--metabears-real-kl", action="store_true")
    parser.add_argument("--metabears-max-batches", type=int, default=None)
    parser.add_argument(
        "--bdd-metabears-data-dir",
        default=None,
        help=(
            "Preprocessed BDD-OIA feature directory (e.g. "
            "data/bdd2048_resnet); required for metabears_bdd."
        ),
    )
    parser.add_argument(
        "--bdd-metabears-checkpoints",
        nargs="*",
        default=None,
        help=(
            "Explicit model_best-<seed>.pth.tar checkpoints from the SAME "
            "trained BDD-OIA variant (do not mix base/entropy/csup)."
        ),
    )
    parser.add_argument(
        "--bdd-metabears-output-dir",
        default=None,
        help="Optional BDD-OIA MetaBEARS artifact directory.",
    )
    parser.add_argument(
        "--bdd-metabears-batch-size",
        type=int,
        default=64,
        help="Evaluation batch size for the BDD-OIA MetaBEARS run.",
    )
    parser.add_argument(
        "--bdd-metabears-compositional-ood",
        action="store_true",
        help=(
            "Evaluate the frozen compositional OOD split instead of the "
            "plain val/test splits. Generate the split first with the "
            "bdd_ood_split job."
        ),
    )
    parser.add_argument(
        "--bdd-ood-max-fraction",
        type=float,
        default=0.1,
        help=(
            "Maximum cumulative fraction of training samples the frozen "
            "rare combined-action set may cover (bdd_ood_split job)."
        ),
    )
    parser.add_argument(
        "--bdd-ood-output-summary",
        default=None,
        help="Optional JSON summary path for the bdd_ood_split job.",
    )
    parser.add_argument(
        "--bdd-freeze-results-root",
        default=None,
        help=(
            "Directory containing dpl_auc/, dpl_auc_entropy/, dpl_auc_csup/ "
            "MetaBEARS output subdirectories to freeze (bdd_freeze job)."
        ),
    )
    parser.add_argument(
        "--bdd-freeze-id",
        default=None,
        help="Identifier for this BDD-OIA freeze (bdd_freeze job).",
    )
    parser.add_argument(
        "--bdd-freeze-scope",
        default=None,
        help="One-sentence description of what this freeze covers (bdd_freeze job).",
    )
    parser.add_argument(
        "--bdd-freeze-output",
        default=None,
        help="Output path for the freeze manifest JSON (bdd_freeze job).",
    )
    parser.add_argument(
        "--minikand-checkpoints",
        nargs="*",
        default=None,
        help="Explicit trained MiniKandinsky checkpoints for MetaBEARS.",
    )
    parser.add_argument(
        "--minikand-output-dir",
        default=None,
        help="Optional MiniKandinsky MetaBEARS artifact directory.",
    )
    parser.add_argument(
        "--minikand-intervention",
        choices=["none", "figure_permute", "palette_cycle"],
        default="figure_permute",
    )
    parser.add_argument(
        "--minikand-c-sup",
        type=float,
        default=1.0,
        help="MiniKandinsky concept-supervision fraction for training/loading.",
    )
    parser.add_argument(
        "--minikand-w-c",
        type=float,
        default=10.0,
        help="MiniKandinsky concept-loss weight for training/loading.",
    )
    parser.add_argument(
        "--minikand-checkpoint-tag",
        default=None,
        choices=["v1-task-loss", "v2-entropy-task-loss"],
        help="Optional protocol tag used to preserve earlier checkpoints.",
    )
    parser.add_argument(
        "--minikand-entropy",
        action="store_true",
        help="Enable the MiniKandinsky concept-balance entropy regularizer.",
    )
    parser.add_argument(
        "--minikand-w-h",
        type=float,
        default=1.0,
        help="MiniKandinsky entropy-loss weight.",
    )
    parser.add_argument(
        "--minikand-representation-key",
        default="CS",
        help="MiniKandinsky output used for familiarity distances.",
    )
    parser.add_argument(
        "--minikand-representation-normalization",
        choices=["none", "zscore", "l2", "zscore_l2"],
        default="none",
    )
    parser.add_argument(
        "--minikand-ood-validation-transform",
        choices=["none", "palette_desaturate", "palette_pastel"],
        default="none",
    )
    parser.add_argument(
        "--minikand-ood-transform",
        choices=["none", "palette_desaturate", "palette_pastel"],
        default="palette_desaturate",
    )
    parser.add_argument(
        "--minikand-shortcut-max-false-review-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--minikand-familiarity-max-false-review-rate",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--minikand-sweep-minimum-auroc",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--minikand-sweep-minimum-average-precision",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--minikand-sweep-minimum-recall",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--minikand-cross-fit-folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--minikand-shrinkage",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--metabears-intervention",
        choices=[
            "none",
            "half_swap",
            "patch_neutral",
            "patch_conflict",
            "patch_removed",
            "patch_shuffled",
        ],
        default="none",
    )
    parser.add_argument(
        "--metabears-shortcut-patch-training",
        action="store_true",
    )
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--n-ensembles", type=int, default=5)
    parser.add_argument("--lambda-h", type=float, default=1.0)
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
    parser.add_argument(
        "--feature-workers",
        type=int,
        default=4,
        help="DataLoader worker processes for parallel BDD feature extraction.",
    )
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
        "metabears_demo": metabears_demo,
        "metabears_halfmnist": metabears_halfmnist,
        "metabears_bdd": metabears_bdd,
        "bdd_ood_split": bdd_ood_split,
        "bdd_freeze": bdd_freeze,
        "minikand_smoke": minikand_smoke,
        "minikand_train": minikand_train,
        "metabears_minikandinsky": metabears_minikandinsky,
        "minikand_representation_sweep": minikand_representation_sweep,
        "minikand_scoring_sweep": minikand_scoring_sweep,
        "minikand_uncertainty_ablation": lambda parsed: minikand_scoring_sweep(
            parsed, uncertainty_ablation=True
        ),
        "bdd_preprocess_smoke": lambda parsed: bdd_preprocess(parsed, full=False),
        "bdd_train_smoke": lambda parsed: bdd_train(parsed, full=False),
        "bdd_preprocess_full": lambda parsed: bdd_preprocess(parsed, full=True),
        "bdd_train_full": lambda parsed: bdd_train(parsed, full=True),
        "archive_results": archive_results,
    }
    jobs[args.job](args)


if __name__ == "__main__":
    main()
