"""Validate and aggregate frozen MetaBEARS runs across independent seeds."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from XOR_MNIST.metacog.protocol import (
    FrozenProtocol,
    load_protocol,
    validate_protocol_configuration,
)


METRIC_FIELDS = (
    "id_base_task_accuracy",
    "id_perturbed_task_accuracy",
    "id_accuracy_drop",
    "mismatch_normalized_accuracy_drop",
    "task_failure_prevalence",
    "task_failure_precision",
    "task_failure_recall",
    "task_failure_f1",
    "semantic_instability_prevalence",
    "semantic_instability_auroc",
    "semantic_instability_average_precision",
    "semantic_instability_f1",
    "review_rate",
    "coverage",
    "ood_auroc",
    "ood_average_precision",
    "ood_f1",
    "effective_patch_mismatch_rate",
)

OOD_METRIC_FIELDS = (
    "ood_auroc",
    "ood_average_precision",
    "ood_f1",
)

# Two-sided 95% Student-t critical values. Frozen experiments currently use
# three independent seeds (two degrees of freedom), but the complete table
# keeps partial and future reporting statistically well-defined.
_T_CRITICAL_975 = (
    None,
    12.706204736,
    4.302652730,
    3.182446305,
    2.776445105,
    2.570581836,
    2.446911851,
    2.364624252,
    2.306004135,
    2.262157163,
    2.228138852,
    2.200985160,
    2.178812830,
    2.160368656,
    2.144786688,
    2.131449546,
    2.119905299,
    2.109815578,
    2.100922040,
    2.093024054,
    2.085963447,
    2.079613845,
    2.073873068,
    2.068657610,
    2.063898562,
    2.059538553,
    2.055529439,
    2.051830516,
    2.048407142,
    2.045229642,
    2.042272456,
)


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _artifact_fingerprint(records: Any, *, name: str) -> str:
    if not isinstance(records, list) or not records:
        raise ValueError(f"Run provenance is missing {name} records.")
    digests = []
    for record in records:
        if not isinstance(record, Mapping) or not record.get("exists"):
            raise ValueError(f"Run provenance contains a missing {name} artifact.")
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Run provenance contains an invalid {name} SHA-256.")
        digests.append(digest)
    return "|".join(sorted(digests))


def extract_run_row(
    summary: Mapping[str, Any], summary_path: Path
) -> Dict[str, Any]:
    configuration = summary["configuration"]
    intervention = summary["intervention"]
    id_metrics = intervention["id_test"]
    task_detection = id_metrics["task_invariance_failure_shortcut_flags"]
    semantic_detection = id_metrics["semantic_instability_detection"]
    base_accuracy = float(id_metrics["base_task_accuracy"])
    perturbed_accuracy = float(id_metrics["perturbed_task_accuracy"])
    accuracy_drop = base_accuracy - perturbed_accuracy
    mismatch_rate = _nested(
        id_metrics, "input_assignment", "effective_mismatch_rate"
    )
    normalized_drop = (
        accuracy_drop / float(mismatch_rate)
        if mismatch_rate is not None and float(mismatch_rate) > 0.0
        else None
    )
    provenance = summary["provenance"]
    return {
        "summary_path": str(summary_path.resolve()),
        "seed": int(configuration["seed"]),
        "intervention": str(intervention["name"]),
        "git_commit": _nested(provenance, "git", "commit"),
        "dataset_fingerprint": _artifact_fingerprint(
            provenance.get("dataset_artifacts"), name="dataset"
        ),
        "checkpoint_fingerprint": _artifact_fingerprint(
            provenance.get("checkpoints"), name="checkpoint"
        ),
        "id_samples": int(id_metrics["samples"]),
        "id_base_task_accuracy": base_accuracy,
        "id_perturbed_task_accuracy": perturbed_accuracy,
        "id_accuracy_drop": accuracy_drop,
        "mismatch_normalized_accuracy_drop": normalized_drop,
        "task_failure_count": int(task_detection["positive_count"]),
        "task_failure_prevalence": task_detection["prevalence"],
        "task_failure_precision": task_detection["precision"],
        "task_failure_recall": task_detection["recall"],
        "task_failure_f1": task_detection["f1"],
        "semantic_instability_count": int(semantic_detection["positive_count"]),
        "semantic_instability_prevalence": semantic_detection["prevalence"],
        "semantic_instability_auroc": semantic_detection["auroc"],
        "semantic_instability_average_precision": semantic_detection[
            "average_precision"
        ],
        "semantic_instability_f1": semantic_detection["f1"],
        "review_rate": summary["splits"]["id_test"]["review_rate"],
        "coverage": summary["splits"]["id_test"]["coverage"],
        "ood_auroc": _nested(summary, "ood_detection", "auroc"),
        "ood_average_precision": _nested(
            summary, "ood_detection", "average_precision"
        ),
        "ood_f1": _nested(summary, "ood_detection", "f1"),
        "effective_patch_mismatch_rate": mismatch_rate,
    }


def _validate_summary(
    protocol: FrozenProtocol,
    summary: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_intervention: str,
) -> None:
    configuration = summary.get("configuration", {})
    validate_protocol_configuration(protocol, configuration)
    provenance = summary.get("provenance", {})
    protocol_record = provenance.get("protocol")
    if not isinstance(protocol_record, Mapping):
        raise ValueError("Run is missing frozen protocol provenance.")
    if protocol_record.get("id") != protocol.protocol_id:
        raise ValueError("Run protocol ID does not match the aggregator protocol.")
    if protocol_record.get("sha256") != protocol.sha256:
        raise ValueError("Run protocol SHA-256 does not match the frozen manifest.")
    if configuration.get("seed") != expected_seed:
        raise ValueError("Run seed does not match its structured directory.")
    if configuration.get("intervention") != expected_intervention:
        raise ValueError("Run intervention does not match its directory.")
    expected_members = int(protocol.data["ensemble"]["members"])
    checkpoint_records = provenance.get("checkpoints")
    if not isinstance(checkpoint_records, list) or len(checkpoint_records) != expected_members:
        raise ValueError(
            f"Run must contain provenance for {expected_members} checkpoints."
        )
    if expected_intervention == protocol.data["evaluation"]["primary_intervention"]:
        mismatch_rate = _nested(
            summary,
            "intervention",
            "id_test",
            "input_assignment",
            "effective_mismatch_rate",
        )
        if mismatch_rate is None:
            raise ValueError("Primary shuffled run is missing assignment metrics.")


def discover_rows(
    protocol: FrozenProtocol,
    results_root: Path,
    *,
    seeds: Sequence[int],
    interventions: Sequence[str],
    allow_partial: bool,
) -> List[Dict[str, Any]]:
    rows = []
    missing = []
    for seed in seeds:
        for intervention in interventions:
            summary_path = (
                results_root
                / f"seed_{seed}"
                / intervention
                / "run_summary.json"
            )
            if not summary_path.is_file():
                missing.append(str(summary_path))
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            _validate_summary(
                protocol,
                summary,
                expected_seed=seed,
                expected_intervention=intervention,
            )
            rows.append(extract_run_row(summary, summary_path))
    if missing and not allow_partial:
        raise FileNotFoundError(
            "Frozen matrix is incomplete:\n" + "\n".join(f"  - {p}" for p in missing)
        )
    if not rows:
        raise ValueError("No valid frozen run summaries were found.")
    dataset_fingerprints = {row["dataset_fingerprint"] for row in rows}
    if len(dataset_fingerprints) != 1:
        raise ValueError("Runs do not share the same dataset SHA-256 fingerprints.")
    for seed in seeds:
        seed_fingerprints = {
            row["checkpoint_fingerprint"] for row in rows if row["seed"] == seed
        }
        if len(seed_fingerprints) > 1:
            raise ValueError(
                f"Controls for seed {seed} do not share identical checkpoints."
            )
    fingerprint_seeds: Dict[str, int] = {}
    for row in rows:
        fingerprint = str(row["checkpoint_fingerprint"])
        seed = int(row["seed"])
        previous_seed = fingerprint_seeds.setdefault(fingerprint, seed)
        if previous_seed != seed:
            raise ValueError(
                f"Seeds {previous_seed} and {seed} reuse identical checkpoints."
            )
    return rows


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    interventions = sorted({str(row["intervention"]) for row in rows})
    aggregates = []
    for intervention in interventions:
        selected = [row for row in rows if row["intervention"] == intervention]
        for metric in METRIC_FIELDS:
            values = [
                float(row[metric])
                for row in selected
                if row.get(metric) is not None
            ]
            aggregates.append(
                _aggregate_values(
                    values,
                    grouping_key="intervention",
                    grouping_value=intervention,
                    metric=metric,
                )
            )
    return aggregates


def _t_critical_975(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive.")
    if degrees_of_freedom < len(_T_CRITICAL_975):
        return float(_T_CRITICAL_975[degrees_of_freedom])
    # The normal critical value is a close approximation once the number of
    # independent runs is larger than the frozen study matrix.
    return 1.959963985


def _aggregate_values(
    values: Sequence[float],
    *,
    grouping_key: str,
    grouping_value: str,
    metric: str,
) -> Dict[str, Any]:
    count = len(values)
    average = mean(values) if values else None
    sample_std = stdev(values) if count > 1 else None
    standard_error = (
        sample_std / math.sqrt(count) if sample_std is not None else None
    )
    margin = (
        _t_critical_975(count - 1) * standard_error
        if standard_error is not None
        else None
    )
    return {
        grouping_key: grouping_value,
        "metric": metric,
        "n": count,
        "mean": average,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "ci95_low": average - margin if margin is not None else None,
        "ci95_high": average + margin if margin is not None else None,
        "ci_method": (
            "two-sided Student-t over independent seeds"
            if margin is not None
            else None
        ),
    }


def aggregate_unique_models(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Aggregate OOD metrics once per independently trained ensemble.

    Intervention controls reuse identical checkpoints, so counting their OOD
    results separately would duplicate the same model-level observation.
    """

    unique: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        fingerprint = str(row["checkpoint_fingerprint"])
        candidate = {
            "seed": int(row["seed"]),
            "git_commit": row.get("git_commit"),
            "dataset_fingerprint": row.get("dataset_fingerprint"),
            "checkpoint_fingerprint": fingerprint,
            **{metric: row.get(metric) for metric in OOD_METRIC_FIELDS},
        }
        previous = unique.get(fingerprint)
        if previous is not None:
            if previous["seed"] != candidate["seed"]:
                raise ValueError(
                    "Independent seeds reuse an identical checkpoint fingerprint."
                )
            for metric in OOD_METRIC_FIELDS:
                if previous.get(metric) != candidate.get(metric):
                    raise ValueError(
                        "Controls sharing checkpoints contain inconsistent "
                        f"{metric} values."
                    )
        else:
            unique[fingerprint] = candidate

    model_rows = sorted(unique.values(), key=lambda row: int(row["seed"]))
    aggregates = []
    for metric in OOD_METRIC_FIELDS:
        values = [
            float(row[metric]) for row in model_rows if row.get(metric) is not None
        ]
        aggregates.append(
            _aggregate_values(
                values,
                grouping_key="unit",
                grouping_value="unique_ensemble",
                metric=metric,
            )
        )
    return model_rows, aggregates


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_aggregates(
    output_directory: Path,
    aggregates: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib is unavailable; aggregate plots were skipped."

    metrics = (
        ("id_accuracy_drop", "ID accuracy drop (%)"),
        ("task_failure_f1", "Task-failure F1 (%)"),
        ("semantic_instability_auroc", "Semantic AUROC (%)"),
        ("review_rate", "Review rate (%)"),
    )
    interventions = sorted({str(row["intervention"]) for row in aggregates})
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (metric, title) in zip(axes.flat, metrics):
        metric_rows = {
            row["intervention"]: row
            for row in aggregates
            if row["metric"] == metric
        }
        values = [100.0 * metric_rows[name]["mean"] for name in interventions]
        errors = [
            100.0 * (metric_rows[name]["sample_std"] or 0.0)
            for name in interventions
        ]
        positions = list(range(len(interventions)))
        axis.bar(positions, values, yerr=errors, capsize=4, alpha=0.75)
        for position, name in zip(positions, interventions):
            seed_values = [
                100.0 * float(row[metric])
                for row in rows
                if row["intervention"] == name and row.get(metric) is not None
            ]
            axis.scatter(
                [position] * len(seed_values),
                seed_values,
                color="black",
                marker="o",
                s=24,
                zorder=3,
            )
        axis.set_title(title)
        axis.set_xticks(positions, interventions, rotation=20)
        axis.set_ylim(bottom=0.0)
        if metric in {"semantic_instability_auroc", "review_rate"}:
            axis.set_ylim(0.0, 100.0)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Mean +/- sample SD; points are independent ensemble seeds",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(output_directory / "aggregate_metrics.png", dpi=180)
    plt.close(figure)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate a frozen MetaBEARS result matrix."
    )
    parser.add_argument("--protocol", default="experiment_protocol.json")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--interventions", nargs="+", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    protocol = load_protocol(repo_root / args.protocol)
    results_root = Path(args.results_root).expanduser().resolve()
    output_directory = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else results_root / "aggregate"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    evaluation = protocol.data["evaluation"]
    seeds = args.seeds or list(protocol.data["ensemble"]["base_seeds"])
    interventions = args.interventions or [
        evaluation["primary_intervention"],
        *evaluation.get("secondary_interventions", []),
    ]
    rows = discover_rows(
        protocol,
        results_root,
        seeds=seeds,
        interventions=interventions,
        allow_partial=args.allow_partial,
    )
    aggregates = aggregate_rows(rows)
    model_rows, model_aggregates = aggregate_unique_models(rows)
    _write_csv(output_directory / "run_results.csv", rows)
    _write_csv(output_directory / "aggregate_results.csv", aggregates)
    _write_csv(output_directory / "model_results.csv", model_rows)
    _write_csv(output_directory / "model_aggregate_results.csv", model_aggregates)
    warning = _plot_aggregates(output_directory, aggregates, rows)
    payload = {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "runs": rows,
        "aggregates": aggregates,
        "unique_models": model_rows,
        "model_aggregates": model_aggregates,
        "warning": warning,
    }
    (output_directory / "aggregate_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Aggregated {len(rows)} runs into: {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
