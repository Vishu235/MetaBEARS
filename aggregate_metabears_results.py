"""Validate and aggregate frozen MetaBEARS runs across independent seeds."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import math
import platform
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

from XOR_MNIST.metacog.posthoc import (
    DETECTOR_DESCRIPTIONS,
    TARGET_DEFINITIONS,
    calibrate_fusion_references_from_result_directory,
    evaluate_fusion_result_directory,
    evaluate_result_directory,
    fit_leave_one_intervention_out_fusion_from_result_directories,
    fit_validation_fusion_from_result_directory,
)
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

PAIRED_METRIC_FIELDS = (
    "id_accuracy_drop",
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
)

DETECTOR_AGGREGATE_METRICS = (
    "prevalence",
    "auroc",
    "average_precision",
    "aurc",
    "risk_at_80_coverage",
    "review_rate_at_95_recall",
)

DETECTOR_COMPARISON_METRICS = tuple(
    metric for metric in DETECTOR_AGGREGATE_METRICS if metric != "prevalence"
)

DETECTOR_COMPARISON_BASELINES = (
    "full_metabears",
    "validation_fitted_fusion_v2",
    "intervention_calibrated_fusion_v3",
    "perturbation_js",
    "task_distribution_js",
    "concept_instability_with_perturbation",
    "concept_instability_without_perturbation",
)

DETECTOR_COMPARISON_CANDIDATES = (
    "full_metabears",
    "validation_fitted_fusion_v2",
    "intervention_calibrated_fusion_v3",
    "leave_one_intervention_out_fusion_v4",
    "external_negative_control_fusion_v5",
)

FUSION_THRESHOLD_METRICS = (
    "review_rate",
    "precision",
    "recall",
    "f1",
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


def load_analysis_protocol(
    path: Path, base_protocol: FrozenProtocol
) -> FrozenProtocol:
    """Load and validate the validation-only post-hoc analysis manifest."""

    resolved = path.expanduser().resolve()
    content = resolved.read_bytes()
    data = json.loads(content.decode("utf-8"))
    required = {
        "protocol_id",
        "base_protocol_id",
        "base_protocol_sha256",
        "fit_split",
        "fit_intervention",
        "fit_target",
        "evaluation_split",
        "signals",
        "weight_grid_step",
        "cross_validation_folds",
        "threshold_target_recall",
        "test_labels_used_for_fitting",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(
            "Analysis protocol is missing required fields: " + ", ".join(missing)
        )
    if data["base_protocol_id"] != base_protocol.protocol_id:
        raise ValueError("Analysis protocol references a different base protocol ID.")
    if data["base_protocol_sha256"] != base_protocol.sha256:
        raise ValueError("Analysis protocol references a different base protocol hash.")
    if data["fit_split"] != "validation" or data["evaluation_split"] != "id_test":
        raise ValueError("Analysis fitting must use validation and evaluate ID test.")
    if data["test_labels_used_for_fitting"] is not False:
        raise ValueError("Analysis protocol must prohibit test-label fitting.")
    signals = data["signals"]
    if not isinstance(signals, list) or not signals:
        raise ValueError("Analysis protocol signals must be a non-empty list.")
    protocol = FrozenProtocol(
        path=resolved,
        data=data,
        sha256=hashlib.sha256(content).hexdigest(),
    )
    parent_file = data.get("parent_analysis_protocol_file")
    if parent_file is not None:
        parent_path = (resolved.parent / str(parent_file)).resolve()
        parent = load_analysis_protocol(parent_path, base_protocol)
        if data.get("parent_analysis_protocol_id") != parent.protocol_id:
            raise ValueError("Analysis protocol references a different parent ID.")
        if data.get("parent_analysis_protocol_sha256") != parent.sha256:
            raise ValueError("Analysis protocol references a different parent hash.")
        leave_one_out = data.get("outer_evaluation") == "leave_one_intervention_out"
        external_negative_control = (
            data.get("outer_evaluation") == "external_negative_control"
        )
        if leave_one_out:
            if (
                data.get("held_out_intervention_validation_used_for_fitting")
                is not False
            ):
                raise ValueError(
                    "Held-out intervention validation data must be excluded."
                )
            if data.get("held_out_intervention_labels_used_for_fitting") is not False:
                raise ValueError(
                    "Held-out intervention labels must be excluded."
                )
            interventions = data.get("evaluation_interventions")
            if not isinstance(interventions, list) or len(interventions) < 3:
                raise ValueError(
                    "Leave-one-intervention-out analysis requires at least "
                    "three interventions."
                )
            evaluation = base_protocol.data["evaluation"]
            allowed = {
                evaluation["primary_intervention"],
                *evaluation.get("secondary_interventions", []),
                *evaluation.get("supplementary_interventions", []),
                evaluation["negative_control"],
            }
            unsupported = sorted(set(interventions).difference(allowed))
            if unsupported:
                raise ValueError(
                    f"Analysis protocol has unfrozen interventions: {unsupported}"
                )
            inherited_fields = (
                "fit_split",
                "fit_target",
                "evaluation_split",
                "signals",
                "weight_grid_step",
                "threshold_target_recall",
            )
        elif external_negative_control:
            if data.get("negative_control_validation_used_for_fitting") is not False:
                raise ValueError(
                    "Negative-control validation data must be excluded from fitting."
                )
            if data.get("negative_control_labels_used_for_fitting") is not False:
                raise ValueError(
                    "Negative-control labels must be excluded from fitting."
                )
            training_interventions = data.get("training_interventions")
            if (
                not isinstance(training_interventions, list)
                or len(training_interventions) < 3
                or len(set(training_interventions)) != len(training_interventions)
            ):
                raise ValueError(
                    "External negative-control analysis requires at least three "
                    "unique training interventions."
                )
            evaluation = base_protocol.data["evaluation"]
            negative_control = data.get("negative_control_intervention")
            if negative_control != evaluation["negative_control"]:
                raise ValueError(
                    "Analysis negative control does not match the base protocol."
                )
            if negative_control in training_interventions:
                raise ValueError(
                    "Negative control must be excluded from training interventions."
                )
            allowed = {
                evaluation["primary_intervention"],
                *evaluation.get("secondary_interventions", []),
                *evaluation.get("supplementary_interventions", []),
            }
            unsupported = sorted(set(training_interventions).difference(allowed))
            if unsupported:
                raise ValueError(
                    f"Negative-control protocol has unfrozen training "
                    f"interventions: {unsupported}"
                )
            evaluation_interventions = data.get("evaluation_interventions")
            expected_evaluations = {*training_interventions, negative_control}
            if (
                not isinstance(evaluation_interventions, list)
                or len(evaluation_interventions) != len(expected_evaluations)
                or set(evaluation_interventions) != expected_evaluations
            ):
                raise ValueError(
                    "Negative-control evaluation must contain every training "
                    "intervention and the external control exactly once."
                )
            if int(data.get("cross_validation_folds", 0)) != len(
                training_interventions
            ):
                raise ValueError(
                    "Negative-control cross-validation folds must match the "
                    "number of training interventions."
                )
            inherited_fields = (
                "fit_split",
                "fit_target",
                "evaluation_split",
                "signals",
                "weight_grid_step",
                "threshold_target_recall",
                "selection_metric",
                "selection_tiebreakers",
            )
        else:
            if data.get("normalization_labels_used") is not False:
                raise ValueError(
                    "Conditioned normalization must prohibit validation-label use."
                )
            if data.get("secondary_intervention_labels_used_for_fitting") is not False:
                raise ValueError(
                    "Secondary intervention labels must not be used for fitting."
                )
            inherited_fields = (
                "fit_split",
                "fit_intervention",
                "fit_target",
                "evaluation_split",
                "signals",
                "weight_grid_step",
                "cross_validation_folds",
                "threshold_target_recall",
            )
        changed = [
            name
            for name in inherited_fields
            if data.get(name) != parent.data.get(name)
        ]
        if changed:
            raise ValueError(
                "Analysis protocol changes frozen parent fitting fields: "
                + ", ".join(changed)
            )
    return protocol


def analysis_protocol_chain(
    active: FrozenProtocol, base_protocol: FrozenProtocol
) -> List[FrozenProtocol]:
    """Return analysis manifests from the oldest parent to the active one."""

    chain = [active]
    visited = {active.path}
    current = active
    while current.data.get("parent_analysis_protocol_file") is not None:
        parent_path = (
            current.path.parent
            / str(current.data["parent_analysis_protocol_file"])
        ).resolve()
        if parent_path in visited:
            raise ValueError("Analysis protocol parent chain contains a cycle.")
        current = load_analysis_protocol(parent_path, base_protocol)
        chain.append(current)
        visited.add(parent_path)
    chain.reverse()
    return chain


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


def paired_control_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_intervention: str,
    comparator_interventions: Sequence[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compare controls within seed so shared ensembles remain paired."""

    indexed = {
        (int(row["seed"]), str(row["intervention"])): row for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    paired_rows: List[Dict[str, Any]] = []
    aggregate_rows_output: List[Dict[str, Any]] = []
    for comparator in comparator_interventions:
        for seed in seeds:
            primary = indexed.get((seed, primary_intervention))
            secondary = indexed.get((seed, comparator))
            if primary is None or secondary is None:
                continue
            if primary["checkpoint_fingerprint"] != secondary["checkpoint_fingerprint"]:
                raise ValueError(
                    f"Paired controls for seed {seed} do not share checkpoints."
                )
            for metric in PAIRED_METRIC_FIELDS:
                primary_value = primary.get(metric)
                comparator_value = secondary.get(metric)
                if primary_value is None or comparator_value is None:
                    continue
                paired_rows.append(
                    {
                        "seed": seed,
                        "primary_intervention": primary_intervention,
                        "comparator_intervention": comparator,
                        "metric": metric,
                        "primary_value": float(primary_value),
                        "comparator_value": float(comparator_value),
                        "paired_difference": (
                            float(primary_value) - float(comparator_value)
                        ),
                    }
                )

        for metric in PAIRED_METRIC_FIELDS:
            selected = [
                row
                for row in paired_rows
                if row["comparator_intervention"] == comparator
                and row["metric"] == metric
            ]
            values = [float(row["paired_difference"]) for row in selected]
            if not values:
                continue
            aggregate = _aggregate_values(
                values,
                grouping_key="comparison",
                grouping_value=f"{primary_intervention}_minus_{comparator}",
                metric=metric,
            )
            aggregate.update(
                {
                    "primary_intervention": primary_intervention,
                    "comparator_intervention": comparator,
                }
            )
            aggregate_rows_output.append(aggregate)
    return paired_rows, aggregate_rows_output


def aggregate_detector_results(
    detector_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate threshold-free detector results across independent seeds."""

    groups = sorted(
        {
            (
                str(row["intervention"]),
                str(row["target"]),
                str(row["detector"]),
            )
            for row in detector_rows
        }
    )
    output: List[Dict[str, Any]] = []
    for intervention, target, detector in groups:
        selected = [
            row
            for row in detector_rows
            if row["intervention"] == intervention
            and row["target"] == target
            and row["detector"] == detector
        ]
        for metric in DETECTOR_AGGREGATE_METRICS:
            values = [
                float(row[metric])
                for row in selected
                if row.get(metric) is not None
            ]
            aggregate = _aggregate_values(
                values,
                grouping_key="intervention",
                grouping_value=intervention,
                metric=metric,
            )
            aggregate.update({"target": target, "detector": detector})
            output.append(aggregate)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reporting_provenance(
    repo_root: Path,
    *,
    analysis_protocol_paths: Sequence[Path] | None = None,
) -> Dict[str, Any]:
    """Record which reporting code produced an aggregate result bundle."""

    git_metadata: Dict[str, Any] = {"commit": None, "dirty": None}
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
        git_metadata = {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        pass

    protocol_paths = list(
        analysis_protocol_paths
        or (
            repo_root / "analysis_protocol_v2.json",
            repo_root / "analysis_protocol_v3.json",
            repo_root / "analysis_protocol_v4.json",
        )
    )
    source_paths = [
        repo_root / "aggregate_metabears_results.py",
        *protocol_paths,
        repo_root / "XOR_MNIST" / "metacog" / "posthoc.py",
    ]
    source_paths = list(dict.fromkeys(path.resolve() for path in source_paths))
    package_versions = {}
    for package in ("numpy", "matplotlib"):
        try:
            package_versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            package_versions[package] = None

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_arguments": list(sys.argv[1:]),
        "git": git_metadata,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions,
        },
        "source_files": [
            {
                "path": str(path.relative_to(repo_root)),
                "sha256": _sha256(path),
            }
            for path in source_paths
        ],
    }


def evaluate_detector_matrix(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[List[Mapping[str, Any]], List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """Evaluate saved held-out predictions for every seed/control run."""

    metrics: List[Mapping[str, Any]] = []
    precision_recall: List[Mapping[str, Any]] = []
    risk_coverage: List[Mapping[str, Any]] = []
    for row in rows:
        result_directory = Path(str(row["summary_path"])).parent
        analysis = evaluate_result_directory(
            result_directory,
            seed=int(row["seed"]),
            intervention=str(row["intervention"]),
        )
        full_semantic = next(
            metric
            for metric in analysis.metrics
            if metric["target"] == "semantic_instability"
            and metric["detector"] == "full_metabears"
        )
        expected_auroc = row.get("semantic_instability_auroc")
        expected_ap = row.get("semantic_instability_average_precision")
        for observed, expected, name in (
            (full_semantic["auroc"], expected_auroc, "AUROC"),
            (full_semantic["average_precision"], expected_ap, "average precision"),
        ):
            if observed is None and expected is None:
                continue
            if observed is None or expected is None or not math.isclose(
                float(observed), float(expected), rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError(
                    "Post-hoc full MetaBEARS semantic "
                    f"{name} does not reproduce the frozen run summary."
                )
        metrics.extend(analysis.metrics)
        precision_recall.extend(analysis.precision_recall_curve)
        risk_coverage.extend(analysis.risk_coverage_curve)
    return metrics, precision_recall, risk_coverage


def evaluate_validation_fusion_matrix(
    rows: Sequence[Mapping[str, Any]],
    analysis_protocol: FrozenProtocol,
) -> tuple[
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
]:
    """Fit on primary validation only, then score every untouched ID test."""

    data = analysis_protocol.data
    fit_intervention = str(data["fit_intervention"])
    indexed = {
        (int(row["seed"]), str(row["intervention"])): row for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    metrics: List[Mapping[str, Any]] = []
    precision_recall: List[Mapping[str, Any]] = []
    risk_coverage: List[Mapping[str, Any]] = []
    model_records: List[Mapping[str, Any]] = []
    threshold_records: List[Mapping[str, Any]] = []
    for seed in seeds:
        fit_row = indexed.get((seed, fit_intervention))
        if fit_row is None:
            raise ValueError(
                f"Seed {seed} is missing fusion fit intervention {fit_intervention}."
            )
        fit_directory = Path(str(fit_row["summary_path"])).parent
        model = fit_validation_fusion_from_result_directory(
            fit_directory,
            signal_names=list(data["signals"]),
            target_name=str(data["fit_target"]),
            weight_grid_step=float(data["weight_grid_step"]),
            cross_validation_folds=int(data["cross_validation_folds"]),
            threshold_target_recall=float(data["threshold_target_recall"]),
            seed=seed,
        )
        model_records.append(
            {
                "seed": seed,
                "fit_intervention": fit_intervention,
                "analysis_protocol_id": analysis_protocol.protocol_id,
                "analysis_protocol_sha256": analysis_protocol.sha256,
                **model.to_record(),
            }
        )
        for row in rows:
            if int(row["seed"]) != seed:
                continue
            result = evaluate_fusion_result_directory(
                model,
                Path(str(row["summary_path"])).parent,
                seed=seed,
                intervention=str(row["intervention"]),
            )
            metrics.extend(result.analysis.metrics)
            precision_recall.extend(result.analysis.precision_recall_curve)
            risk_coverage.extend(result.analysis.risk_coverage_curve)
            threshold_records.extend(result.threshold_metrics)
    return (
        metrics,
        precision_recall,
        risk_coverage,
        model_records,
        threshold_records,
    )


def evaluate_intervention_calibrated_fusion_matrix(
    rows: Sequence[Mapping[str, Any]],
    analysis_protocol: FrozenProtocol,
) -> tuple[
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
]:
    """Fit v2 weights, then recalibrate each intervention without labels."""

    data = analysis_protocol.data
    fit_intervention = str(data["fit_intervention"])
    indexed = {
        (int(row["seed"]), str(row["intervention"])): row for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    metrics: List[Mapping[str, Any]] = []
    precision_recall: List[Mapping[str, Any]] = []
    risk_coverage: List[Mapping[str, Any]] = []
    model_records: List[Mapping[str, Any]] = []
    reference_records: List[Mapping[str, Any]] = []
    threshold_records: List[Mapping[str, Any]] = []
    for seed in seeds:
        fit_row = indexed.get((seed, fit_intervention))
        if fit_row is None:
            raise ValueError(
                f"Seed {seed} is missing fusion fit intervention {fit_intervention}."
            )
        fit_directory = Path(str(fit_row["summary_path"])).parent
        model = fit_validation_fusion_from_result_directory(
            fit_directory,
            signal_names=list(data["signals"]),
            target_name=str(data["fit_target"]),
            weight_grid_step=float(data["weight_grid_step"]),
            cross_validation_folds=int(data["cross_validation_folds"]),
            threshold_target_recall=float(data["threshold_target_recall"]),
            seed=seed,
        )
        model_records.append(
            {
                "seed": seed,
                "fit_intervention": fit_intervention,
                "normalization_scope": str(data["normalization_scope"]),
                "normalization_labels_used": bool(
                    data["normalization_labels_used"]
                ),
                "analysis_protocol_id": analysis_protocol.protocol_id,
                "analysis_protocol_sha256": analysis_protocol.sha256,
                **model.to_record(),
            }
        )
        for row in rows:
            if int(row["seed"]) != seed:
                continue
            intervention = str(row["intervention"])
            result_directory = Path(str(row["summary_path"])).parent
            references = calibrate_fusion_references_from_result_directory(
                result_directory,
                signal_names=model.signal_names,
            )
            reference_records.append(
                {
                    "seed": seed,
                    "intervention": intervention,
                    "split": str(data["normalization_split"]),
                    "labels_used": False,
                    "reference_samples": int(
                        next(iter(references.values())).size
                    ),
                    "signals": "|".join(model.signal_names),
                }
            )
            result = evaluate_fusion_result_directory(
                model,
                result_directory,
                seed=seed,
                intervention=intervention,
                detector_name="intervention_calibrated_fusion_v3",
                references=references,
            )
            metrics.extend(result.analysis.metrics)
            precision_recall.extend(result.analysis.precision_recall_curve)
            risk_coverage.extend(result.analysis.risk_coverage_curve)
            threshold_records.extend(result.threshold_metrics)
    return (
        metrics,
        precision_recall,
        risk_coverage,
        model_records,
        reference_records,
        threshold_records,
    )


def evaluate_leave_one_intervention_out_fusion_matrix(
    rows: Sequence[Mapping[str, Any]],
    analysis_protocol: FrozenProtocol,
) -> tuple[
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
]:
    """Fit on all other interventions, then evaluate the excluded control."""

    data = analysis_protocol.data
    interventions = tuple(str(name) for name in data["evaluation_interventions"])
    indexed = {
        (int(row["seed"]), str(row["intervention"])): row for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    metrics: List[Mapping[str, Any]] = []
    precision_recall: List[Mapping[str, Any]] = []
    risk_coverage: List[Mapping[str, Any]] = []
    model_records: List[Mapping[str, Any]] = []
    threshold_records: List[Mapping[str, Any]] = []
    for seed in seeds:
        missing = [
            name for name in interventions if (seed, name) not in indexed
        ]
        if missing:
            raise ValueError(
                f"Seed {seed} is missing v4 interventions: {', '.join(missing)}"
            )
        for held_out in interventions:
            training = tuple(name for name in interventions if name != held_out)
            training_directories = {
                name: Path(str(indexed[(seed, name)]["summary_path"])).parent
                for name in training
            }
            model = (
                fit_leave_one_intervention_out_fusion_from_result_directories(
                    training_directories,
                    signal_names=list(data["signals"]),
                    target_name=str(data["fit_target"]),
                    weight_grid_step=float(data["weight_grid_step"]),
                    threshold_target_recall=float(
                        data["threshold_target_recall"]
                    ),
                )
            )
            if model.cross_validation_folds != int(data["cross_validation_folds"]):
                raise ValueError(
                    "Observed v4 intervention folds do not match the protocol."
                )
            model_records.append(
                {
                    "seed": seed,
                    "held_out_intervention": held_out,
                    "training_interventions": "|".join(training),
                    "held_out_validation_used": False,
                    "analysis_protocol_id": analysis_protocol.protocol_id,
                    "analysis_protocol_sha256": analysis_protocol.sha256,
                    **model.to_record(),
                }
            )
            held_out_directory = Path(
                str(indexed[(seed, held_out)]["summary_path"])
            ).parent
            result = evaluate_fusion_result_directory(
                model,
                held_out_directory,
                seed=seed,
                intervention=held_out,
                detector_name="leave_one_intervention_out_fusion_v4",
            )
            metrics.extend(result.analysis.metrics)
            precision_recall.extend(result.analysis.precision_recall_curve)
            risk_coverage.extend(result.analysis.risk_coverage_curve)
            threshold_records.extend(result.threshold_metrics)
    return (
        metrics,
        precision_recall,
        risk_coverage,
        model_records,
        threshold_records,
    )


def evaluate_external_negative_control_fusion_matrix(
    rows: Sequence[Mapping[str, Any]],
    analysis_protocol: FrozenProtocol,
) -> tuple[
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
]:
    """Fit on patch interventions and evaluate the excluded negative control."""

    data = analysis_protocol.data
    training_interventions = tuple(
        str(name) for name in data["training_interventions"]
    )
    negative_control = str(data["negative_control_intervention"])
    indexed = {
        (int(row["seed"]), str(row["intervention"])): row for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    metrics: List[Mapping[str, Any]] = []
    precision_recall: List[Mapping[str, Any]] = []
    risk_coverage: List[Mapping[str, Any]] = []
    model_records: List[Mapping[str, Any]] = []
    threshold_records: List[Mapping[str, Any]] = []
    for seed in seeds:
        required = (*training_interventions, negative_control)
        missing = [name for name in required if (seed, name) not in indexed]
        if missing:
            raise ValueError(
                f"Seed {seed} is missing v5 interventions: {', '.join(missing)}"
            )
        training_directories = {
            name: Path(str(indexed[(seed, name)]["summary_path"])).parent
            for name in training_interventions
        }
        model = fit_leave_one_intervention_out_fusion_from_result_directories(
            training_directories,
            signal_names=list(data["signals"]),
            target_name=str(data["fit_target"]),
            weight_grid_step=float(data["weight_grid_step"]),
            threshold_target_recall=float(data["threshold_target_recall"]),
        )
        if model.cross_validation_folds != int(data["cross_validation_folds"]):
            raise ValueError(
                "Observed v5 intervention folds do not match the protocol."
            )
        model_records.append(
            {
                "seed": seed,
                "negative_control_intervention": negative_control,
                "training_interventions": "|".join(training_interventions),
                "negative_control_validation_used": False,
                "analysis_protocol_id": analysis_protocol.protocol_id,
                "analysis_protocol_sha256": analysis_protocol.sha256,
                **model.to_record(),
            }
        )
        control_directory = Path(
            str(indexed[(seed, negative_control)]["summary_path"])
        ).parent
        result = evaluate_fusion_result_directory(
            model,
            control_directory,
            seed=seed,
            intervention=negative_control,
            detector_name="external_negative_control_fusion_v5",
        )
        metrics.extend(result.analysis.metrics)
        precision_recall.extend(result.analysis.precision_recall_curve)
        risk_coverage.extend(result.analysis.risk_coverage_curve)
        threshold_records.extend(result.threshold_metrics)
    return (
        metrics,
        precision_recall,
        risk_coverage,
        model_records,
        threshold_records,
    )


def aggregate_fusion_threshold_results(
    rows: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    """Aggregate frozen-threshold operating points across independent seeds."""

    output: List[Mapping[str, Any]] = []
    groups = sorted(
        {
            (
                str(row["intervention"]),
                str(row["target"]),
                str(row["detector"]),
            )
            for row in rows
        }
    )
    for intervention, target, detector in groups:
        selected = [
            row
            for row in rows
            if row["intervention"] == intervention
            and row["target"] == target
            and row["detector"] == detector
        ]
        for metric in FUSION_THRESHOLD_METRICS:
            aggregate = _aggregate_values(
                [float(row[metric]) for row in selected],
                grouping_key="operating_point",
                grouping_value=f"{detector}:{intervention}:{target}",
                metric=metric,
            )
            aggregate.update(
                {
                    "intervention": intervention,
                    "target": target,
                    "detector": detector,
                }
            )
            output.append(aggregate)
    return output


def paired_detector_analysis(
    detector_rows: Sequence[Mapping[str, Any]],
    *,
    candidates: Sequence[str] = DETECTOR_COMPARISON_CANDIDATES,
    baselines: Sequence[str] = DETECTOR_COMPARISON_BASELINES,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute paired within-seed candidate-minus-baseline detector effects."""

    indexed = {
        (
            int(row["seed"]),
            str(row["intervention"]),
            str(row["target"]),
            str(row["detector"]),
        ): row
        for row in detector_rows
    }
    seeds = sorted({int(row["seed"]) for row in detector_rows})
    interventions = sorted({str(row["intervention"]) for row in detector_rows})
    targets = sorted({str(row["target"]) for row in detector_rows})
    paired_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for intervention in interventions:
            for target in targets:
                for candidate in candidates:
                    candidate_row = indexed.get(
                        (seed, intervention, target, candidate)
                    )
                    if candidate_row is None:
                        continue
                    for baseline in baselines:
                        if candidate == baseline:
                            continue
                        baseline_row = indexed.get(
                            (seed, intervention, target, baseline)
                        )
                        if baseline_row is None:
                            continue
                        for metric in DETECTOR_COMPARISON_METRICS:
                            candidate_value = candidate_row.get(metric)
                            baseline_value = baseline_row.get(metric)
                            if candidate_value is None or baseline_value is None:
                                continue
                            paired_rows.append(
                                {
                                    "seed": seed,
                                    "intervention": intervention,
                                    "target": target,
                                    "candidate": candidate,
                                    "baseline": baseline,
                                    "metric": metric,
                                    "higher_is_better": metric
                                    in {"auroc", "average_precision"},
                                    "candidate_value": float(candidate_value),
                                    "baseline_value": float(baseline_value),
                                    "paired_difference": float(candidate_value)
                                    - float(baseline_value),
                                }
                            )

    aggregate_rows_output: List[Dict[str, Any]] = []
    groups = sorted(
        {
            (
                str(row["intervention"]),
                str(row["target"]),
                str(row["candidate"]),
                str(row["baseline"]),
                str(row["metric"]),
            )
            for row in paired_rows
        }
    )
    for intervention, target, candidate, baseline, metric in groups:
        selected = [
            row
            for row in paired_rows
            if row["intervention"] == intervention
            and row["target"] == target
            and row["candidate"] == candidate
            and row["baseline"] == baseline
            and row["metric"] == metric
        ]
        aggregate = _aggregate_values(
            [float(row["paired_difference"]) for row in selected],
            grouping_key="comparison",
            grouping_value=f"{candidate}_minus_{baseline}",
            metric=metric,
        )
        aggregate.update(
            {
                "intervention": intervention,
                "target": target,
                "candidate": candidate,
                "baseline": baseline,
                "higher_is_better": selected[0]["higher_is_better"],
            }
        )
        aggregate_rows_output.append(aggregate)
    return paired_rows, aggregate_rows_output


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
        positions = list(range(len(interventions)))
        seeds = sorted({int(row["seed"]) for row in rows})
        for seed in seeds:
            seed_values = []
            seed_positions = []
            for position, name in zip(positions, interventions):
                match = next(
                    (
                        row
                        for row in rows
                        if int(row["seed"]) == seed
                        and row["intervention"] == name
                        and row.get(metric) is not None
                    ),
                    None,
                )
                if match is not None:
                    seed_positions.append(position)
                    seed_values.append(100.0 * float(match[metric]))
            if seed_values:
                axis.plot(
                    seed_positions,
                    seed_values,
                    color="0.75",
                    linewidth=1.0,
                    zorder=1,
                )
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
                zorder=2,
            )
            axis.scatter(
                [position],
                [100.0 * float(metric_rows[name]["mean"])],
                color="#1f77b4",
                edgecolor="black",
                marker="D",
                s=58,
                zorder=3,
            )
        axis.set_title(title)
        axis.set_xticks(positions, interventions, rotation=20)
        axis.set_ylim(bottom=0.0)
        if metric in {"semantic_instability_auroc", "review_rate"}:
            axis.set_ylim(0.0, 100.0)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Circles are independent ensemble seeds; diamonds are means",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(output_directory / "aggregate_metrics.png", dpi=180)
    plt.close(figure)
    return None


def _plot_paired_effects(
    output_directory: Path,
    paired_rows: Sequence[Mapping[str, Any]],
    paired_aggregates: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib is unavailable; paired-control plots were skipped."

    metrics = (
        ("id_accuracy_drop", "Accuracy-drop difference (pp)"),
        ("task_failure_f1", "Task-failure F1 difference (pp)"),
        ("semantic_instability_auroc", "Semantic AUROC difference (pp)"),
        ("review_rate", "Review-rate difference (pp)"),
    )
    comparisons = sorted(
        {str(row["comparator_intervention"]) for row in paired_rows}
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (metric, title) in zip(axes.flat, metrics):
        positions = list(range(len(comparisons)))
        for position, comparator in zip(positions, comparisons):
            values = [
                100.0 * float(row["paired_difference"])
                for row in paired_rows
                if row["metric"] == metric
                and row["comparator_intervention"] == comparator
            ]
            aggregate = next(
                (
                    row
                    for row in paired_aggregates
                    if row["metric"] == metric
                    and row["comparator_intervention"] == comparator
                ),
                None,
            )
            if not values or aggregate is None:
                continue
            axis.scatter([position] * len(values), values, color="black", s=28)
            axis.scatter(
                [position],
                [100.0 * float(aggregate["mean"])],
                color="#d62728",
                edgecolor="black",
                marker="D",
                s=62,
                zorder=3,
            )
        axis.axhline(0.0, color="0.5", linewidth=1.0, linestyle="--")
        axis.set_title(title)
        axis.set_xticks(positions, comparisons, rotation=15)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Primary minus comparator; circles are seeds and diamonds are means",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(output_directory / "paired_control_effects.png", dpi=180)
    plt.close(figure)
    return None


def _plot_detector_comparison(
    output_directory: Path,
    detector_rows: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib is unavailable; detector plots were skipped."

    detectors = (
        "task_entropy",
        "task_distribution_js",
        "ensemble_concept_disagreement",
        "concept_instability_without_perturbation",
        "perturbation_js",
        "full_metabears",
        "validation_fitted_fusion_v2",
        "intervention_calibrated_fusion_v3",
        "leave_one_intervention_out_fusion_v4",
    )
    labels = (
        "Task entropy",
        "Task-change JS",
        "Concept JS",
        "Base instability",
        "Perturbation JS",
        "MetaBEARS v1",
        "Validation fusion v2",
        "Calibrated fusion v3",
        "LOIO fusion v4",
    )
    targets = ("task_invariance_failure", "semantic_instability")
    interventions = sorted({str(row["intervention"]) for row in detector_rows})
    figure, axes = plt.subplots(
        len(targets), len(interventions), figsize=(15, 7.5), squeeze=False
    )
    for target_index, target in enumerate(targets):
        for intervention_index, intervention in enumerate(interventions):
            axis = axes[target_index, intervention_index]
            for position, detector in enumerate(detectors):
                values = [
                    100.0 * float(row["average_precision"])
                    for row in detector_rows
                    if row["target"] == target
                    and row["intervention"] == intervention
                    and row["detector"] == detector
                    and row.get("average_precision") is not None
                ]
                if not values:
                    continue
                axis.scatter([position] * len(values), values, color="black", s=22)
                axis.scatter(
                    [position],
                    [mean(values)],
                    color="#2ca02c",
                    edgecolor="black",
                    marker="D",
                    s=54,
                    zorder=3,
                )
            axis.set_title(f"{intervention}: {target.replace('_', ' ')}")
            axis.set_xticks(range(len(detectors)), labels, rotation=25, ha="right")
            axis.set_ylabel("Average precision (%)")
            axis.set_ylim(0.0, 100.0)
            axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Detector baselines on controlled held-out targets", fontsize=12)
    figure.tight_layout()
    figure.savefig(output_directory / "detector_average_precision.png", dpi=180)
    plt.close(figure)
    return None


def _plot_detector_curves(
    output_directory: Path,
    precision_recall_rows: Sequence[Mapping[str, Any]],
    risk_coverage_rows: Sequence[Mapping[str, Any]],
    *,
    primary_intervention: str,
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib is unavailable; detector curves were skipped."

    detectors = (
        "task_distribution_js",
        "concept_instability_without_perturbation",
        "perturbation_js",
        "full_metabears",
        "validation_fitted_fusion_v2",
        "intervention_calibrated_fusion_v3",
        "leave_one_intervention_out_fusion_v4",
    )
    colors = (
        "#7f7f7f",
        "#ff7f0e",
        "#2ca02c",
        "#1f77b4",
        "#9467bd",
        "#d62728",
        "#17becf",
    )
    target = "task_invariance_failure"
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for detector, color in zip(detectors, colors):
        seeds = sorted(
            {
                int(row["seed"])
                for row in precision_recall_rows
                if row["intervention"] == primary_intervention
                and row["target"] == target
                and row["detector"] == detector
            }
        )
        for seed_index, seed in enumerate(seeds):
            pr = [
                row
                for row in precision_recall_rows
                if row["intervention"] == primary_intervention
                and row["target"] == target
                and row["detector"] == detector
                and int(row["seed"]) == seed
            ]
            rc = [
                row
                for row in risk_coverage_rows
                if row["intervention"] == primary_intervention
                and row["target"] == target
                and row["detector"] == detector
                and int(row["seed"]) == seed
            ]
            label = detector.replace("_", " ") if seed_index == 0 else None
            axes[0].plot(
                [float(row["recall"]) for row in pr],
                [float(row["precision"]) for row in pr],
                color=color,
                alpha=0.45,
                linewidth=1.2,
                label=label,
            )
            axes[1].plot(
                [float(row["coverage"]) for row in rc],
                [float(row["selective_risk"]) for row in rc],
                color=color,
                alpha=0.45,
                linewidth=1.2,
                label=label,
            )
    axes[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_title("Precision-recall by seed")
    axes[1].set(
        xlabel="Automatic coverage",
        ylabel="Failure risk among accepted samples",
        xlim=(0, 1),
    )
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Risk-coverage by seed")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(f"{primary_intervention}: controlled task failures", fontsize=12)
    figure.tight_layout()
    figure.savefig(output_directory / "detector_curves.png", dpi=180)
    plt.close(figure)
    return None


def _plot_ood_metrics(
    output_directory: Path, model_rows: Sequence[Mapping[str, Any]]
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib is unavailable; OOD plots were skipped."

    metrics = (
        ("ood_auroc", "AUROC"),
        ("ood_average_precision", "Average precision"),
        ("ood_f1", "F1"),
    )
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for position, (metric, label) in enumerate(metrics):
        values = [
            100.0 * float(row[metric])
            for row in model_rows
            if row.get(metric) is not None
        ]
        axis.scatter([position] * len(values), values, color="black", s=28)
        axis.scatter(
            [position],
            [mean(values)],
            color="#9467bd",
            edgecolor="black",
            marker="D",
            s=62,
            zorder=3,
        )
    axis.set_xticks(range(len(metrics)), [label for _, label in metrics])
    axis.set_ylabel("Score (%)")
    axis.set_ylim(90.0, 100.0)
    axis.grid(axis="y", alpha=0.25)
    axis.set_title("OOD detection across unique ensembles")
    figure.tight_layout()
    figure.savefig(output_directory / "ood_metrics.png", dpi=180)
    plt.close(figure)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate a frozen MetaBEARS result matrix."
    )
    parser.add_argument("--protocol", default="experiment_protocol.json")
    parser.add_argument(
        "--analysis-protocol", default="analysis_protocol_v4.json"
    )
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--interventions", nargs="+", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--skip-detector-analysis",
        action="store_true",
        help="Aggregate summaries without loading held-out prediction artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    protocol = load_protocol(repo_root / args.protocol)
    analysis_protocol = load_analysis_protocol(
        repo_root / args.analysis_protocol, protocol
    )
    protocol_chain = analysis_protocol_chain(analysis_protocol, protocol)
    fusion_v2_protocol = protocol_chain[0]
    fusion_v3_protocol = next(
        (
            item
            for item in protocol_chain
            if item.data.get("normalization_scope") == "per_seed_and_intervention"
        ),
        None,
    )
    fusion_v4_protocol = next(
        (
            item
            for item in protocol_chain
            if item.data.get("outer_evaluation") == "leave_one_intervention_out"
        ),
        None,
    )
    fusion_v5_protocol = next(
        (
            item
            for item in protocol_chain
            if item.data.get("outer_evaluation") == "external_negative_control"
        ),
        None,
    )
    results_root = Path(args.results_root).expanduser().resolve()
    output_directory = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else results_root / "aggregate"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    evaluation = protocol.data["evaluation"]
    seeds = args.seeds or list(protocol.data["ensemble"]["base_seeds"])
    interventions = args.interventions or analysis_protocol.data.get(
        "evaluation_interventions",
        [
            evaluation["primary_intervention"],
            *evaluation.get("secondary_interventions", []),
        ],
    )
    rows = discover_rows(
        protocol,
        results_root,
        seeds=seeds,
        interventions=interventions,
        allow_partial=args.allow_partial,
    )
    aggregates = aggregate_rows(rows)
    model_rows, model_aggregates = aggregate_unique_models(rows)
    secondary_interventions = [
        intervention
        for intervention in interventions
        if intervention != evaluation["primary_intervention"]
    ]
    paired_rows, paired_aggregates = paired_control_analysis(
        rows,
        primary_intervention=evaluation["primary_intervention"],
        comparator_interventions=secondary_interventions,
    )
    if (
        fusion_v2_protocol.data["fit_intervention"]
        != evaluation["primary_intervention"]
    ):
        raise ValueError(
            "Analysis fusion must fit the frozen primary intervention."
        )
    detector_rows: List[Mapping[str, Any]] = []
    detector_aggregates: List[Mapping[str, Any]] = []
    precision_recall_rows: List[Mapping[str, Any]] = []
    risk_coverage_rows: List[Mapping[str, Any]] = []
    fusion_model_records: List[Mapping[str, Any]] = []
    fusion_threshold_records: List[Mapping[str, Any]] = []
    fusion_threshold_aggregates: List[Mapping[str, Any]] = []
    fusion_v3_model_records: List[Mapping[str, Any]] = []
    fusion_v3_reference_records: List[Mapping[str, Any]] = []
    fusion_v3_threshold_records: List[Mapping[str, Any]] = []
    fusion_v4_model_records: List[Mapping[str, Any]] = []
    fusion_v4_threshold_records: List[Mapping[str, Any]] = []
    fusion_v5_model_records: List[Mapping[str, Any]] = []
    fusion_v5_threshold_records: List[Mapping[str, Any]] = []
    detector_paired_rows: List[Dict[str, Any]] = []
    detector_paired_aggregates: List[Dict[str, Any]] = []
    if not args.skip_detector_analysis:
        (
            detector_rows,
            precision_recall_rows,
            risk_coverage_rows,
        ) = evaluate_detector_matrix(rows)
        (
            fusion_rows,
            fusion_precision_recall,
            fusion_risk_coverage,
            fusion_model_records,
            fusion_threshold_records,
        ) = evaluate_validation_fusion_matrix(rows, fusion_v2_protocol)
        detector_rows.extend(fusion_rows)
        precision_recall_rows.extend(fusion_precision_recall)
        risk_coverage_rows.extend(fusion_risk_coverage)
        if fusion_v3_protocol is not None:
            fusion_v3_input_rows = rows
            if fusion_v5_protocol is not None:
                negative_control = str(
                    fusion_v5_protocol.data["negative_control_intervention"]
                )
                fusion_v3_input_rows = [
                    row
                    for row in rows
                    if str(row["intervention"]) != negative_control
                ]
            (
                fusion_v3_rows,
                fusion_v3_precision_recall,
                fusion_v3_risk_coverage,
                fusion_v3_model_records,
                fusion_v3_reference_records,
                fusion_v3_threshold_records,
            ) = evaluate_intervention_calibrated_fusion_matrix(
                fusion_v3_input_rows, fusion_v3_protocol
            )
            detector_rows.extend(fusion_v3_rows)
            precision_recall_rows.extend(fusion_v3_precision_recall)
            risk_coverage_rows.extend(fusion_v3_risk_coverage)
        if fusion_v4_protocol is not None:
            (
                fusion_v4_rows,
                fusion_v4_precision_recall,
                fusion_v4_risk_coverage,
                fusion_v4_model_records,
                fusion_v4_threshold_records,
            ) = evaluate_leave_one_intervention_out_fusion_matrix(
                rows, fusion_v4_protocol
            )
            detector_rows.extend(fusion_v4_rows)
            precision_recall_rows.extend(fusion_v4_precision_recall)
            risk_coverage_rows.extend(fusion_v4_risk_coverage)
        if fusion_v5_protocol is not None:
            (
                fusion_v5_rows,
                fusion_v5_precision_recall,
                fusion_v5_risk_coverage,
                fusion_v5_model_records,
                fusion_v5_threshold_records,
            ) = evaluate_external_negative_control_fusion_matrix(
                rows, fusion_v5_protocol
            )
            detector_rows.extend(fusion_v5_rows)
            precision_recall_rows.extend(fusion_v5_precision_recall)
            risk_coverage_rows.extend(fusion_v5_risk_coverage)
        fusion_threshold_aggregates = aggregate_fusion_threshold_results(
            [
                *fusion_threshold_records,
                *fusion_v3_threshold_records,
                *fusion_v4_threshold_records,
                *fusion_v5_threshold_records,
            ]
        )
        detector_aggregates = aggregate_detector_results(detector_rows)
        (
            detector_paired_rows,
            detector_paired_aggregates,
        ) = paired_detector_analysis(detector_rows)

    provenance = reporting_provenance(
        repo_root,
        analysis_protocol_paths=[item.path for item in protocol_chain],
    )
    _write_csv(output_directory / "run_results.csv", rows)
    _write_csv(output_directory / "aggregate_results.csv", aggregates)
    _write_csv(output_directory / "model_results.csv", model_rows)
    _write_csv(output_directory / "model_aggregate_results.csv", model_aggregates)
    _write_csv(output_directory / "paired_control_results.csv", paired_rows)
    _write_csv(
        output_directory / "paired_control_aggregate_results.csv",
        paired_aggregates,
    )
    if detector_rows:
        _write_csv(output_directory / "detector_results.csv", detector_rows)
        _write_csv(
            output_directory / "detector_aggregate_results.csv",
            detector_aggregates,
        )
        _write_csv(
            output_directory / "detector_precision_recall_curves.csv",
            precision_recall_rows,
        )
        _write_csv(
            output_directory / "detector_risk_coverage_curves.csv",
            risk_coverage_rows,
        )
        _write_csv(
            output_directory / "fusion_v2_models.csv", fusion_model_records
        )
        _write_csv(
            output_directory / "fusion_v2_threshold_results.csv",
            fusion_threshold_records,
        )
        _write_csv(
            output_directory / "fusion_v3_models.csv",
            fusion_v3_model_records,
        )
        _write_csv(
            output_directory / "fusion_v3_reference_calibrations.csv",
            fusion_v3_reference_records,
        )
        _write_csv(
            output_directory / "fusion_v3_threshold_results.csv",
            fusion_v3_threshold_records,
        )
        _write_csv(
            output_directory / "fusion_v4_models.csv",
            fusion_v4_model_records,
        )
        _write_csv(
            output_directory / "fusion_v4_threshold_results.csv",
            fusion_v4_threshold_records,
        )
        _write_csv(
            output_directory / "fusion_v5_models.csv",
            fusion_v5_model_records,
        )
        _write_csv(
            output_directory / "fusion_v5_threshold_results.csv",
            fusion_v5_threshold_records,
        )
        _write_csv(
            output_directory / "fusion_threshold_aggregate_results.csv",
            fusion_threshold_aggregates,
        )
        _write_csv(
            output_directory / "detector_paired_results.csv",
            detector_paired_rows,
        )
        _write_csv(
            output_directory / "detector_paired_aggregate_results.csv",
            detector_paired_aggregates,
        )
    (output_directory / "reporting_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    plot_warnings = [
        _plot_aggregates(output_directory, aggregates, rows),
        _plot_paired_effects(output_directory, paired_rows, paired_aggregates),
        _plot_ood_metrics(output_directory, model_rows),
    ]
    if detector_rows:
        plot_warnings.extend(
            [
                _plot_detector_comparison(output_directory, detector_rows),
                _plot_detector_curves(
                    output_directory,
                    precision_recall_rows,
                    risk_coverage_rows,
                    primary_intervention=evaluation["primary_intervention"],
                ),
            ]
        )
    warnings = [warning for warning in plot_warnings if warning is not None]
    payload = {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "analysis_protocol_id": analysis_protocol.protocol_id,
        "analysis_protocol_sha256": analysis_protocol.sha256,
        "analysis_protocol": analysis_protocol.data,
        "reporting_provenance": provenance,
        "runs": rows,
        "aggregates": aggregates,
        "unique_models": model_rows,
        "model_aggregates": model_aggregates,
        "paired_control_results": paired_rows,
        "paired_control_aggregates": paired_aggregates,
        "detector_analysis": {
            "performed": not args.skip_detector_analysis,
            "detector_definitions": DETECTOR_DESCRIPTIONS,
            "target_definitions": TARGET_DEFINITIONS,
            "results": detector_rows,
            "aggregates": detector_aggregates,
            "paired_results": detector_paired_rows,
            "paired_aggregates": detector_paired_aggregates,
            "fusion_models": fusion_model_records,
            "fusion_threshold_results": fusion_threshold_records,
            "fusion_threshold_aggregates": fusion_threshold_aggregates,
            "fusion_v3_models": fusion_v3_model_records,
            "fusion_v3_reference_calibrations": fusion_v3_reference_records,
            "fusion_v3_threshold_results": fusion_v3_threshold_records,
            "fusion_v4_models": fusion_v4_model_records,
            "fusion_v4_threshold_results": fusion_v4_threshold_records,
            "fusion_v5_models": fusion_v5_model_records,
            "fusion_v5_threshold_results": fusion_v5_threshold_records,
            "curve_files": (
                {
                    "precision_recall": "detector_precision_recall_curves.csv",
                    "risk_coverage": "detector_risk_coverage_curves.csv",
                }
                if detector_rows
                else None
            ),
        },
        "warnings": warnings,
        "warning": "; ".join(warnings) if warnings else None,
    }
    (output_directory / "aggregate_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Aggregated {len(rows)} runs into: {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
