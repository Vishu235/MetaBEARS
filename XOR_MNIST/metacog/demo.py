"""Deterministic synthetic demonstration of the MetaBEARS diagnostic layer."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .report import MetaCognitiveReport, build_meta_cognitive_report
from .thresholds import select_review_threshold


def _peaked_distribution(class_index: int, class_count: int, peak: float) -> np.ndarray:
    distribution = np.full(class_count, (1.0 - peak) / (class_count - 1))
    distribution[class_index] = peak
    return distribution


@dataclass(frozen=True)
class DemoSplit:
    """A synthetic split with injected mechanism-check labels."""

    concepts: np.ndarray
    labels: np.ndarray
    representation_distances: np.ndarray
    reference_distances: np.ndarray
    shortcut_like_cases: np.ndarray
    unfamiliar: np.ndarray


def create_demo_split(
    seed: int,
    *,
    sample_count: int = 10,
    shortcut_indices: Sequence[int] = (2, 7),
    unfamiliar_indices: Sequence[int] = (5,),
    reference_distances: Optional[np.ndarray] = None,
) -> DemoSplit:
    """Create a synthetic split with stable, shortcut-like, and unfamiliar samples.

    ``shortcut_indices`` retain stable task predictions while ensemble
    members disagree sharply on concept semantics: the shortcut-like
    signature. ``unfamiliar_indices`` are deliberately placed beyond the
    reference distribution's support while remaining conceptually
    consistent. Both index sets double as positive labels for this controlled
    mechanism check, so they must be disjoint. They are not independent
    ground-truth evidence of real reasoning shortcuts.
    """

    if set(shortcut_indices) & set(unfamiliar_indices):
        raise ValueError("shortcut_indices and unfamiliar_indices must be disjoint.")
    if any(not 0 <= index < sample_count for index in (*shortcut_indices, *unfamiliar_indices)):
        raise ValueError("shortcut_indices and unfamiliar_indices must fall within sample_count.")

    rng = np.random.default_rng(seed)
    member_count, concept_count = 5, 2
    concept_classes, label_classes = 3, 5
    concepts = np.empty(
        (member_count, sample_count, concept_count, concept_classes),
        dtype=np.float64,
    )
    labels = np.empty(
        (member_count, sample_count, label_classes), dtype=np.float64
    )

    for member in range(member_count):
        for sample in range(sample_count):
            for concept in range(concept_count):
                concept_class = (sample + concept) % concept_classes
                concepts[member, sample, concept] = _peaked_distribution(
                    concept_class, concept_classes, peak=0.96
                )
            label_class = sample % label_classes
            labels[member, sample] = _peaked_distribution(
                label_class, label_classes, peak=0.97
            )

    for sample in shortcut_indices:
        for member in range(member_count):
            for concept in range(concept_count):
                concept_class = (sample + concept + member % 2) % concept_classes
                concepts[member, sample, concept] = _peaked_distribution(
                    concept_class, concept_classes, peak=0.98
                )

    effective_reference = (
        np.clip(rng.normal(0.40, 0.08, size=200), 0.0, None)
        if reference_distances is None
        else np.asarray(reference_distances, dtype=np.float64)
    )
    sample_distances = np.clip(
        rng.normal(0.40, 0.08, size=sample_count), 0.0, None
    )
    for sample in unfamiliar_indices:
        sample_distances[sample] = effective_reference.max() + 0.5

    shortcut_like_cases = np.zeros(sample_count, dtype=bool)
    shortcut_like_cases[list(shortcut_indices)] = True
    unfamiliar = np.zeros(sample_count, dtype=bool)
    unfamiliar[list(unfamiliar_indices)] = True

    return DemoSplit(
        concepts=concepts,
        labels=labels,
        representation_distances=sample_distances,
        reference_distances=effective_reference,
        shortcut_like_cases=shortcut_like_cases,
        unfamiliar=unfamiliar,
    )


def create_demo_predictions(
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create stable, shortcut-like, and unfamiliar synthetic examples."""

    split = create_demo_split(seed)
    return (
        split.concepts,
        split.labels,
        split.representation_distances,
        split.reference_distances,
    )


def run_demo(output_dir: Path, seed: int = 42) -> MetaCognitiveReport:
    # A calibration split, disjoint from the reported split below, supplies
    # injected positive labels for this mechanism check. Its own
    # review_flag/shortcut_flag/ood_flag are discarded: only shortcut_risk and
    # neural_familiarity are used for calibration. These synthetic labels are
    # not independent evidence of real shortcut-detection performance.
    validation = create_demo_split(
        seed + 1,
        sample_count=40,
        shortcut_indices=tuple(range(2, 40, 5)),
        unfamiliar_indices=tuple(range(4, 40, 10)),
    )
    validation_report = build_meta_cognitive_report(
        validation.concepts,
        validation.labels,
        validation.representation_distances,
        validation.reference_distances,
    )
    shortcut_selection = select_review_threshold(
        validation_report.shortcut_risk,
        validation.shortcut_like_cases,
    )
    familiarity_selection = select_review_threshold(
        validation_report.neural_familiarity,
        validation.unfamiliar,
        higher_is_riskier=False,
    )

    # The reported split reuses the validation split's reference distances,
    # as familiarity_from_reference's contract requires: both distances must
    # come from the same reference distribution.
    reported = create_demo_split(seed, reference_distances=validation.reference_distances)
    report = build_meta_cognitive_report(
        reported.concepts,
        reported.labels,
        reported.representation_distances,
        reported.reference_distances,
        review_threshold=shortcut_selection.threshold,
        familiarity_threshold=familiarity_selection.threshold,
    )

    json_path = report.write_json(output_dir / "metabears_report.json")
    csv_path = report.write_csv(output_dir / "metabears_report.csv")

    print(
        f"Calibration split: {validation.shortcut_like_cases.sum()} injected "
        f"shortcut-like disagreement cases, {validation.unfamiliar.sum()} injected "
        f"unfamiliar cases out of {validation.shortcut_like_cases.shape[0]}, "
        "disjoint from the reported split.",
        flush=True,
    )
    print(
        "shortcut_risk threshold: "
        f"{shortcut_selection.threshold:.3f} (policy={shortcut_selection.policy}, "
        f"precision={shortcut_selection.precision:.3f}, "
        f"recall={shortcut_selection.recall:.3f}, f1={shortcut_selection.f1:.3f})",
        flush=True,
    )
    print(
        "familiarity threshold: "
        f"{familiarity_selection.threshold:.3f} (policy={familiarity_selection.policy}, "
        f"precision={familiarity_selection.precision:.3f}, "
        f"recall={familiarity_selection.recall:.3f}, f1={familiarity_selection.f1:.3f})\n",
        flush=True,
    )

    print(
        "sample label task-conf familiarity consistency shortcut-risk shortcut   ood review",
        flush=True,
    )
    for record in report.to_records():
        print(
            f"{record['sample_id']:>6} "
            f"{record['predicted_label']:>5} "
            f"{record['task_confidence']:>8.3f} "
            f"{record['neural_familiarity']:>11.3f} "
            f"{record['concept_consistency']:>11.3f} "
            f"{record['shortcut_risk']:>13.3f} "
            f"{str(record['shortcut_flag']):>8} "
            f"{str(record['ood_flag']):>5} "
            f"{str(record['review_flag']):>6}",
            flush=True,
        )

    print(f"\nSummary: {report.summary()}", flush=True)
    print(f"JSON report: {json_path}", flush=True)
    print(f"CSV report: {csv_path}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic MetaBEARS diagnostic demonstration."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("colab_outputs") / "metabears_demo",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_demo(args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
