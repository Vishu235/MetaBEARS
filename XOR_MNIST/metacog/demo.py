"""Deterministic synthetic demonstration of the MetaBEARS diagnostic layer."""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from .report import MetaCognitiveReport, build_meta_cognitive_report


def _peaked_distribution(class_index: int, class_count: int, peak: float) -> np.ndarray:
    distribution = np.full(class_count, (1.0 - peak) / (class_count - 1))
    distribution[class_index] = peak
    return distribution


def create_demo_predictions(
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create stable, shortcut-like, and unfamiliar synthetic examples."""

    rng = np.random.default_rng(seed)
    member_count, sample_count, concept_count = 5, 10, 2
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

    # Samples 2 and 7 retain stable task predictions while ensemble members
    # disagree sharply on concept semantics: the shortcut-like signature.
    for sample in (2, 7):
        for member in range(member_count):
            for concept in range(concept_count):
                concept_class = (sample + concept + member % 2) % concept_classes
                concepts[member, sample, concept] = _peaked_distribution(
                    concept_class, concept_classes, peak=0.98
                )

    reference_distances = np.clip(rng.normal(0.40, 0.08, size=200), 0.0, None)
    sample_distances = np.clip(
        rng.normal(0.40, 0.08, size=sample_count), 0.0, None
    )
    # Sample 5 is deliberately unfamiliar but conceptually consistent.
    sample_distances[5] = reference_distances.max() + 0.5

    return concepts, labels, sample_distances, reference_distances


def run_demo(output_dir: Path, seed: int = 42) -> MetaCognitiveReport:
    concepts, labels, distances, reference = create_demo_predictions(seed)
    report = build_meta_cognitive_report(
        concepts,
        labels,
        distances,
        reference,
        review_threshold=0.45,
    )

    json_path = report.write_json(output_dir / "metabears_report.json")
    csv_path = report.write_csv(output_dir / "metabears_report.csv")

    print(
        "sample label symbolic familiarity consistency shortcut-risk review",
        flush=True,
    )
    for record in report.to_records():
        print(
            f"{record['sample_id']:>6} "
            f"{record['predicted_label']:>5} "
            f"{record['symbolic_confidence']:.3f} "
            f"{record['neural_familiarity']:.3f} "
            f"{record['concept_consistency']:.3f} "
            f"{record['shortcut_risk']:.3f} "
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
