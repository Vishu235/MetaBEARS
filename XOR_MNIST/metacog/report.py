"""Meta-cognitive confidence report construction and serialization."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .consistency import (
    _as_probability_array,
    predictive_disagreement,
    probe_concept_consistency,
)
from .familiarity import familiarity_from_reference


PathLike = Union[str, Path]


@dataclass(frozen=True)
class MetaCognitiveReport:
    """Separated confidence signals for every evaluated sample."""

    predicted_label: np.ndarray
    symbolic_confidence: np.ndarray
    neural_familiarity: np.ndarray
    shortcut_risk: np.ndarray
    concept_consistency: np.ndarray
    label_disagreement: np.ndarray
    review_flag: np.ndarray
    review_threshold: float

    def __post_init__(self) -> None:
        sample_count = self.predicted_label.shape[0]
        fields = (
            self.symbolic_confidence,
            self.neural_familiarity,
            self.shortcut_risk,
            self.concept_consistency,
            self.label_disagreement,
            self.review_flag,
        )
        if self.predicted_label.ndim != 1 or any(
            field.ndim != 1 or field.shape[0] != sample_count for field in fields
        ):
            raise ValueError("Every report field must be a vector of equal length.")

    def to_records(self) -> List[Dict[str, object]]:
        """Return JSON/CSV-compatible per-sample dictionaries."""

        records: List[Dict[str, object]] = []
        for index in range(self.predicted_label.shape[0]):
            records.append(
                {
                    "sample_id": index,
                    "predicted_label": int(self.predicted_label[index]),
                    "symbolic_confidence": float(self.symbolic_confidence[index]),
                    "neural_familiarity": float(self.neural_familiarity[index]),
                    "concept_consistency": float(self.concept_consistency[index]),
                    "label_disagreement": float(self.label_disagreement[index]),
                    "shortcut_risk": float(self.shortcut_risk[index]),
                    "review_flag": bool(self.review_flag[index]),
                }
            )
        return records

    def summary(self) -> Dict[str, object]:
        """Return aggregate information without hiding the separated signals."""

        sample_count = int(self.predicted_label.shape[0])
        review_count = int(np.count_nonzero(self.review_flag))
        return {
            "samples": sample_count,
            "review_count": review_count,
            "review_rate": review_count / sample_count if sample_count else 0.0,
            "review_threshold": float(self.review_threshold),
            "mean_symbolic_confidence": float(self.symbolic_confidence.mean()),
            "mean_neural_familiarity": float(self.neural_familiarity.mean()),
            "mean_concept_consistency": float(self.concept_consistency.mean()),
            "mean_shortcut_risk": float(self.shortcut_risk.mean()),
        }

    def write_json(self, path: PathLike) -> Path:
        """Write a report with aggregate and per-sample sections."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": self.summary(), "samples": self.to_records()}
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return destination

    def write_csv(self, path: PathLike) -> Path:
        """Write the per-sample report as CSV."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        records = self.to_records()
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        return destination


def build_meta_cognitive_report(
    concept_member_probabilities: np.ndarray,
    label_member_probabilities: np.ndarray,
    representation_distances: np.ndarray,
    reference_distances: np.ndarray,
    *,
    perturbed_member_probabilities: Optional[np.ndarray] = None,
    review_threshold: float = 0.5,
) -> MetaCognitiveReport:
    """Build the first MetaBEARS three-signal diagnostic report.

    The shortcut score intentionally targets the characteristic mismatch of a
    reasoning shortcut: unstable concept semantics alongside a stable,
    confident task prediction. Neural familiarity is reported independently
    so an unfamiliar input is not automatically mislabeled as a shortcut.
    Thresholds must be selected on validation data before benchmark reporting.
    """

    concepts = _as_probability_array(
        concept_member_probabilities,
        name="concept_member_probabilities",
        dimensions=4,
    )
    labels = _as_probability_array(
        label_member_probabilities,
        name="label_member_probabilities",
        dimensions=3,
    )
    if concepts.shape[0] < 2 or labels.shape[0] < 2:
        raise ValueError("At least two prediction members are required.")
    if labels.shape[:2] != concepts.shape[:2]:
        raise ValueError(
            "Label and concept predictions must share member and sample axes."
        )
    if not 0.0 <= review_threshold <= 1.0:
        raise ValueError("review_threshold must lie within [0, 1].")

    consistency = probe_concept_consistency(
        concepts,
        perturbed_member_probabilities=perturbed_member_probabilities,
    )
    mean_label_probability = labels.mean(axis=0)
    predicted_label = np.argmax(mean_label_probability, axis=-1)
    symbolic_confidence = np.max(mean_label_probability, axis=-1)
    label_disagreement = predictive_disagreement(labels)
    neural_familiarity = familiarity_from_reference(
        representation_distances,
        reference_distances,
    )
    if neural_familiarity.shape[0] != concepts.shape[1]:
        raise ValueError(
            "representation_distances must contain one value per sample."
        )

    task_stability = 1.0 - label_disagreement
    shortcut_risk = np.clip(
        consistency.instability * task_stability * symbolic_confidence,
        0.0,
        1.0,
    )
    review_flag = shortcut_risk >= review_threshold

    return MetaCognitiveReport(
        predicted_label=predicted_label.astype(np.int64),
        symbolic_confidence=symbolic_confidence,
        neural_familiarity=neural_familiarity,
        shortcut_risk=shortcut_risk,
        concept_consistency=consistency.score,
        label_disagreement=label_disagreement,
        review_flag=review_flag,
        review_threshold=float(review_threshold),
    )
