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
    """Separated confidence signals for every evaluated sample.

    ``review_flag`` is the union of two independently triggered paths,
    retained separately so a consumer can tell why a sample was flagged:
    ``shortcut_flag`` fires on high shortcut risk, ``ood_flag`` fires on low
    neural familiarity. Keeping them apart makes the evidence behind a review
    explicit, but does not guarantee that the flags identify distinct causal
    failure types on real data.
    """

    predicted_label: np.ndarray
    task_confidence: np.ndarray
    neural_familiarity: np.ndarray
    shortcut_risk: np.ndarray
    concept_consistency: np.ndarray
    ensemble_concept_disagreement: np.ndarray
    concept_vote_disagreement: np.ndarray
    perturbation_js: np.ndarray
    label_disagreement: np.ndarray
    shortcut_flag: np.ndarray
    ood_flag: np.ndarray
    review_flag: np.ndarray
    review_threshold: float
    familiarity_threshold: float

    def __post_init__(self) -> None:
        sample_count = self.predicted_label.shape[0]
        fields = (
            self.task_confidence,
            self.neural_familiarity,
            self.shortcut_risk,
            self.concept_consistency,
            self.ensemble_concept_disagreement,
            self.concept_vote_disagreement,
            self.perturbation_js,
            self.label_disagreement,
            self.shortcut_flag,
            self.ood_flag,
            self.review_flag,
        )
        if self.predicted_label.ndim != 1 or any(
            field.ndim != 1 or field.shape[0] != sample_count for field in fields
        ):
            raise ValueError("Every report field must be a vector of equal length.")
        for flag_field, name in (
            (self.shortcut_flag, "shortcut_flag"),
            (self.ood_flag, "ood_flag"),
            (self.review_flag, "review_flag"),
        ):
            if flag_field.dtype != np.bool_:
                raise ValueError(f"{name} must be a boolean array.")
        if not 0.0 <= self.review_threshold <= 1.0:
            raise ValueError("review_threshold must lie within [0, 1].")
        if not 0.0 <= self.familiarity_threshold <= 1.0:
            raise ValueError("familiarity_threshold must lie within [0, 1].")
        if not np.array_equal(self.review_flag, self.shortcut_flag | self.ood_flag):
            raise ValueError("review_flag must equal shortcut_flag | ood_flag.")

    def to_records(self) -> List[Dict[str, object]]:
        """Return JSON/CSV-compatible per-sample dictionaries."""

        records: List[Dict[str, object]] = []
        for index in range(self.predicted_label.shape[0]):
            records.append(
                {
                    "sample_id": index,
                    "predicted_label": int(self.predicted_label[index]),
                    "task_confidence": float(self.task_confidence[index]),
                    "neural_familiarity": float(self.neural_familiarity[index]),
                    "concept_consistency": float(self.concept_consistency[index]),
                    "ensemble_concept_disagreement": float(
                        self.ensemble_concept_disagreement[index]
                    ),
                    "concept_vote_disagreement": float(
                        self.concept_vote_disagreement[index]
                    ),
                    "perturbation_js": float(self.perturbation_js[index]),
                    "label_disagreement": float(self.label_disagreement[index]),
                    "shortcut_risk": float(self.shortcut_risk[index]),
                    "shortcut_flag": bool(self.shortcut_flag[index]),
                    "ood_flag": bool(self.ood_flag[index]),
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
            "familiarity_threshold": float(self.familiarity_threshold),
            "shortcut_flag_count": int(np.count_nonzero(self.shortcut_flag)),
            "ood_flag_count": int(np.count_nonzero(self.ood_flag)),
            "mean_task_confidence": float(self.task_confidence.mean()),
            "mean_neural_familiarity": float(self.neural_familiarity.mean()),
            "mean_concept_consistency": float(self.concept_consistency.mean()),
            "mean_ensemble_concept_disagreement": float(
                self.ensemble_concept_disagreement.mean()
            ),
            "mean_concept_vote_disagreement": float(
                self.concept_vote_disagreement.mean()
            ),
            "mean_perturbation_js": float(self.perturbation_js.mean()),
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
    familiarity_threshold: float = 0.0,
) -> MetaCognitiveReport:
    """Build the MetaBEARS diagnostic report.

    ``shortcut_risk`` is deliberately a pure function of concept instability,
    task-label stability, and task confidence. It intentionally targets the
    characteristic mismatch of a reasoning shortcut: unstable concept
    semantics alongside a stable, confident task prediction. Neural
    familiarity is excluded from that score on purpose: folding it in would
    directly conflate representation-shift evidence with semantic-instability
    evidence.

    Familiarity instead drives its own decision path. ``shortcut_flag`` fires
    when ``shortcut_risk >= review_threshold``; ``ood_flag`` fires when
    ``neural_familiarity <= familiarity_threshold``; ``review_flag`` is their
    union, so an unfamiliar-but-concept-stable input is now surfaced for
    review even though its ``shortcut_risk`` correctly stays low. The default
    ``familiarity_threshold=0.0`` flags only inputs at or beyond the
    reference distribution's observed support.

    ``task_confidence`` is the ensemble-mean probability of the predicted
    task label. No symbolic rule-support or entailment signal is implemented
    here, so the field is named for what it actually measures rather than
    for the symbolic-confidence signal this diagnostic layer will eventually
    need.

    Both thresholds should be selected on validation data (see
    ``select_review_threshold`` in ``thresholds.py``), never on the test set.
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
    if not 0.0 <= familiarity_threshold <= 1.0:
        raise ValueError("familiarity_threshold must lie within [0, 1].")

    consistency = probe_concept_consistency(
        concepts,
        perturbed_member_probabilities=perturbed_member_probabilities,
    )
    mean_label_probability = labels.mean(axis=0)
    predicted_label = np.argmax(mean_label_probability, axis=-1)
    task_confidence = np.max(mean_label_probability, axis=-1)
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
        consistency.instability * task_stability * task_confidence,
        0.0,
        1.0,
    )
    shortcut_flag = shortcut_risk >= review_threshold
    ood_flag = neural_familiarity <= familiarity_threshold
    review_flag = shortcut_flag | ood_flag

    return MetaCognitiveReport(
        predicted_label=predicted_label.astype(np.int64),
        task_confidence=task_confidence,
        neural_familiarity=neural_familiarity,
        shortcut_risk=shortcut_risk,
        concept_consistency=consistency.score,
        ensemble_concept_disagreement=consistency.ensemble_js,
        concept_vote_disagreement=consistency.vote_disagreement,
        perturbation_js=consistency.perturbation_js,
        label_disagreement=label_disagreement,
        shortcut_flag=shortcut_flag,
        ood_flag=ood_flag,
        review_flag=review_flag,
        review_threshold=float(review_threshold),
        familiarity_threshold=float(familiarity_threshold),
    )
