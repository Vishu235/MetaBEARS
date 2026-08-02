"""Concept-level consistency probes for stochastic or ensemble predictions."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


_EPSILON = 1e-12


def _as_probability_array(
    values: np.ndarray, *, name: str, dimensions: int
) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != dimensions:
        raise ValueError(
            f"{name} must have {dimensions} dimensions; got shape "
            f"{probabilities.shape}."
        )
    if probabilities.shape[-1] < 2:
        raise ValueError(f"{name} must contain at least two classes.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(f"{name} values must be within [0, 1].")

    totals = probabilities.sum(axis=-1)
    if not np.allclose(totals, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError(f"{name} must sum to one along its class axis.")
    return probabilities


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    class_count = probabilities.shape[-1]
    safe = np.clip(probabilities, _EPSILON, 1.0)
    entropy = -np.sum(probabilities * np.log(safe), axis=-1)
    return entropy / np.log(class_count)


def predictive_disagreement(member_probabilities: np.ndarray) -> np.ndarray:
    """Return normalized Jensen-Shannon disagreement for each predicted item.

    Args:
        member_probabilities: Array shaped ``[members, ..., classes]``.

    Returns:
        An array with the member and class axes removed. Values lie in
        ``[0, 1]`` and estimate epistemic disagreement as predictive entropy
        minus expected member entropy.
    """

    raw_probabilities = np.asarray(member_probabilities, dtype=np.float64)
    if raw_probabilities.ndim < 3:
        raise ValueError(
            "member_probabilities must have a member axis, at least one item "
            "axis, and a class axis."
        )
    probabilities = _as_probability_array(
        raw_probabilities,
        name="member_probabilities",
        dimensions=raw_probabilities.ndim,
    )
    if probabilities.shape[0] < 2:
        raise ValueError("At least two prediction members are required.")

    mean_probability = probabilities.mean(axis=0)
    disagreement = _normalized_entropy(mean_probability) - _normalized_entropy(
        probabilities
    ).mean(axis=0)
    return np.clip(disagreement, 0.0, 1.0)


def _vote_disagreement(member_probabilities: np.ndarray) -> np.ndarray:
    votes = np.argmax(member_probabilities, axis=-1)
    member_count = votes.shape[0]
    class_count = member_probabilities.shape[-1]
    flattened = votes.reshape(member_count, -1)
    agreement = np.empty(flattened.shape[1], dtype=np.float64)

    for index, item_votes in enumerate(flattened.T):
        counts = np.bincount(item_votes, minlength=class_count)
        agreement[index] = counts.max() / member_count

    return (1.0 - agreement).reshape(votes.shape[1:])


def _pairwise_js(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (first + second)
    safe_first = np.clip(first, _EPSILON, 1.0)
    safe_second = np.clip(second, _EPSILON, 1.0)
    safe_midpoint = np.clip(midpoint, _EPSILON, 1.0)
    first_kl = np.sum(first * (np.log(safe_first) - np.log(safe_midpoint)), axis=-1)
    second_kl = np.sum(
        second * (np.log(safe_second) - np.log(safe_midpoint)), axis=-1
    )
    return np.clip(0.5 * (first_kl + second_kl) / np.log(2.0), 0.0, 1.0)


@dataclass(frozen=True)
class ConsistencyResult:
    """Per-sample decomposition returned by the consistency probe."""

    score: np.ndarray
    ensemble_js: np.ndarray
    vote_disagreement: np.ndarray
    perturbation_js: np.ndarray

    @property
    def instability(self) -> np.ndarray:
        """Return one minus the consistency score."""

        return 1.0 - self.score


def probe_concept_consistency(
    concept_member_probabilities: np.ndarray,
    *,
    perturbed_member_probabilities: Optional[np.ndarray] = None,
) -> ConsistencyResult:
    """Measure whether concept semantics remain stable for each sample.

    Args:
        concept_member_probabilities: Probabilities shaped
            ``[members, samples, concepts, classes]``.
        perturbed_member_probabilities: Optional label-preserving perturbation
            predictions shaped
            ``[members, samples, perturbations, concepts, classes]``.

    The base probe combines distributional Jensen-Shannon disagreement and
    discrete vote disagreement. If perturbations are supplied, semantic
    stability between the original and perturbed ensemble means contributes
    to the score as a third, separately reported component.

    Base instability is the fixed 0.6/0.4 blend of ensemble Jensen--Shannon
    disagreement and vote disagreement. Perturbation instability is combined
    with that base using a probabilistic OR. Consequently, a perfectly stable
    perturbation leaves the base score unchanged, while either source of
    instability can increase the final score. These heuristics have not been
    calibrated against validation data.
    """

    base = _as_probability_array(
        concept_member_probabilities,
        name="concept_member_probabilities",
        dimensions=4,
    )
    if base.shape[0] < 2:
        raise ValueError("At least two prediction members are required.")

    ensemble_js_per_concept = predictive_disagreement(base)
    vote_per_concept = _vote_disagreement(base)
    ensemble_js = ensemble_js_per_concept.mean(axis=-1)
    vote_disagreement = vote_per_concept.mean(axis=-1)
    base_instability = 0.6 * ensemble_js + 0.4 * vote_disagreement

    if perturbed_member_probabilities is None:
        perturbation_js = np.zeros(base.shape[1], dtype=np.float64)
        instability = base_instability
    else:
        perturbed = _as_probability_array(
            perturbed_member_probabilities,
            name="perturbed_member_probabilities",
            dimensions=5,
        )
        expected_shape = (base.shape[0], base.shape[1], base.shape[2], base.shape[3])
        observed_shape = (
            perturbed.shape[0],
            perturbed.shape[1],
            perturbed.shape[3],
            perturbed.shape[4],
        )
        if observed_shape != expected_shape:
            raise ValueError(
                "perturbed_member_probabilities must match the base member, "
                "sample, concept, and class dimensions."
            )
        if perturbed.shape[2] < 1:
            raise ValueError("At least one perturbation is required.")

        base_mean = base.mean(axis=0)[:, np.newaxis, :, :]
        perturbation_mean = perturbed.mean(axis=0)
        perturbation_js = _pairwise_js(base_mean, perturbation_mean).mean(
            axis=(1, 2)
        )
        instability = 1.0 - (1.0 - base_instability) * (1.0 - perturbation_js)

    return ConsistencyResult(
        score=np.clip(1.0 - instability, 0.0, 1.0),
        ensemble_js=np.clip(ensemble_js, 0.0, 1.0),
        vote_disagreement=np.clip(vote_disagreement, 0.0, 1.0),
        perturbation_js=np.clip(perturbation_js, 0.0, 1.0),
    )
