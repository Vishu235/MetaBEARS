"""Distribution-relative neural familiarity scores."""

import numpy as np


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {vector.shape}.")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values.")
    return vector


def familiarity_from_reference(
    representation_distances: np.ndarray,
    reference_distances: np.ndarray,
) -> np.ndarray:
    """Convert representation distances into empirical familiarity scores.

    ``reference_distances`` should be computed on a held-out in-distribution
    validation split using the same embedding and distance function. A score
    near one denotes a distance at the familiar end of that distribution; a
    score near zero denotes a distance at or beyond its unfamiliar tail.
    """

    distances = _finite_vector(
        representation_distances, name="representation_distances"
    )
    reference = _finite_vector(reference_distances, name="reference_distances")
    if reference.size < 2:
        raise ValueError("reference_distances must contain at least two values.")
    if np.any(distances < 0.0) or np.any(reference < 0.0):
        raise ValueError("Representation distances must be non-negative.")

    sorted_reference = np.sort(reference)
    quantiles = np.linspace(0.0, 1.0, sorted_reference.size)
    empirical_cdf = np.interp(
        distances,
        sorted_reference,
        quantiles,
        left=0.0,
        right=1.0,
    )
    return np.clip(1.0 - empirical_cdf, 0.0, 1.0)
