"""Label-preserving interventions for controlled MetaBEARS evaluation."""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class PredictionIntervention:
    """An input transform and its known concept-axis alignment.

    ``transform_images`` produces a label-preserving input. The model predicts
    concepts in the transformed coordinate system, so
    ``align_concept_probabilities`` maps those probabilities back to the
    original concept order before consistency is measured.
    """

    name: str
    description: str
    transform_images: Callable[[Any], Any]
    align_concept_probabilities: Callable[[np.ndarray], np.ndarray]


def swap_halfmnist_image_halves(images: Any) -> Any:
    """Swap the two digit images in a concatenated HalfMNIST input.

    HalfMNIST addition is commutative, so the task label is unchanged. The
    digit concepts exchange positions and are aligned back separately.
    """

    shape = getattr(images, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("HalfMNIST images must include a width dimension.")
    width = int(shape[-1])
    if width < 2 or width % 2 != 0:
        raise ValueError("HalfMNIST image width must be a positive even number.")
    midpoint = width // 2

    if isinstance(images, np.ndarray):
        return np.concatenate(
            [images[..., midpoint:], images[..., :midpoint]], axis=-1
        )

    try:
        import torch
    except ModuleNotFoundError as error:
        raise TypeError(
            "Non-NumPy HalfMNIST images require PyTorch for concatenation."
        ) from error
    if not torch.is_tensor(images):
        raise TypeError("HalfMNIST images must be a NumPy array or torch tensor.")
    return torch.cat([images[..., midpoint:], images[..., :midpoint]], dim=-1)


def align_swapped_concept_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Map swapped-pair concept probabilities back to the original order."""

    aligned = np.asarray(probabilities)
    if aligned.ndim != 4:
        raise ValueError(
            "Concept probabilities must be shaped "
            "[members, samples, concepts, classes]."
        )
    if aligned.shape[-2] != 2:
        raise ValueError("The HalfMNIST swap intervention requires two concepts.")
    return np.flip(aligned, axis=-2).copy()


HALFMNIST_HALF_SWAP = PredictionIntervention(
    name="half_swap",
    description=(
        "Swap the left and right HalfMNIST digits. Addition is label-preserving; "
        "predicted concept positions are reversed back before comparison."
    ),
    transform_images=swap_halfmnist_image_halves,
    align_concept_probabilities=align_swapped_concept_probabilities,
)


def get_intervention(name: str) -> PredictionIntervention:
    """Resolve a supported controlled intervention by CLI name."""

    if name == HALFMNIST_HALF_SWAP.name:
        return HALFMNIST_HALF_SWAP
    raise ValueError(f"Unsupported MetaBEARS intervention: {name}")
