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
    transform_images: Callable[[Any, Any, Any], Any]
    align_concept_probabilities: Callable[[np.ndarray], np.ndarray]


def swap_halfmnist_image_halves(
    images: Any, labels: Any = None, concepts: Any = None
) -> Any:
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


def _label_array(labels: Any) -> np.ndarray:
    converted = labels
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    return np.asarray(converted, dtype=np.int64).reshape(-1)


def apply_halfmnist_label_patch(
    images: Any,
    labels: Any,
    *,
    mode: str = "correlated",
    patch_size: int = 3,
) -> Any:
    """Add a canonical-pair patch encoding a HalfMNIST addition label.

    Five fixed cells are reserved in the top of each 28-pixel digit half. For
    task label ``y``, the left patch encodes ``floor(y/2)`` and the right patch
    encodes ``y-floor(y/2)`` as one-hot pseudo-digits. Their symbolic sum is
    therefore ``y`` even when the visible digits represent different concepts.

    ``correlated`` encodes the true task label, ``conflict`` encodes the next
    cyclic label, and ``neutral`` fills every reserved cell with 0.5.
    """

    if mode not in {"correlated", "conflict", "neutral"}:
        raise ValueError("Patch mode must be correlated, conflict, or neutral.")
    if not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError("patch_size must be a positive integer.")

    shape = getattr(images, "shape", None)
    if shape is None or len(shape) not in {3, 4}:
        raise ValueError(
            "HalfMNIST patch inputs must be [channels, height, width] or "
            "[batch, channels, height, width]."
        )
    unbatched = len(shape) == 3
    width = int(shape[-1])
    height = int(shape[-2])
    if width < 2 or width % 2 != 0:
        raise ValueError("HalfMNIST image width must be a positive even number.")
    half_width = width // 2
    cell_count = 5
    gap = 1
    required_width = cell_count * patch_size + (cell_count + 1) * gap
    if half_width < required_width or height < patch_size + 2 * gap:
        raise ValueError("HalfMNIST image is too small for the label patch.")

    if isinstance(images, np.ndarray):
        patched = images.copy()
    else:
        try:
            import torch
        except ModuleNotFoundError as error:
            raise TypeError(
                "Non-NumPy HalfMNIST images require PyTorch patch support."
            ) from error
        if not torch.is_tensor(images):
            raise TypeError("HalfMNIST images must be a NumPy array or torch tensor.")
        patched = images.clone()

    batch = patched[np.newaxis, ...] if unbatched else patched
    target_labels = _label_array(labels)
    if target_labels.shape[0] != batch.shape[0]:
        raise ValueError("Patch labels must contain one value per image.")
    if np.any(target_labels < 0) or np.any(target_labels > 8):
        raise ValueError("HalfMNIST addition patch labels must lie in [0, 8].")

    cell_starts = [gap + index * (patch_size + gap) for index in range(cell_count)]
    row_start = gap
    for sample_index, task_label in enumerate(target_labels):
        encoded_label = (
            (int(task_label) + 1) % 9 if mode == "conflict" else int(task_label)
        )
        pseudo_digits = (
            encoded_label // 2,
            encoded_label - encoded_label // 2,
        )
        for half_index, pseudo_digit in enumerate(pseudo_digits):
            half_offset = half_index * half_width
            for cell_index, column_start in enumerate(cell_starts):
                value = (
                    0.5
                    if mode == "neutral"
                    else float(cell_index == pseudo_digit)
                )
                batch[
                    sample_index,
                    :,
                    row_start : row_start + patch_size,
                    half_offset
                    + column_start : half_offset
                    + column_start
                    + patch_size,
                ] = value
    return patched


class HalfMNISTLabelPatchDataset:
    """Dataset view that injects a task-correlated patch at access time."""

    def __init__(self, dataset: Any, *, patch_size: int = 3) -> None:
        self.dataset = dataset
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        image, label, concepts = self.dataset[index]
        return (
            apply_halfmnist_label_patch(
                image,
                label,
                mode="correlated",
                patch_size=self.patch_size,
            ),
            label,
            concepts,
        )

    def __getattr__(self, name: str) -> Any:
        if name == "dataset":
            raise AttributeError(name)
        dataset = object.__getattribute__(self, "dataset")
        return getattr(dataset, name)


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


def align_identity_concept_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Return concept probabilities unchanged after validating their axes."""

    aligned = np.asarray(probabilities)
    if aligned.ndim != 4:
        raise ValueError(
            "Concept probabilities must be shaped "
            "[members, samples, concepts, classes]."
        )
    return aligned.copy()


def neutralize_halfmnist_label_patch(
    images: Any, labels: Any, concepts: Any = None
) -> Any:
    """Replace every label-patch cell with an uninformative value."""

    return apply_halfmnist_label_patch(images, labels, mode="neutral")


def contradict_halfmnist_label_patch(
    images: Any, labels: Any, concepts: Any = None
) -> Any:
    """Replace the patch with one encoding an incorrect cyclic task label."""

    return apply_halfmnist_label_patch(images, labels, mode="conflict")


HALFMNIST_HALF_SWAP = PredictionIntervention(
    name="half_swap",
    description=(
        "Swap the left and right HalfMNIST digits. Addition is label-preserving; "
        "predicted concept positions are reversed back before comparison."
    ),
    transform_images=swap_halfmnist_image_halves,
    align_concept_probabilities=align_swapped_concept_probabilities,
)

HALFMNIST_PATCH_NEUTRAL = PredictionIntervention(
    name="patch_neutral",
    description=(
        "Replace the task-correlated canonical-pair patch with neutral cells "
        "while preserving the visible digits, concepts, and addition label."
    ),
    transform_images=neutralize_halfmnist_label_patch,
    align_concept_probabilities=align_identity_concept_probabilities,
)

HALFMNIST_PATCH_CONFLICT = PredictionIntervention(
    name="patch_conflict",
    description=(
        "Replace the task-correlated canonical-pair patch with a patch encoding "
        "the next cyclic sum while preserving the visible digits and true label."
    ),
    transform_images=contradict_halfmnist_label_patch,
    align_concept_probabilities=align_identity_concept_probabilities,
)


def get_intervention(name: str) -> PredictionIntervention:
    """Resolve a supported controlled intervention by CLI name."""

    interventions = {
        item.name: item
        for item in (
            HALFMNIST_HALF_SWAP,
            HALFMNIST_PATCH_NEUTRAL,
            HALFMNIST_PATCH_CONFLICT,
        )
    }
    if name in interventions:
        return interventions[name]
    raise ValueError(f"Unsupported MetaBEARS intervention: {name}")
