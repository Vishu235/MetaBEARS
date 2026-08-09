"""MiniKandinsky adapters and controlled MetaBEARS transformations."""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Callable

import numpy as np

from .interventions import PredictionIntervention


_FIGURE_ORDER = (1, 2, 0)
_FIGURE_INVERSE_ORDER = tuple(np.argsort(_FIGURE_ORDER).tolist())
_SOURCE_PALETTE = (
    (1.0, 0.0, 0.0),  # red
    (1.0, 1.0, 0.0),  # yellow
    (0.0, 0.0, 1.0),  # blue
)
_CYCLIC_PALETTE = (
    _SOURCE_PALETTE[1],
    _SOURCE_PALETTE[2],
    _SOURCE_PALETTE[0],
)
_DESATURATED_PALETTE = (
    (0.35, 0.35, 0.35),
    (0.70, 0.70, 0.70),
    (0.15, 0.15, 0.15),
)


class MiniKandinskyModelAdapter:
    """Expose MiniKandinsky probabilities in MetaBEARS concept format.

    The legacy model returns ``pCS`` as ``[batch, figures, 18]`` where each
    consecutive group of three values is a categorical distribution. The
    generic MetaBEARS collector expects ``[batch, concepts, classes]``.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def eval(self) -> "MiniKandinskyModelAdapter":
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self

    def __call__(self, images: Any) -> Mapping[str, Any]:
        outputs = self.model(images)
        if not isinstance(outputs, Mapping):
            raise ValueError("MiniKandinsky models must return a mapping.")
        if "pCS" not in outputs:
            raise ValueError("MiniKandinsky model output is missing 'pCS'.")

        probabilities = outputs["pCS"]
        shape = getattr(probabilities, "shape", None)
        if shape is None or len(shape) != 3:
            raise ValueError(
                "MiniKandinsky pCS must be shaped [batch, figures, features]."
            )
        if int(shape[-1]) == 3:
            adapted = probabilities
        elif int(shape[-1]) % 3 == 0:
            adapted = probabilities.reshape(int(shape[0]), -1, 3)
        else:
            raise ValueError(
                "MiniKandinsky pCS features must contain groups of three classes."
            )

        converted = dict(outputs)
        converted["pCS"] = adapted
        return converted

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        model = object.__getattribute__(self, "model")
        return getattr(model, name)


class MiniKandinskyTargetLoader:
    """Select the final binary task label from legacy auxiliary targets."""

    def __init__(self, loader: Iterable[Any]) -> None:
        self.loader = loader

    def __iter__(self) -> Iterator[Any]:
        for batch in self.loader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 3:
                raise ValueError(
                    "MiniKandinsky batches must be (images, labels, concepts)."
                )
            images, labels, concepts = batch
            shape = getattr(labels, "shape", None)
            if shape is None or len(shape) < 1:
                raise ValueError("MiniKandinsky labels must include a batch axis.")
            final_labels = labels if len(shape) == 1 else labels[..., -1]
            yield images, final_labels, concepts

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]


class ImageTransformLoader:
    """Re-iterable loader view that transforms images but preserves targets."""

    def __init__(
        self,
        loader: Iterable[Any],
        transform: Callable[[Any, Any, Any], Any],
    ) -> None:
        self.loader = loader
        self.transform = transform

    def __iter__(self) -> Iterator[Any]:
        for images, labels, concepts in self.loader:
            yield self.transform(images, labels, concepts), labels, concepts

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]


def _validate_rgb_images(images: Any) -> tuple:
    shape = getattr(images, "shape", None)
    if shape is None or len(shape) not in {3, 4}:
        raise ValueError(
            "MiniKandinsky images must be [channels, height, width] or "
            "[batch, channels, height, width]."
        )
    if int(shape[-3]) != 3:
        raise ValueError("MiniKandinsky image transformations require RGB inputs.")
    return tuple(int(value) for value in shape)


def permute_minikandinsky_figures(
    images: Any, labels: Any = None, concepts: Any = None
) -> Any:
    """Cyclically permute the three figures in a task-label-preserving way."""

    shape = _validate_rgb_images(images)
    width = shape[-1]
    if width % 3 != 0:
        raise ValueError("MiniKandinsky image width must be divisible by three.")
    figure_width = width // 3

    if isinstance(images, np.ndarray):
        figures = np.split(images, (figure_width, 2 * figure_width), axis=-1)
        return np.concatenate([figures[index] for index in _FIGURE_ORDER], axis=-1)

    try:
        import torch
    except ModuleNotFoundError as error:
        raise TypeError(
            "Non-NumPy MiniKandinsky images require PyTorch."
        ) from error
    if not torch.is_tensor(images):
        raise TypeError("MiniKandinsky images must be NumPy arrays or torch tensors.")
    figures = torch.split(images, figure_width, dim=-1)
    return torch.cat([figures[index] for index in _FIGURE_ORDER], dim=-1)


def align_permuted_minikandinsky_concepts(
    probabilities: np.ndarray,
) -> np.ndarray:
    """Map cyclically permuted figure concepts back to their original order."""

    values = np.asarray(probabilities)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(
            "MiniKandinsky concept probabilities must be "
            "[members, samples, concepts, 3]."
        )
    if values.shape[-2] % 3 != 0:
        raise ValueError("MiniKandinsky concepts must divide evenly over figures.")
    per_figure = values.shape[-2] // 3
    grouped = values.reshape(*values.shape[:2], 3, per_figure, 3)
    aligned = grouped[:, :, _FIGURE_INVERSE_ORDER, :, :]
    return aligned.reshape(values.shape).copy()


def _numpy_palette_transform(images: np.ndarray, target_palette: tuple) -> np.ndarray:
    original_dtype = images.dtype
    values = images.astype(np.float64, copy=False)
    scale = 255.0 if values.size and float(np.nanmax(values)) > 1.5 else 1.0
    values = values / scale
    pixels = np.moveaxis(values, -3, -1)
    white = np.ones(3, dtype=np.float64)
    sources = np.asarray(_SOURCE_PALETTE, dtype=np.float64)
    targets = np.asarray(target_palette, dtype=np.float64)
    directions = sources - white
    delta = pixels[..., np.newaxis, :] - white
    alpha = np.sum(delta * directions, axis=-1) / np.sum(
        directions * directions, axis=-1
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    reconstructed = white + alpha[..., np.newaxis] * directions
    errors = np.sum((pixels[..., np.newaxis, :] - reconstructed) ** 2, axis=-1)
    best = np.argmin(errors, axis=-1)
    selected_alpha = np.take_along_axis(alpha, best[..., np.newaxis], axis=-1)[
        ..., 0
    ]
    selected_error = np.take_along_axis(errors, best[..., np.newaxis], axis=-1)[
        ..., 0
    ]
    selected_targets = targets[best]
    mapped = white + selected_alpha[..., np.newaxis] * (selected_targets - white)
    transformed = np.where(selected_error[..., np.newaxis] <= 0.08, mapped, pixels)
    transformed = np.moveaxis(transformed, -1, -3) * scale
    return transformed.astype(original_dtype, copy=False)


def _torch_palette_transform(images: Any, target_palette: tuple) -> Any:
    import torch

    if not torch.is_tensor(images):
        raise TypeError("MiniKandinsky images must be NumPy arrays or torch tensors.")
    if not torch.is_floating_point(images):
        raise TypeError("PyTorch MiniKandinsky images must use a floating dtype.")
    pixels = torch.movedim(images, -3, -1)
    white = torch.ones(3, dtype=images.dtype, device=images.device)
    sources = torch.tensor(
        _SOURCE_PALETTE, dtype=images.dtype, device=images.device
    )
    targets = torch.tensor(target_palette, dtype=images.dtype, device=images.device)
    directions = sources - white
    delta = pixels.unsqueeze(-2) - white
    alpha = (delta * directions).sum(dim=-1) / (directions * directions).sum(
        dim=-1
    )
    alpha = alpha.clamp(0.0, 1.0)
    reconstructed = white + alpha.unsqueeze(-1) * directions
    errors = ((pixels.unsqueeze(-2) - reconstructed) ** 2).sum(dim=-1)
    best = errors.argmin(dim=-1)
    selected_alpha = torch.gather(alpha, -1, best.unsqueeze(-1)).squeeze(-1)
    selected_error = torch.gather(errors, -1, best.unsqueeze(-1)).squeeze(-1)
    selected_targets = targets[best]
    mapped = white + selected_alpha.unsqueeze(-1) * (selected_targets - white)
    transformed = torch.where(selected_error.unsqueeze(-1) <= 0.08, mapped, pixels)
    return torch.movedim(transformed, -1, -3)


def _palette_transform(images: Any, target_palette: tuple) -> Any:
    _validate_rgb_images(images)
    if isinstance(images, np.ndarray):
        return _numpy_palette_transform(images, target_palette)
    try:
        return _torch_palette_transform(images, target_palette)
    except ModuleNotFoundError as error:
        raise TypeError(
            "Non-NumPy MiniKandinsky images require PyTorch."
        ) from error


def cycle_minikandinsky_palette(
    images: Any, labels: Any = None, concepts: Any = None
) -> Any:
    """Cycle red to yellow, yellow to blue, and blue to red."""

    return _palette_transform(images, _CYCLIC_PALETTE)


def desaturate_minikandinsky_palette(
    images: Any, labels: Any = None, concepts: Any = None
) -> Any:
    """Map the three known colors to distinct grayscale levels for OOD tests."""

    return _palette_transform(images, _DESATURATED_PALETTE)


def align_cycled_minikandinsky_colors(
    probabilities: np.ndarray,
) -> np.ndarray:
    """Map predictions for the cycled palette back to source color classes."""

    values = np.asarray(probabilities)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(
            "MiniKandinsky concept probabilities must be "
            "[members, samples, concepts, 3]."
        )
    if values.shape[-2] != 18:
        raise ValueError("MiniKandinsky palette alignment expects 18 concepts.")
    grouped = values.reshape(*values.shape[:2], 3, 6, 3).copy()
    color_probabilities = grouped[:, :, :, 3:, :].copy()
    grouped[:, :, :, 3:, :] = color_probabilities[..., [1, 2, 0]]
    return grouped.reshape(values.shape)


MINIKANDINSKY_FIGURE_PERMUTE = PredictionIntervention(
    name="figure_permute",
    description=(
        "Cyclically permute the three MiniKandinsky figures. The task checks "
        "whether all figure predicates agree, so the task label is invariant; "
        "concept positions are mapped back before comparison."
    ),
    transform_images=permute_minikandinsky_figures,
    align_concept_probabilities=align_permuted_minikandinsky_concepts,
)

MINIKANDINSKY_PALETTE_CYCLE = PredictionIntervention(
    name="palette_cycle",
    description=(
        "Cycle the red, yellow, and blue palette globally. Equality relations "
        "and the binary task label are preserved; color-class probabilities "
        "are mapped back to their source semantics before comparison."
    ),
    transform_images=cycle_minikandinsky_palette,
    align_concept_probabilities=align_cycled_minikandinsky_colors,
)


def get_minikandinsky_intervention(name: str) -> PredictionIntervention:
    """Resolve a supported MiniKandinsky controlled intervention."""

    interventions = {
        intervention.name: intervention
        for intervention in (
            MINIKANDINSKY_FIGURE_PERMUTE,
            MINIKANDINSKY_PALETTE_CYCLE,
        )
    }
    if name in interventions:
        return interventions[name]
    raise ValueError(f"Unsupported MiniKandinsky intervention: {name}")
