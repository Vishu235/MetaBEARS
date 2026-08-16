"""BDD-OIA adapters for the MetaBEARS diagnostic layer.

BDD-OIA is structurally different from HalfMNIST/MiniKandinsky: its 21 known
concepts and its 4 driving actions (forward, stop, left, right) are each
independent *binary* decisions (sigmoid-style), not one multi-class
categorical variable per concept. The generic MetaBEARS collector
(``metacog.integration.collect_ensemble_predictions``) expects concept
probabilities shaped ``[batch, concepts, classes]`` and a *single* task
distribution shaped ``[batch, task_classes]``.

Each binary concept already maps cleanly onto that shape as a 2-class
distribution ``[1 - p, p]``. The 4 actions do not: there is no single
categorical "task" variable in the source model. This module resolves that by
combining the four independent action-pair distributions into one 16-way
"action combination" categorical (the product measure over the four pairs),
so the existing, tested single-task MetaBEARS machinery can be reused
unmodified rather than forked.

This is a deliberate simplification, not a claim that the four actions are
learned independently by the model. One consequence: the resulting
"task confidence" / "task accuracy" require every one of the four actions to
match simultaneously (the same notion as this project's existing
``action_exact_match`` baseline metric), which is stricter than per-action
correctness. Report both alongside each other; do not present the combined
metric as equivalent to per-action F1.
"""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Tuple

import numpy as np


ACTION_NAMES = ("forward", "stop", "left", "right")
_ACTION_BIT_WEIGHTS = (8, 4, 2, 1)


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "reshape")


def _combine_action_pairs(action_pairs: Any) -> Any:
    """Combine four independent 2-class action distributions into one 16-way categorical.

    Args:
        action_pairs: Array or tensor shaped ``[..., 4, 2]``, where the last
            axis is ``[not-action, action]`` for each of the four actions in
            :data:`ACTION_NAMES` order.

    Returns:
        An array or tensor shaped ``[..., 16]``: the outer product of the
        four pairs, flattened with action 0 (forward) as the most
        significant bit. Index ``i`` decodes as
        ``i = 8*forward + 4*stop + 2*left + 1*right``.
    """

    shape = getattr(action_pairs, "shape", None)
    if shape is None or len(shape) < 2 or int(shape[-2]) != 4 or int(shape[-1]) != 2:
        raise ValueError(
            f"action_pairs must be shaped [..., 4, 2]; got {shape}."
        )

    forward = action_pairs[..., 0, :]
    stop = action_pairs[..., 1, :]
    left = action_pairs[..., 2, :]
    right = action_pairs[..., 3, :]
    combination = (
        forward[..., :, None, None, None]
        * stop[..., None, :, None, None]
        * left[..., None, None, :, None]
        * right[..., None, None, None, :]
    )
    leading_shape = tuple(combination.shape[:-4])
    return combination.reshape(*leading_shape, 16)


def _action_bits_to_index(action_bits: Any) -> Any:
    """Encode four binary action labels as a single index in ``[0, 16)``.

    Uses the same bit order as :func:`_combine_action_pairs`: forward is the
    most significant bit, right is the least significant.
    """

    shape = getattr(action_bits, "shape", None)
    if shape is None or int(shape[-1]) != 4:
        raise ValueError(f"action_bits must have a trailing axis of 4; got {shape}.")

    if _is_torch_tensor(action_bits):
        import torch

        weights = torch.tensor(
            _ACTION_BIT_WEIGHTS, dtype=action_bits.dtype, device=action_bits.device
        )
        return (action_bits * weights).sum(dim=-1).round().to(torch.int64)

    values = np.asarray(action_bits, dtype=np.float64)
    weights = np.array(_ACTION_BIT_WEIGHTS, dtype=np.float64)
    return np.round((values * weights).sum(axis=-1)).astype(np.int64)


def decode_action_combination(index: int) -> Tuple[bool, bool, bool, bool]:
    """Decode a combined action index back into (forward, stop, left, right)."""

    if not 0 <= int(index) < 16:
        raise ValueError("index must lie within [0, 16).")
    value = int(index)
    return tuple(bool(value & weight) for weight in _ACTION_BIT_WEIGHTS)  # type: ignore[return-value]


class BDDModelAdapter:
    """Expose a trained BDD-OIA ``DPL_AUC`` model in MetaBEARS ensemble format.

    ``DPL_AUC.forward`` returns only the 8-dimensional action prediction
    (four action pairs concatenated) and stores the normalized 21-concept
    pair probabilities as the ``pC`` attribute and the raw labeled-concept
    output as ``concepts_labeled``, both as a side effect of the forward
    call. This adapter reads those attributes immediately after calling the
    model to build the ``pCS``/``YS``/``CS`` mapping the generic MetaBEARS
    collector expects, without modifying the underlying model.

    ``CS`` (the representation used for familiarity/OOD distance) is the raw
    labeled-concept output before the pairwise normalization used for
    ``pCS`` — the model's own concept-space representation of the input, not
    a claim that it is analogous to HalfMNIST's pre-softmax logits in every
    respect.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def eval(self) -> "BDDModelAdapter":
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self

    def __call__(self, images: Any) -> Mapping[str, Any]:
        pred = self.model(images)
        pair_probabilities = getattr(self.model, "pC", None)
        if pair_probabilities is None:
            raise ValueError(
                "The wrapped model did not expose 'pC' after forward(); "
                "BDDModelAdapter requires a DPL_AUC-style model."
            )
        representation = getattr(self.model, "concepts_labeled", None)
        if representation is None:
            raise ValueError(
                "The wrapped model did not expose 'concepts_labeled' after "
                "forward(); BDDModelAdapter requires a DPL_AUC-style model."
            )

        batch_size = int(pred.shape[0])
        concept_probabilities = pair_probabilities.reshape(batch_size, -1, 2)
        action_pairs = pred.reshape(batch_size, len(ACTION_NAMES), 2)
        action_combination = _combine_action_pairs(action_pairs)

        return {
            "pCS": concept_probabilities,
            "YS": action_combination,
            "CS": representation.reshape(batch_size, -1),
        }

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        model = object.__getattribute__(self, "model")
        return getattr(model, name)


class BDDTargetLoader:
    """Adapt ``BDD.dataset.load_data`` batches to the MetaBEARS target format.

    ``load_data`` yields ``(image_features, action_labels, concept_labels)``.
    ``action_labels`` is 5-dimensional because
    ``preprocess_lastframe.py`` pads a missing 4th category with a constant
    zero; only the first four dimensions carry real forward/stop/left/right
    labels. This loader drops that padding column and encodes the remaining
    four binary labels as a single combined-action index using the same bit
    order as :func:`_combine_action_pairs`, so ground truth and predictions
    are directly comparable.
    """

    def __init__(self, loader: Iterable[Any]) -> None:
        self.loader = loader

    def __iter__(self) -> Iterator[Any]:
        for batch in self.loader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 3:
                raise ValueError(
                    "BDD-OIA batches must be (images, actions, concepts)."
                )
            images, actions, concepts = batch
            shape = getattr(actions, "shape", None)
            if shape is None or int(shape[-1]) < 4:
                raise ValueError(
                    "BDD-OIA action batches must have at least 4 columns."
                )
            combined = _action_bits_to_index(actions[..., :4])
            yield images, combined, concepts

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]
