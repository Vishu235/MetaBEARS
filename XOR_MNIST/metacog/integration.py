"""Adapters from BEARS ensemble outputs to the MetaBEARS diagnostic layer."""

from contextlib import nullcontext
from dataclasses import dataclass
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


def _to_numpy(values: Any, *, name: str) -> np.ndarray:
    """Detach a tensor-like value and return it as a NumPy array."""

    converted = values
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    try:
        return np.asarray(converted)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} cannot be converted to a NumPy array.") from error


def _probability_array(values: Any, *, name: str, dimensions: int) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != dimensions:
        raise ValueError(
            f"{name} must have {dimensions} dimensions; got "
            f"{probabilities.shape}."
        )
    if probabilities.shape[-1] < 2:
        raise ValueError(f"{name} must contain at least two classes.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(f"{name} values must lie within [0, 1].")
    if not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError(f"{name} must sum to one along its class axis.")
    return probabilities


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def _inference_context() -> Any:
    """Use ``torch.no_grad`` when a BEARS model has already imported torch."""

    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return nullcontext()
    return torch_module.no_grad()


def _move_to_device(values: Any, device: Any) -> Any:
    if device is not None and hasattr(values, "to"):
        return values.to(device)
    return values


@dataclass(frozen=True)
class EnsemblePredictions:
    """Member-preserving predictions collected from a BEARS data loader.

    The probability fields have shapes ``[members, samples, concepts, classes]``
    and ``[members, samples, task_classes]``. ``member_representations`` has
    shape ``[members, samples, features]``. For HalfMNIST DPL, the collector's
    default representation is the flattened ``CS`` tensor: learned pre-softmax
    concept logits whose coordinates retain the same digit semantics across
    independently trained members.
    """

    concept_member_probabilities: np.ndarray
    label_member_probabilities: np.ndarray
    member_representations: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    concepts: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        concept_probabilities = _probability_array(
            self.concept_member_probabilities,
            name="concept_member_probabilities",
            dimensions=4,
        )
        label_probabilities = _probability_array(
            self.label_member_probabilities,
            name="label_member_probabilities",
            dimensions=3,
        )
        if concept_probabilities.shape[:2] != label_probabilities.shape[:2]:
            raise ValueError(
                "Concept and label probabilities must have matching member "
                "and sample axes."
            )
        if concept_probabilities.shape[0] < 2:
            raise ValueError("At least two ensemble members are required.")

        object.__setattr__(
            self, "concept_member_probabilities", concept_probabilities
        )
        object.__setattr__(self, "label_member_probabilities", label_probabilities)

        sample_count = concept_probabilities.shape[1]
        if self.member_representations is not None:
            representations = np.asarray(
                self.member_representations, dtype=np.float64
            )
            expected_prefix = (concept_probabilities.shape[0], sample_count)
            if (
                representations.ndim != 3
                or representations.shape[:2] != expected_prefix
            ):
                raise ValueError(
                    "member_representations must be shaped "
                    "[members, samples, features]."
                )
            if representations.shape[-1] < 1:
                raise ValueError(
                    "member_representations must contain at least one feature."
                )
            if not np.all(np.isfinite(representations)):
                raise ValueError("member_representations contains non-finite values.")
            object.__setattr__(self, "member_representations", representations)

        for field_name in ("labels", "concepts"):
            values = getattr(self, field_name)
            if values is None:
                continue
            array = np.asarray(values)
            if array.ndim == 0 or array.shape[0] != sample_count:
                raise ValueError(
                    f"{field_name} must contain one entry per collected sample."
                )
            object.__setattr__(self, field_name, array)


def collect_ensemble_predictions(
    models: Sequence[Any],
    loader: Iterable[Any],
    *,
    apply_label_softmax: bool = False,
    representation_key: Optional[str] = "CS",
    device: Any = None,
) -> EnsemblePredictions:
    """Collect BEARS outputs without averaging away the ensemble member axis.

    Each loader batch must be ``(images, labels, concepts)``. Models must return
    mappings containing ``pCS`` and ``YS``. By default ``CS`` is flattened into
    a per-member representation suitable for reference-distance computation.
    Set ``representation_key=None`` to collect probabilities only, or provide a
    different output key when a model exposes a dedicated trained embedding.

    ``apply_label_softmax`` is intended for BEARS variants whose ``YS`` output
    contains logits. It must remain false for DPL, where ``YS`` is already a
    normalized ProbLog query distribution.
    """

    if len(models) < 2:
        raise ValueError("At least two ensemble members are required.")
    for model in models:
        if hasattr(model, "eval"):
            model.eval()

    concept_batches = []
    label_batches = []
    representation_batches = []
    label_targets = []
    concept_targets = []

    with _inference_context():
        for batch_index, batch in enumerate(loader):
            if not isinstance(batch, (tuple, list)) or len(batch) != 3:
                raise ValueError(
                    "Each loader batch must be (images, labels, concepts)."
                )
            images, labels, concepts = batch

            output_dicts = []
            for member_index, model in enumerate(models):
                member_device = (
                    device if device is not None else getattr(model, "device", None)
                )
                outputs = model(_move_to_device(images, member_device))
                if not isinstance(outputs, Mapping):
                    raise ValueError(
                        f"Ensemble member {member_index} must return a mapping."
                    )
                for required_key in ("pCS", "YS"):
                    if required_key not in outputs:
                        raise ValueError(
                            f"Ensemble member {member_index} output is missing "
                            f"'{required_key}'."
                        )
                if representation_key is not None and representation_key not in outputs:
                    raise ValueError(
                        f"Ensemble member {member_index} output is missing "
                        f"representation key '{representation_key}'."
                    )
                output_dicts.append(outputs)

            label_batch = np.stack(
                [
                    _to_numpy(outputs["YS"], name=f"batch {batch_index} YS")
                    for outputs in output_dicts
                ],
                axis=0,
            )
            if apply_label_softmax:
                label_batch = _softmax(label_batch.astype(np.float64))
            concept_batch = np.stack(
                [
                    _to_numpy(outputs["pCS"], name=f"batch {batch_index} pCS")
                    for outputs in output_dicts
                ],
                axis=0,
            )

            if label_batch.ndim != 3:
                raise ValueError(
                    "YS outputs must be shaped [batch, classes]; got "
                    f"{label_batch.shape}."
                )
            if concept_batch.ndim != 4:
                raise ValueError(
                    "pCS outputs must be shaped [batch, concepts, classes]; "
                    f"got {concept_batch.shape}."
                )
            if label_batch.shape[:2] != concept_batch.shape[:2]:
                raise ValueError(
                    "YS and pCS outputs must have matching member and batch axes."
                )

            batch_size = label_batch.shape[1]
            labels_array = _to_numpy(labels, name=f"batch {batch_index} labels")
            concepts_array = _to_numpy(
                concepts, name=f"batch {batch_index} concepts"
            )
            if labels_array.ndim == 0 or labels_array.shape[0] != batch_size:
                raise ValueError("Each label batch must align with its image batch.")
            if concepts_array.ndim == 0 or concepts_array.shape[0] != batch_size:
                raise ValueError("Each concept batch must align with its image batch.")

            concept_batches.append(concept_batch)
            label_batches.append(label_batch)
            label_targets.append(labels_array)
            concept_targets.append(concepts_array)

            if representation_key is not None:
                member_representations = []
                for outputs in output_dicts:
                    representation = _to_numpy(
                        outputs[representation_key],
                        name=f"batch {batch_index} {representation_key}",
                    )
                    if representation.ndim < 2 or representation.shape[0] != batch_size:
                        raise ValueError(
                            f"{representation_key} must have a batch axis followed "
                            "by at least one feature axis."
                        )
                    member_representations.append(
                        representation.reshape(batch_size, -1)
                    )
                representation_batches.append(
                    np.stack(member_representations, axis=0)
                )

    if not concept_batches:
        raise ValueError("The loader did not yield any batches.")

    return EnsemblePredictions(
        concept_member_probabilities=np.concatenate(concept_batches, axis=1),
        label_member_probabilities=np.concatenate(label_batches, axis=1),
        member_representations=(
            np.concatenate(representation_batches, axis=1)
            if representation_key is not None
            else None
        ),
        labels=np.concatenate(label_targets, axis=0),
        concepts=np.concatenate(concept_targets, axis=0),
    )


def _representation_array(values: Any, *, name: str) -> np.ndarray:
    representations = np.asarray(values, dtype=np.float64)
    if representations.ndim != 3:
        raise ValueError(
            f"{name} must be shaped [members, samples, features]; got "
            f"{representations.shape}."
        )
    if representations.shape[0] < 1 or representations.shape[1] < 1:
        raise ValueError(f"{name} must contain members and samples.")
    if representations.shape[2] < 1:
        raise ValueError(f"{name} must contain at least one feature.")
    if not np.all(np.isfinite(representations)):
        raise ValueError(f"{name} contains non-finite values.")
    return representations


def _member_nearest_distances(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    chunk_size: int,
    exclude_matching_index: bool,
) -> np.ndarray:
    distances = np.empty((queries.shape[0], queries.shape[1]), dtype=np.float64)
    for member_index in range(queries.shape[0]):
        member_queries = queries[member_index]
        member_references = references[member_index]
        reference_norms = np.sum(member_references**2, axis=1)[np.newaxis, :]

        for start in range(0, member_queries.shape[0], chunk_size):
            stop = min(start + chunk_size, member_queries.shape[0])
            query_chunk = member_queries[start:stop]
            squared_distances = (
                np.sum(query_chunk**2, axis=1)[:, np.newaxis]
                + reference_norms
                - 2.0 * query_chunk @ member_references.T
            )
            np.maximum(squared_distances, 0.0, out=squared_distances)
            if exclude_matching_index:
                row_indices = np.arange(stop - start)
                column_indices = np.arange(start, stop)
                squared_distances[row_indices, column_indices] = np.inf
            distances[member_index, start:stop] = np.sqrt(
                squared_distances.min(axis=1)
            )
    return distances


def ensemble_nearest_reference_distances(
    query_representations: np.ndarray,
    reference_representations: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Return each query's mean member-wise nearest-reference distance.

    Nearest neighbours are found independently in each ensemble member's own
    representation space. Only the resulting scalar distances are averaged,
    so coordinates from independently trained networks are never averaged.
    """

    queries = _representation_array(
        query_representations, name="query_representations"
    )
    references = _representation_array(
        reference_representations, name="reference_representations"
    )
    if queries.shape[0] != references.shape[0]:
        raise ValueError("Query and reference member counts must match.")
    if queries.shape[2] != references.shape[2]:
        raise ValueError("Query and reference feature counts must match.")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")

    member_distances = _member_nearest_distances(
        queries,
        references,
        chunk_size=chunk_size,
        exclude_matching_index=False,
    )
    return member_distances.mean(axis=0)


def ensemble_leave_one_out_reference_distances(
    reference_representations: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Build an in-distribution distance baseline without self-neighbours.

    For every validation sample and member, the function finds the nearest
    *other* validation representation. It then averages those distances across
    members, producing the reference distribution consumed by
    :func:`familiarity_from_reference`.
    """

    references = _representation_array(
        reference_representations, name="reference_representations"
    )
    if references.shape[1] < 2:
        raise ValueError("At least two reference samples are required.")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")

    member_distances = _member_nearest_distances(
        references,
        references,
        chunk_size=chunk_size,
        exclude_matching_index=True,
    )
    return member_distances.mean(axis=0)
