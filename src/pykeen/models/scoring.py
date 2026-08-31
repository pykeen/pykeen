"""A unified representation of scoring requests.

The scoring methods of :class:`~pykeen.models.Model` differ only in *which* of the three positions of a triple is
scored against many candidates: :meth:`~pykeen.models.Model.score_t` scores $(h, r, *)$,
:meth:`~pykeen.models.Model.score_h` scores $(*, r, t)$, and :meth:`~pykeen.models.Model.score_r` scores $(h, *, t)$.
:class:`TargetScoringBatch` captures that commonality: it holds an index tensor for each position, plus the `target`
naming the position which is scored against many candidates. Its sibling :class:`TripleScoringBatch` covers the
remaining case, where all three positions are given.

This lets :class:`~pykeen.models.ERModel` implement the three 1:n scoring methods once, cf.
:meth:`~pykeen.models.ERModel._score`, instead of maintaining three near-identical copies of the same broadcasting,
slicing, and repetition logic.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import TypeAlias

import torch

from ..constants import COLUMN_LABELS, TARGET_TO_INDEX
from ..typing import LongTensor, Target, TargetColumn
from ..utils import pad_trailing_dims

__all__ = [
    "ScoringBatch",
    "TargetScoringBatch",
    "TripleScoringBatch",
]


def _broadcast_index_shapes(shapes: Iterable[tuple[int, ...]]) -> tuple[int, ...]:
    """Determine the common shape of the given index shapes.

    :param shapes:
        the shapes of the index tensors

    :raises ValueError:
        if the shapes are not broadcastable

    :return:
        the broadcasted shape
    """
    materialized = list(shapes)
    try:
        return tuple(torch.broadcast_shapes(*materialized))
    except RuntimeError as error:
        raise ValueError(f"Cannot broadcast index shapes {materialized}") from error


def _align_batch_indices(
    indices: Mapping[Target, LongTensor | None],
) -> tuple[Mapping[Target, LongTensor], int, tuple[int, ...]]:
    """Align the index tensors which determine the batch shape.

    :param indices:
        the index tensors which determine the batch shape, i.e., all but the scoring target's

    :raises ValueError:
        if an index tensor is missing, or if the shapes are not broadcastable

    :return:
        the aligned index tensors, the number of batch dimensions, and the batch shape
    """
    missing = sorted(label for label, index in indices.items() if index is None)
    if missing:
        raise ValueError(f"Missing index tensors for {missing}; only the scoring target may be None")
    # the comprehension narrows the value type to LongTensor, given the check above
    present = {label: index for label, index in indices.items() if index is not None}

    # index tensors are left-aligned; pad them so that torch's right-aligned broadcasting agrees
    batch_ndim = max(index.ndim for index in present.values())
    aligned = {label: pad_trailing_dims(index, ndim=batch_ndim) for label, index in present.items()}
    return aligned, batch_ndim, _broadcast_index_shapes(index.shape for index in aligned.values())


@dataclasses.dataclass
class TripleScoringBatch:
    """A request to score the given triples.

    All three index tensors are required, and are broadcast against each other, so that the resulting score tensor
    has shape ``(*batch_shape,)``. This covers :meth:`~pykeen.models.Model.score_hrt`, as well as the general case
    of scoring an arbitrarily shaped block of triples.
    """

    #: shape: broadcastable to ``(*batch_shape,)``
    head: LongTensor

    #: shape: broadcastable to ``(*batch_shape,)``
    relation: LongTensor

    #: shape: broadcastable to ``(*batch_shape,)``
    tail: LongTensor

    #: the number of batch dimensions; inferred from the index tensors
    batch_ndim: int = dataclasses.field(init=False)

    #: the common shape of the batch dimensions; inferred from the index tensors
    batch_shape: tuple[int, ...] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Align the index tensors and infer the batch shape."""
        aligned, self.batch_ndim, self.batch_shape = _align_batch_indices(
            dict(zip(COLUMN_LABELS, self.indices, strict=True))
        )
        self.head, self.relation, self.tail = (aligned[label] for label in COLUMN_LABELS)

    @property
    def indices(self) -> tuple[LongTensor, LongTensor, LongTensor]:
        """Return the index tensors, in the order ``(head, relation, tail)``."""
        return self.head, self.relation, self.tail

    @property
    def lookup_indices(self) -> tuple[LongTensor, LongTensor, LongTensor]:
        """Return the index tensors to look up representations with."""
        return self.indices

    @property
    def device(self) -> torch.device:
        """Return the device of the index tensors."""
        return self.head.device


@dataclasses.dataclass
class TargetScoringBatch:
    """A request to score one position of a triple against many candidates.

    The index tensor of the `target` position may be

    - `None`, to score against *all* candidates,
    - of shape ``(num,)``, to score against the same candidates for each batch element, or
    - of shape ``(*batch_shape, num)``, to score against different candidates per batch element.

    The latter is what grouped sLCWA training uses, cf. :class:`~pykeen.triples.instances.GroupedSLCWABatch`.

    The other two index tensors are required, and determine the batch shape; the resulting score tensor has shape
    ``(*batch_shape, num)``.
    """

    #: shape: broadcastable to ``(*batch_shape,)``, or the target shape described above
    head: LongTensor | None

    #: shape: broadcastable to ``(*batch_shape,)``, or the target shape described above
    relation: LongTensor | None

    #: shape: broadcastable to ``(*batch_shape,)``, or the target shape described above
    tail: LongTensor | None

    #: the position which is scored against many candidates
    target: Target

    #: the number of batch dimensions; inferred from the non-target index tensors
    batch_ndim: int = dataclasses.field(init=False)

    #: the common shape of the batch dimensions; inferred from the non-target index tensors
    batch_shape: tuple[int, ...] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Align the non-target index tensors, infer the batch shape, and validate the target IDs.

        :raises ValueError:
            if the target is invalid, or if the target IDs do not have a usable number of dimensions
        """
        if self.target not in COLUMN_LABELS:
            raise ValueError(f"Unknown target={self.target}; must be one of {COLUMN_LABELS}")

        aligned, self.batch_ndim, self.batch_shape = _align_batch_indices(
            {label: index for label, index in zip(COLUMN_LABELS, self.indices, strict=True) if label != self.target}
        )
        self.head, self.relation, self.tail = (
            aligned.get(label, index) for label, index in zip(COLUMN_LABELS, self.indices, strict=True)
        )

        if self.target_ids is not None and self.target_ids.ndim not in (1, self.batch_ndim + 1):
            raise ValueError(
                f"The target IDs for {self.target} must have shape (num,) or (*batch_shape, num) with "
                f"batch_shape={self.batch_shape}, but have shape {tuple(self.target_ids.shape)}"
            )

    @property
    def indices(self) -> tuple[LongTensor | None, LongTensor | None, LongTensor | None]:
        """Return the index tensors, in the order ``(head, relation, tail)``."""
        return self.head, self.relation, self.tail

    @property
    def target_index(self) -> TargetColumn:
        """Return the index of the target into ``(head, relation, tail)``."""
        return TARGET_TO_INDEX[self.target]

    @property
    def target_ids(self) -> LongTensor | None:
        """Return the target's index tensor, or None if scoring against all candidates."""
        return self.indices[self.target_index]

    @property
    def shared_target(self) -> bool:
        """Return whether the same candidates are scored for each batch element."""
        return self.target_ids is None or self.target_ids.ndim == 1

    @property
    def device(self) -> torch.device:
        """Return the device of the index tensors."""
        return next(index.device for index in self.indices if index is not None)

    @property
    def lookup_indices(self) -> tuple[LongTensor | None, LongTensor | None, LongTensor | None]:
        """Return the index tensors to look up representations with.

        The non-target index tensors receive an additional singleton dimension, so that the looked-up
        representations broadcast against the target's candidate dimension.

        :return:
            the head, relation, and tail index tensors
        """
        # the `index is None` check is redundant - only the target may be None, cf. __post_init__ - but narrows
        head, relation, tail = (
            index if label == self.target or index is None else index.unsqueeze(dim=self.batch_ndim)
            for label, index in zip(COLUMN_LABELS, self.indices, strict=True)
        )
        return head, relation, tail

    def with_target_ids(self, ids: LongTensor) -> TargetScoringBatch:
        """Return a copy of this batch with the target's index tensor replaced.

        :param ids:
            the new target index tensor

        :return:
            the new batch
        """
        return dataclasses.replace(self, **{self.target: ids})


#: A scoring request: either the given triples, or one position scored against many candidates.
ScoringBatch: TypeAlias = TripleScoringBatch | TargetScoringBatch
