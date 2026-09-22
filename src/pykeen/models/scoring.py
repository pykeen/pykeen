r"""A unified representation of scoring requests.

The scoring methods of :class:`~pykeen.models.Model` differ only in *which* of the three
positions of a triple is scored against many candidates:
:meth:`~pykeen.models.Model.score_t` scores $(h, r, \*)$,
:meth:`~pykeen.models.Model.score_h` scores $(\*, r, t)$, and
:meth:`~pykeen.models.Model.score_r` scores $(h, \*, t)$. :class:`TargetScoringBatch`
captures that commonality: it holds an index tensor for each position, plus the `target`
naming the position which is scored against many candidates. Its sibling
:class:`TripleScoringBatch` covers the remaining case, where all three positions are
given.

This lets :class:`~pykeen.models.ERModel` implement the three ``1:n`` scoring methods
once, cf. :meth:`~pykeen.models.ERModel._score`, instead of maintaining three
near-identical copies of the same broadcasting, slicing, and repetition logic.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, NamedTuple, Self, TypeAlias, TypeVar, overload

import torch

from ..constants import COLUMN_LABELS, TARGET_TO_INDEX
from ..typing import LongTensor, Target
from ..utils import pad_trailing_dims

__all__ = [
    "ScoringBatch",
    "Indices",
    "OptionalIndices",
    "TargetScoringBatch",
    "TripleScoringBatch",
]


class Indices(NamedTuple):
    """The index tensors of a scoring request, one per triple position."""

    head: LongTensor

    relation: LongTensor

    tail: LongTensor

    @classmethod
    def from_batch(cls, batch: LongTensor) -> Self:
        """Construct indices from an HRT batch."""
        return cls(batch[:, 0], batch[:, 1], batch[:, 2])


class OptionalIndices(NamedTuple):
    """The index tensors of a scoring request, where the scoring target's may be missing."""

    #: shape: broadcastable to ``(*batch_shape,)``, or the target shape described above
    head: LongTensor | None

    #: shape: broadcastable to ``(*batch_shape,)``, or the target shape described above
    relation: LongTensor | None

    #: shape: broadcastable to ``(*batch_shape,)``, or the target shape described above
    tail: LongTensor | None

    @property
    def device(self) -> torch.device:
        """Return the device of the index tensors."""
        return next(index.device for index in self if index is not None)

    def with_target_ids(self, ids: LongTensor, target: Target) -> Self:
        """Return a copy of this batch with the target's index tensor replaced.

        :param ids: the new target index tensor
        :param target: the target where to put the new tensor

        :returns: the new batch
        """
        match target:
            case "head":
                return self.__class__(ids, self.relation, self.tail)
            case "relation":
                return self.__class__(self.head, ids, self.tail)
            case "tail":
                return self.__class__(self.head, self.relation, ids)


_IndicesType = TypeVar("_IndicesType", Indices, OptionalIndices)


class _AlignedIndices(NamedTuple, Generic[_IndicesType]):
    """The result of aligning a scoring request's index tensors."""

    #: the index tensors, with all but the scoring target's aligned
    indices: _IndicesType

    #: the number of batch dimensions
    batch_ndim: int

    #: the common shape of the batch dimensions
    batch_shape: tuple[int, ...]


def _broadcast_index_shapes(shapes: Iterable[tuple[int, ...]]) -> tuple[int, ...]:
    """Determine the common shape of the given index shapes.

    :param shapes: the shapes of the index tensors; they must have the same number of
        dimensions, cf. :func:`~pykeen.utils.pad_trailing_dims`

    :returns: the broadcasted shape

    :raises ValueError: if the shapes are not broadcastable
    """
    # note: this is equivalent to torch.broadcast_shapes for equal-ndim shapes, but about an order of magnitude
    # faster, and scoring constructs one batch per call
    materialized = list(shapes)
    result = []
    for sizes in zip(*materialized, strict=True):
        if len(set(sizes) - {1}) > 1:
            raise ValueError(f"Cannot broadcast index shapes {materialized}")
        result.append(max(sizes))
    return tuple(result)


@overload
def _align_batch_indices(indices: Indices, target: None = ...) -> _AlignedIndices[Indices]: ...


@overload
def _align_batch_indices(indices: OptionalIndices, target: Target) -> _AlignedIndices[OptionalIndices]: ...


def _align_batch_indices(
    indices: Indices | OptionalIndices, target: Target | None = None
) -> _AlignedIndices[Indices] | _AlignedIndices[OptionalIndices]:
    """Align the index tensors which determine the batch shape.

    :param indices: the index tensors, in the order ``(head, relation, tail)``
    :param target: the scoring target, if any; its index tensor does not determine the
        batch shape and is passed through unchanged

    :returns: the aligned index tensors, cf. :class:`_AlignedIndices`

    :raises ValueError: if a non-target index tensor is missing, or if the shapes are
        not broadcastable
    """
    batch_indices: list[LongTensor] = []
    for label, index in zip(COLUMN_LABELS, indices, strict=True):
        if label == target:
            continue
        if index is None:
            raise ValueError(f"Missing index tensor for {label}; only the scoring target may be None")
        batch_indices.append(index)

    # index tensors are left-aligned; pad them so that torch's right-aligned broadcasting agrees
    batch_ndim = max(index.ndim for index in batch_indices)
    aligned = [pad_trailing_dims(index, ndim=batch_ndim) for index in batch_indices]
    batch_shape = _broadcast_index_shapes(index.shape for index in aligned)

    # without a target, every position took part in the alignment, and none of them can be None
    if target is None:
        return _AlignedIndices(
            indices=Indices(*aligned),
            batch_ndim=batch_ndim,
            batch_shape=batch_shape,
        )

    aligned_iter = iter(aligned)
    return _AlignedIndices(
        indices=OptionalIndices(
            *(
                index if label == target else next(aligned_iter)
                for label, index in zip(COLUMN_LABELS, indices, strict=True)
            )
        ),
        batch_ndim=batch_ndim,
        batch_shape=batch_shape,
    )


class TripleScoringBatch(NamedTuple):
    """A request to score the given triples.

    All three index tensors are required, and are broadcast against each other, so that
    the resulting score tensor has shape ``(*batch_shape,)``. This covers
    :meth:`~pykeen.models.Model.score_hrt`, as well as the general case of scoring an
    arbitrarily shaped block of triples.
    """

    indices: Indices

    #: the number of batch dimensions; inferred from the index tensors
    batch_ndim: int

    #: the common shape of the batch dimensions; inferred from the index tensors
    batch_shape: tuple[int, ...]

    @classmethod
    def from_batch(cls, batch: LongTensor) -> Self:
        """Construct from an HRT batch."""
        return cls.from_indices(Indices.from_batch(batch))

    @classmethod
    def from_transposed_batch(cls, head: LongTensor, relation: LongTensor, tail: LongTensor) -> Self:
        """Construct from a transposed HRT batch."""
        return cls.from_indices(Indices(head, relation, tail))

    @classmethod
    def from_indices(cls, indices: Indices) -> Self:
        """Construct from an indices object."""
        indices_, batch_ndim, batch_shape = _align_batch_indices(indices)
        return cls(indices_, batch_ndim, batch_shape)


class TargetScoringBatch(NamedTuple):
    """A request to score one position of a triple against many candidates.

    The index tensor of the `target` position may be one of:

    - `None`, to score against *all* candidates
    - of shape ``(num,)``, to score against the same candidates for each batch element
    - of shape ``(*batch_shape, num)``, to score against different candidates per batch
      element.

    The latter is what grouped sLCWA training uses, cf.
    :class:`~pykeen.triples.instances.GroupedSLCWABatch`.

    The other two index tensors are required, and determine the batch shape; the
    resulting score tensor has shape ``(*batch_shape, num)``.
    """

    indices: OptionalIndices

    #: the position which is scored against many candidates
    target: Target

    #: the number of batch dimensions; inferred from the non-target index tensors
    batch_ndim: int

    #: the common shape of the batch dimensions; inferred from the non-target index tensors
    batch_shape: tuple[int, ...]

    @classmethod
    def from_transposed_batch(
        cls,
        head: LongTensor | None,
        relation: LongTensor | None,
        tail: LongTensor | None,
        target: Target,
    ) -> Self:
        """Construct from an HRT batch."""
        return cls.from_indices(OptionalIndices(head, relation, tail), target)

    @classmethod
    def from_indices(cls, indices: OptionalIndices, target: Target) -> Self:
        """Align the non-target index tensors, infer the batch shape, and validate the target IDs.

        :raises ValueError: if the target is invalid, or if the target IDs do not have a
            usable number of dimensions

        """
        if target not in COLUMN_LABELS:
            raise ValueError(f"Unknown target={target}; must be one of {COLUMN_LABELS}")

        indices_, batch_ndim, batch_shape = _align_batch_indices(indices, target=target)
        rv = cls(indices_, target, batch_ndim, batch_shape)

        target_ids = rv.target_ids
        if target_ids is not None and target_ids.ndim not in (1, batch_ndim + 1):
            raise ValueError(
                f"The target IDs for {target} must have shape (num,) or (*batch_shape, num) with "
                f"batch_shape={batch_shape}, but have shape {tuple(target_ids.shape)}"
            )

        return rv

    @property
    def target_ids(self) -> LongTensor | None:
        """Return the target's index tensor, or None if scoring against all candidates."""
        return self.indices[TARGET_TO_INDEX[self.target]]

    @property
    def shared_target(self) -> bool:
        """Return whether the same candidates are scored for each batch element."""
        return self.target_ids is None or self.target_ids.ndim == 1

    @property
    def device(self) -> torch.device:
        """Return the device of the index tensors."""
        return self.indices.device

    @property
    def lookup_indices(self) -> OptionalIndices:
        """Return the index tensors to look up representations with.

        The non-target index tensors receive an additional singleton dimension, so that
        the looked-up representations broadcast against the target's candidate
        dimension.

        :returns: the head, relation, and tail index tensors
        """
        # the `index is None` check is redundant - only the target may be None, cf. __post_init__ - but narrows
        return OptionalIndices(
            *(
                index if label == self.target or index is None else index.unsqueeze(dim=self.batch_ndim)
                for label, index in zip(COLUMN_LABELS, self.indices, strict=True)
            )
        )

    def with_target_ids(self, ids: LongTensor) -> Self:
        """Return a copy of this batch with the target's index tensor replaced.

        :param ids: the new target index tensor

        :returns: the new batch
        """
        return self.__class__(
            self.indices.with_target_ids(ids, self.target),
            self.target,
            self.batch_ndim,
            self.batch_shape,
        )


#: A scoring request: either the given triples, or one position scored against many candidates.
ScoringBatch: TypeAlias = TripleScoringBatch | TargetScoringBatch
