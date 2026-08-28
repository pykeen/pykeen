"""A unified representation of scoring requests.

The scoring methods of :class:`~pykeen.models.Model` differ only in *which* of the three positions of a triple is
scored against many candidates: :meth:`~pykeen.models.Model.score_t` scores $(h, r, *)$,
:meth:`~pykeen.models.Model.score_h` scores $(*, r, t)$, and :meth:`~pykeen.models.Model.score_r` scores $(h, *, t)$.
:class:`ScoringBatch` captures that commonality: it holds an index tensor for each position, plus the `target`
naming the position which is scored against many candidates.

This lets :class:`~pykeen.models.ERModel` implement the three 1:n scoring methods once, cf.
:meth:`~pykeen.models.ERModel._score`, instead of maintaining three near-identical copies of the same broadcasting,
slicing, and repetition logic.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import cast

import torch

from ..constants import COLUMN_LABELS
from ..typing import LongTensor, Target

__all__ = [
    "ScoringBatch",
]


def broadcast_index_shapes(shapes: Iterable[tuple[int, ...]]) -> tuple[int, ...]:
    """Determine the common shape of the given index shapes.

    :param shapes:
        the shapes of the index tensors

    :raises ValueError:
        if the shapes are not broadcastable

    :return:
        the broadcasted shape
    """
    shapes = list(shapes)
    try:
        return tuple(torch.broadcast_shapes(*shapes))
    except RuntimeError as error:
        raise ValueError(f"Cannot broadcast index shapes {shapes}") from error


def _pad_trailing_dims(x: LongTensor, ndim: int) -> LongTensor:
    """Append singleton dimensions until the tensor has the given number of dimensions.

    Index tensors are aligned from the *left* (the batch dimensions come first), whereas :mod:`torch` broadcasts
    from the right, so the padding has to be explicit.

    :param x:
        the index tensor
    :param ndim:
        the target number of dimensions; must be at least `x.ndim`

    :return:
        the index tensor with trailing singleton dimensions appended
    """
    if x.ndim == ndim:
        return x
    return x.view(*x.shape, *(1,) * (ndim - x.ndim))


@dataclasses.dataclass
class ScoringBatch:
    """A scoring request.

    There are two modes:

    - If `target` is `None`, all three index tensors must be given, and the scores are computed by broadcasting
      them against each other. This covers :meth:`~pykeen.models.Model.score_hrt`, as well as the general case of
      scoring an arbitrarily shaped block of triples.
    - Otherwise, `target` names the position which is scored against many candidates. The index tensor of that
      position may be

      - `None`, to score against *all* candidates,
      - of shape `(num,)`, to score against the same candidates for each batch element, or
      - of shape `(*batch_shape, num)`, to score against different candidates per batch element.

      The latter is what grouped sLCWA training uses, cf. :class:`~pykeen.triples.instances.GroupedSLCWABatch`.

    The resulting score tensor has shape `(*batch_shape,)` in the first mode, and `(*batch_shape, num)` in the
    second one.
    """

    #: shape: ``(*batch_shape,)``, or the target shape described above
    head: LongTensor | None

    #: shape: ``(*batch_shape,)``, or the target shape described above
    relation: LongTensor | None

    #: shape: ``(*batch_shape,)``, or the target shape described above
    tail: LongTensor | None

    #: the position scored against many candidates, or None for plain broadcasting
    target: Target | None = None

    #: the number of batch dimensions; inferred from the non-target index tensors
    batch_ndim: int = dataclasses.field(init=False)

    #: the common shape of the batch dimensions; inferred from the non-target index tensors
    batch_shape: tuple[int, ...] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Normalize and validate the index tensors.

        :raises ValueError:
            if the target is invalid, if a required index tensor is missing, or if the shapes do not fit together
        """
        if self.target is not None and self.target not in COLUMN_LABELS:
            raise ValueError(f"Unknown target={self.target}; must be None or one of {COLUMN_LABELS}")

        raw = {label: index for label, index in zip(COLUMN_LABELS, self.indices, strict=True) if label != self.target}
        missing = sorted(label for label, index in raw.items() if index is None)
        if missing:
            raise ValueError(f"Missing index tensors for {missing}; only the target may be None")
        keys = cast(Mapping[Target, LongTensor], raw)

        # index tensors are left-aligned; pad them so that torch's right-aligned broadcasting agrees
        self.batch_ndim = max(index.ndim for index in keys.values())
        padded = {label: _pad_trailing_dims(index, ndim=self.batch_ndim) for label, index in keys.items()}
        for label, index in padded.items():
            setattr(self, label, index)
        self.batch_shape = broadcast_index_shapes(index.shape for index in padded.values())

        ids = self.target_ids
        if ids is not None and ids.ndim not in (1, self.batch_ndim + 1):
            raise ValueError(
                f"The target IDs for {self.target} must have shape (num,) or (*batch_shape, num) with "
                f"batch_shape={self.batch_shape}, but have shape {tuple(ids.shape)}"
            )

    @property
    def indices(self) -> tuple[LongTensor | None, LongTensor | None, LongTensor | None]:
        """Return the index tensors, in the order ``(head, relation, tail)``."""
        return self.head, self.relation, self.tail

    @property
    def target_index(self) -> int:
        """Return the column index of the target.

        :raises ValueError:
            if there is no target

        :return:
            the index of the target into ``(head, relation, tail)``
        """
        if self.target is None:
            raise ValueError("There is no target.")
        return COLUMN_LABELS.index(self.target)

    @property
    def target_ids(self) -> LongTensor | None:
        """Return the target's index tensor, or None if scoring against all candidates."""
        if self.target is None:
            return None
        return self.indices[self.target_index]

    @property
    def shared_target(self) -> bool:
        """Return whether the same candidates are scored for each batch element."""
        ids = self.target_ids
        return ids is None or ids.ndim == 1

    @property
    def device(self) -> torch.device:
        """Return the device of the index tensors."""
        return next(index.device for index in self.indices if index is not None)

    @property
    def lookup_indices(self) -> tuple[LongTensor | None, LongTensor | None, LongTensor | None]:
        """Return the index tensors to look up representations with.

        For 1:n scoring, the non-target index tensors receive an additional singleton dimension, so that the
        looked-up representations broadcast against the target's candidate dimension.

        :return:
            the head, relation, and tail index tensors
        """
        if self.target is None:
            return self.indices
        # only the target may be None, cf. __post_init__
        return cast(
            tuple[LongTensor | None, LongTensor | None, LongTensor | None],
            tuple(
                index if label == self.target else cast(LongTensor, index).unsqueeze(dim=self.batch_ndim)
                for label, index in zip(COLUMN_LABELS, self.indices, strict=True)
            ),
        )

    def with_target_ids(self, ids: LongTensor) -> ScoringBatch:
        """Return a copy of this batch with the target's index tensor replaced.

        :param ids:
            the new target index tensor

        :raises ValueError:
            if there is no target

        :return:
            the new batch
        """
        if self.target is None:
            raise ValueError("There is no target.")
        return dataclasses.replace(self, **{self.target: ids})
