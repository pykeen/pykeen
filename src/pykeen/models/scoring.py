"""WIP: Fast Scoring."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from typing_extensions import Self

from pykeen import typing as pykeen_typing
from pykeen.inverse import RelationInverter
from pykeen.models import ERModel
from pykeen.nn.modules import parallel_unsqueeze
from pykeen.training.lcwa import LCWABatch
from pykeen.training.slcwa import SLCWABatch
from pykeen.typing import (
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
    FloatTensor,
    InductiveMode,
    LongTensor,
    OneOrSequence,
    Target,
)
from pykeen.utils import upgrade_to_sequence

__all__ = [
    "Batch",
    "Scorer",
]


def guarantee_broadcastable(shapes: Iterable[tuple[int, ...]]) -> None:
    """Raise an error if the shapes are not broadcastable."""
    # dimensions either have to agree, or be 1
    for i, dims in enumerate(zip(*shapes, strict=True)):
        if len(set(dims).difference({1})) > 1:
            raise ValueError(f"Cannot broadcast shapes {shapes=} because of {dims=} at {i=}")


@dataclasses.dataclass
class Batch:
    """A batch for fast scoring."""

    head: LongTensor | None
    relation: LongTensor | None
    tail: LongTensor | None

    use_inverse_relation: bool = False

    # inferred fields
    index_ndim: int = dataclasses.field(init=False)
    all_target: Target | None = dataclasses.field(init=False, default=None)

    @staticmethod
    def maybe_add_trailing_dims(x: LongTensor | None, max_ndim: int) -> LongTensor | None:
        """Fill up trailing dimensions."""
        if x is None:
            return x
        missing = max_ndim - x.ndim
        if not missing:
            return x
        return x.view(*x.shape, *itertools.repeat(1, times=missing))

    @classmethod
    def from_lcwa(cls, x: LCWABatch, target: Target) -> Self:
        """Create LCWA batch."""
        pairs = x["pairs"]
        a, b = pairs.unbind(dim=-1)
        match target:
            case pykeen_typing.LABEL_HEAD:
                return cls(head=None, relation=a, tail=b)
            case pykeen_typing.LABEL_RELATION:
                return cls(head=a, relation=None, tail=b)
            case pykeen_typing.LABEL_TAIL:
                return cls(head=a, relation=b, tail=None)
        raise NotImplementedError(target)

    @classmethod
    def from_slcwa(cls, x: SLCWABatch) -> Self:
        """Create sLCWA batch."""
        # TODO: we cannot easily exploit structure in the negative samples, e.g., shared head/tail
        #: the positive triples, shape: (batch_size, 3)
        pos = x["positives"]
        #: the negative triples, shape: (batch_size, num_negatives_per_positive, 3)
        neg = x["negatives"]
        combined = torch.cat([pos[:, None, :], neg], dim=1)
        return cls(head=combined[:, 0], relation=combined[:, 1], tail=combined[:, 2])

    def __post_init__(self) -> None:
        max_ndim = 0
        for target, t_indices in {
            LABEL_HEAD: self.head,
            LABEL_RELATION: self.relation,
            LABEL_TAIL: self.tail,
        }.items():
            if t_indices is None:
                if self.all_target:
                    raise ValueError(f"Two all targets: {self.all_target} and {target}")
                self.all_target = target
            else:
                max_ndim = max(max_ndim, t_indices.ndim)
        self.head = self.maybe_add_trailing_dims(self.head, max_ndim=max_ndim)
        self.relation = self.maybe_add_trailing_dims(self.relation, max_ndim=max_ndim)
        self.tail = self.maybe_add_trailing_dims(self.tail, max_ndim=max_ndim)
        guarantee_broadcastable(indices.shape for indices in self.indices if indices is not None)
        self.index_ndim = max_ndim

    def __getitem__(self, index: Target) -> LongTensor | None:
        match index:
            case pykeen_typing.LABEL_HEAD:
                return self.head
            case pykeen_typing.LABEL_RELATION:
                return self.relation
            case pykeen_typing.LABEL_TAIL:
                return self.tail
        raise KeyError(index)

    @property
    def indices(self) -> tuple[LongTensor | None, LongTensor | None, LongTensor | None]:
        """Return the indices."""
        return self.head, self.relation, self.tail

    def via_inverse(self, inverter: RelationInverter) -> Batch:
        """Create a batch for scoring via the inverse relation."""
        if self.use_inverse_relation:
            return self
        return Batch(
            head=self.tail,
            relation=inverter.get_inverse_id(self.relation),
            tail=self.head,
            use_inverse_relation=True,
        )

    @staticmethod
    def _iter_slice_indices(slice_size: int, num: int) -> Iterator[LongTensor]:
        """Iterate over indices for slices."""
        for start in range(0, num, slice_size):
            yield torch.arange(start=start, end=min(start + slice_size, num))

    def slice(self, slice_size: int, num: int) -> Iterator[Batch]:
        """Iterate over slices."""
        match self.all_target:
            case None:
                raise ValueError(
                    "Cannot slice, because there are only batch dimensions. Look into subbatching instead."
                )
            case pykeen_typing.LABEL_HEAD:
                for indices in self._iter_slice_indices(slice_size=slice_size, num=num):
                    yield Batch(
                        head=indices,
                        relation=self.relation,
                        tail=self.tail,
                        use_inverse_relation=self.use_inverse_relation,
                    )
            case pykeen_typing.LABEL_RELATION:
                for indices in self._iter_slice_indices(slice_size=slice_size, num=num):
                    yield Batch(
                        head=self.head,
                        relation=indices,
                        tail=self.tail,
                        use_inverse_relation=self.use_inverse_relation,
                    )
            case pykeen_typing.LABEL_TAIL:
                for indices in self._iter_slice_indices(slice_size=slice_size, num=num):
                    yield Batch(
                        head=self.head,
                        relation=self.relation,
                        tail=indices,
                        use_inverse_relation=self.use_inverse_relation,
                    )
        raise AssertionError


def parallel_prefix_unsqueeze(x: OneOrSequence[FloatTensor], ndim: int) -> OneOrSequence[FloatTensor]:
    """Unsqueeze all representations along the given dimension."""
    if not ndim:
        return x
    xs = upgrade_to_sequence(x)
    ones = (1,) * ndim
    xs = [xx.view(*ones, *x.shape) for xx in xs]
    return xs[0] if len(xs) == 1 else xs


# TODO: extract inverse relations scorer as an adapter?


@dataclasses.dataclass
class Scorer:
    """Calculate scores."""

    predict_with_sigmoid: bool = False

    @staticmethod
    def unsqueeze(
        r: OneOrSequence[FloatTensor], indices: LongTensor | None, index_ndim: int
    ) -> OneOrSequence[FloatTensor]:
        """Unsqueeze if necessary."""
        if indices is None:
            return parallel_prefix_unsqueeze(r, ndim=index_ndim)
        return parallel_unsqueeze(r, dim=index_ndim)

    def score(
        self,
        model: ERModel,
        batch: Batch,
        slice_size: int | None = None,
        mode: InductiveMode | None = None,
    ) -> FloatTensor:
        """Calculate scores."""
        if batch.use_inverse_relation and not model.use_inverse_triples:
            raise ValueError

        nums: tuple[()] | tuple[int]
        if batch.all_target is None:
            nums = tuple()
        elif batch.all_target == LABEL_RELATION:
            nums = (model.num_relations,)
        else:
            nums = (model.num_entities,)

        if slice_size:
            if not nums:
                raise ValueError
            return torch.cat(
                [
                    self.score(model=model, batch=batch_slice, slice_size=None, mode=mode)
                    for batch_slice in batch.slice(slice_size=slice_size, num=nums[0])
                ],
                dim=batch.index_ndim,
            )
        # get representations; if None, shape=(num_{entities,relations}, *r.shape), else shape=(*index.shape, *r.shape)
        h, r, t = model._get_representations(h=batch.head, r=batch.relation, t=batch.tail, mode=mode)
        if batch.all_target is not None:
            h, r, t = (
                self.unsqueeze(x, indices=i, index_ndim=batch.index_ndim)
                for x, i in zip((h, r, t), batch.indices, strict=True)
            )

        scores = model.interaction(h=h, r=r, t=t)
        # expected shape: *index.shape, num?
        expected_shape = tuple(torch.broadcast_shapes(*(i.shape for i in batch.indices if i is not None))) + nums
        # repeat if necessary
        return scores.expand(*expected_shape)

    def predict(self, model: ERModel, batch: Batch, **kwargs: Any) -> FloatTensor:
        """Predict."""
        # todo: auto switch to inverse relations?
        model.eval()
        scores = self.score(model, batch, **kwargs)
        if self.predict_with_sigmoid:
            scores = scores.sigmoid()
        return scores
