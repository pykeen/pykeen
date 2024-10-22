"""WIP: Fast Scoring."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterable

import torch
from torch import FloatTensor, LongTensor, broadcast_shapes

from pykeen.inverse import RelationInverter
from pykeen.models import ERModel
from pykeen.nn.modules import parallel_unsqueeze
from pykeen.typing import LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, OneOrSequence, Target
from pykeen.utils import upgrade_to_sequence

__all__ = [
    "Batch",
    "Scorer",
]


@dataclasses.dataclass
class Batch:
    head: LongTensor | None
    relation: LongTensor | None
    tail: LongTensor | None
    use_inverse_relation: bool = False
    index_ndim: int = dataclasses.field(init=False)
    all_target: Target | None = None

    @staticmethod
    def maybe_add_trailing_dims(x: LongTensor | None, max_ndim: int) -> LongTensor | None:
        if x is None:
            return x
        missing = max_ndim - x.ndim
        if not missing:
            return x
        return x.view(*x.shape, *itertools.repeat(1, times=missing))

    def __post_init__(self):
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
        self.index_ndim = max_ndim

    @property
    def indices(self) -> tuple[FloatTensor | None, FloatTensor | None, FloatTensor | None]:
        return self.head, self.relation, self.tail

    def via_inverse(self, inverter: RelationInverter) -> Batch:
        if self.use_inverse_relation:
            return self
        return Batch(
            head=self.tail,
            relation=inverter.get_inverse_id(self.relation),
            tail=self.head,
            use_inverse_relation=True,
        )

    def slice(self, slice_size: int, num: int) -> Iterable[Batch]:
        kwargs = dict(
            head=self.head,
            relation=self.relation,
            tail=self.tail,
            use_inverse_relation=self.use_inverse_relation,
        )
        if self.all_target is None:
            raise ValueError
        for start in range(0, num, slice_size):
            kwargs[self.all_target] = torch.arange(start=start, end=min(start + slice_size, num))
            yield Batch(**kwargs)


def parallel_prefix_unsqueeze(x: OneOrSequence[FloatTensor], ndim: int) -> OneOrSequence[FloatTensor]:
    """Unsqueeze all representations along the given dimension."""
    if not ndim:
        return x
    xs = upgrade_to_sequence(x)
    ones = (1,) * ndim
    xs = [xx.view(*ones, *x.shape) for xx in xs]
    return xs[0] if len(xs) == 1 else xs


class Scorer:
    predict_with_sigmoid: bool

    def __init__(self, predict_with_sigmoid: bool = False):
        self.predict_with_sigmoid = predict_with_sigmoid

    @staticmethod
    def unsqueeze(
        r: OneOrSequence[FloatTensor], indices: LongTensor | None, index_ndim: int
    ) -> OneOrSequence[FloatTensor]:
        if indices is None:
            return parallel_prefix_unsqueeze(r, ndim=index_ndim)
        return parallel_unsqueeze(r, dim=index_ndim)

    def score(self, model: ERModel, batch: Batch, slice_size: int | None = None, mode=None) -> FloatTensor:
        if batch.use_inverse_relation and not model.use_inverse_triples:
            raise ValueError

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
                self.unsqueeze(x, indices=i, index_ndim=batch.index_ndim) for x, i in zip((h, r, t), batch.indices)
            )

        scores = model.interaction(h=h, r=r, t=t)
        # expected shape: *index.shape, num?
        expected_shape = tuple(broadcast_shapes(*(i.shape for i in batch.indices if i is not None))) + nums
        # repeat if necessary
        return scores.expand(*expected_shape)

    def predict(self, model: ERModel, batch: Batch, **kwargs) -> FloatTensor:
        # todo: auto switch to inverse relations?
        model.eval()
        scores = self.score(model, batch, **kwargs)
        if self.predict_with_sigmoid:
            scores = scores.sigmoid()
        return scores
