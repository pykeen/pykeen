"""Sample weights."""

from abc import ABC, abstractmethod

import torch
from class_resolver import ClassResolver
from typing_extensions import Self

from ..typing import COLUMN_RELATION, FloatTensor, LongTensor, MappedTriples

__all__ = [
    "SampleWeighter",
    "RelationSampleWeighter",
    "sample_weighter_resolver",
]


class SampleWeighter(ABC):
    """Determine sample weights."""

    @abstractmethod
    def __call__(self, h: LongTensor | None, r: LongTensor | None, t: LongTensor | None) -> FloatTensor:
        """
        Calculate the sample weights for the given triples.

        Does support broadcasting semantics.

        :param h:
            The head indices, or None to denote all of them.
        :param r:
            The relation indices, or None to denote all of them.
        :param t:
            The tail indices, or None to denote all of them.

        :return:
            The sample weights.
        """
        raise NotImplementedError

    def weight_triples(self, mapped_triples: MappedTriples) -> FloatTensor:
        """Calculate the sample weights for the given batch of triples.

        :param mapped_triples: shape: (..., 3)
            The ID-based triples.

        :return:
            The sample weights.
        """
        h, r, t = mapped_triples.unbind(dim=-1)
        return self(h=h, r=r, t=t)


class RelationSampleWeighter(SampleWeighter):
    """Determine sample weights based solely on the relation."""

    def __init__(self, weights: FloatTensor):
        """Initialize the weighter.

        :param weights: shape: ``(num_relations,)``
            The weight per relation.
        """
        super().__init__()
        self.weights = weights

    # docstr-coverage: inherited
    def __call__(self, h: LongTensor | None, r: LongTensor | None, t: LongTensor | None) -> FloatTensor:
        if r is None:
            return self.weights
        return self.weights[r]

    @classmethod
    def inverse_relation_frequency(cls, mapped_triples: MappedTriples) -> Self:
        """Create a relation weighter with inverse relation frequencies."""
        uniq, counts = mapped_triples[:, COLUMN_RELATION].unique(return_counts=True)
        if not torch.equal(uniq, torch.arange(len(uniq))):
            raise ValueError(f"Non contiguous relation ids: {uniq=}")
        return cls(weights=torch.reciprocal(counts))


sample_weighter_resolver: ClassResolver[SampleWeighter] = ClassResolver.from_subclasses(base=SampleWeighter)
