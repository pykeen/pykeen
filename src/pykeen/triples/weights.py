"""Sample weights."""

import abc

import torch
from class_resolver import ClassResolver
from typing_extensions import Self

from ..typing import COLUMN_RELATION, FloatTensor, LongTensor, MappedTriples

__all__ = [
    "SampleWeighter",
    "RelationSampleWeighter",
    "sample_weighter_resolver",
]


class SampleWeighter(abc.ABC):
    """Determine sample weights."""

    @abc.abstractmethod
    def __call__(self, h: LongTensor, r: LongTensor, t: LongTensor) -> FloatTensor:
        """
        Calculate the sample weights for the given triples.

        Does support broadcasting semantics.

        :param h:
            The head indices.
        :param r:
            The relation indices.
        :param t:
            The tail indices.

        :return:
            The sample weights.
        """
        raise NotImplementedError


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
    def __call__(self, h: LongTensor, r: LongTensor, t: LongTensor) -> FloatTensor:
        return self.weights[r]

    @classmethod
    def inverse_relation_frequency(cls, mapped_triples: MappedTriples) -> Self:
        """Create a relation weighter with inverse relation frequencies."""
        uniq, counts = mapped_triples[:, COLUMN_RELATION].unique(return_counts=True)
        if not torch.equal(uniq, torch.arange(len(uniq))):
            raise ValueError(f"Non contiguous relation ids: {uniq=}")
        return cls(weights=torch.reciprocal(counts))


sample_weighter_resolver = ClassResolver.from_subclasses(base=SampleWeighter)
