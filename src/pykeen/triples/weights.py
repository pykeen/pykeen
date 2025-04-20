"""Loss weighting for triples.

Loss weights influence how much a given triple is weighted in the loss function. They are primarily a tool to shape your
optimization criterion, i.e., what you optimize your embedding models on. For example, you may want to focus more or
less on certain types of triples because they are more or less important for your application. You can also use loss
weights to counteract imbalances in relation frequencies, i.e., to down-weight frequent relation types.

They are not:

1. (static) weights for message passing - we already support those, cf. :class:`~pykeen.nn.weighting.EdgeWeighting`
2. weights in the sense of how reliable a particular triple is. If you have a source of uncertainty about triples, you
   might want to set lower loss weights for uncertain triples (in the sense of "it does not matter so much if the model
   reproduces an uncertain label"). Another alternative would be to use softer labels for them (i.e., try to predict the
   uncertain label directly), or more advanced ways of incorporating uncertainty.
3. more general qualifiers on the triples, e.g., ``(km, multiple_of, m)`` could have a qualifier ``(factor, 10)`` on it

The current implementation defines the general interface for (dynamic) loss weights via
:class:`~pykeen.triples.weights.LossWeighter`, which you can extend with your own weighting logic. It also provides an
implementation of relation-specific weighting, where the loss weight of a triple depends only on the relation type
involved: :class:`~pykeen.triples.weights.RelationLossWeighter`.

Example
-------
Below is a minimal example of how to use it via the high-level :func:`~pykeen.pipeline.api.pipeline` API:

.. literalinclude:: ../examples/training/loss_weights.py
"""

from abc import ABC, abstractmethod

import torch
from class_resolver import ClassResolver
from typing_extensions import Self

from ..typing import COLUMN_RELATION, FloatTensor, LongTensor, MappedTriples

__all__ = [
    "LossWeighter",
    "RelationLossWeighter",
    "loss_weighter_resolver",
]


class LossWeighter(ABC):
    """Determine loss weights for triples."""

    @abstractmethod
    def __call__(self, h: LongTensor | None, r: LongTensor | None, t: LongTensor | None) -> FloatTensor:
        """Calculate the sample weights for the given triples.

        Does support broadcasting semantics.

        :param h: The head indices, or None to denote all of them.
        :param r: The relation indices, or None to denote all of them.
        :param t: The tail indices, or None to denote all of them.

        :returns: The sample weights.
        """
        raise NotImplementedError

    def weight_triples(self, mapped_triples: MappedTriples) -> FloatTensor:
        """Calculate the sample weights for the given batch of triples.

        :param mapped_triples: shape: (..., 3) The ID-based triples.

        :returns: The sample weights.
        """
        h, r, t = mapped_triples.unbind(dim=-1)
        return self(h=h, r=r, t=t)


class RelationLossWeighter(LossWeighter):
    """Determine loss weights based solely on the relation."""

    def __init__(self, weights: FloatTensor):
        """Initialize the weighter.

        :param weights: shape: ``(num_relations,)`` The weight per relation.
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
        """Create a loss weighter with inverse relation frequencies."""
        uniq, counts = mapped_triples[:, COLUMN_RELATION].unique(return_counts=True)
        if not torch.equal(uniq, torch.arange(len(uniq))):
            raise ValueError(f"Non contiguous relation ids: {uniq=}")
        return cls(weights=torch.reciprocal(counts))


#: A resolver for loss weighters
loss_weighter_resolver: ClassResolver[LossWeighter] = ClassResolver.from_subclasses(base=LossWeighter)
