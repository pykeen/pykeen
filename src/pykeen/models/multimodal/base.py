# -*- coding: utf-8 -*-

"""Base classes for multi-modal models."""

from class_resolver.utils import OneOrManyHintOrType, OneOrManyOptionalKwargs

from ..nbase import ERModel
from ...nn.init import PretrainedInitializer
from ...nn.modules import LiteralInteraction
from ...nn.representation import Embedding, Representation
from ...triples import TriplesNumericLiteralsFactory
from ...typing import HeadRepresentation, RelationRepresentation, TailRepresentation
from ...utils import upgrade_to_sequence

__all__ = [
    "LiteralModel",
]


class LiteralModel(ERModel[HeadRepresentation, RelationRepresentation, TailRepresentation], autoreset=False):
    """Base class for models with entity literals that uses combinations from :class:`pykeen.nn.combinations`."""

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        interaction: LiteralInteraction,
        entity_representations: OneOrManyHintOrType[Representation] = None,
        entity_representations_kwargs: OneOrManyOptionalKwargs = None,
        **kwargs,
    ):
        """
        Initialize the model.

        :param triples_factory:
            the (training) triples factory
        :param interaction:
            the interaction function
        :param entity_representations:
            the entity representations (excluding the ones from literals)
        :param entity_representations_kwargs:
            the entity representations keyword-based parameters (excluding the ones from literals)
        :param kwargs:
            additional keyword-based parameters passed to :meth:`ERModel.__init__`
        """
        literals = triples_factory.get_numeric_literals_tensor()
        _max_id, *shape = literals.shape
        entity_representations = tuple(upgrade_to_sequence(entity_representations)) + (Embedding,)
        entity_representations_kwargs = tuple(upgrade_to_sequence(entity_representations_kwargs)) + (
            dict(
                # max_id=max_id,  # will be added by ERModel
                shape=shape,
                initializer=PretrainedInitializer(tensor=literals),
                trainable=False,
            ),
        )
        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            entity_representations=entity_representations,
            entity_representations_kwargs=entity_representations_kwargs,
            **kwargs,
        )
