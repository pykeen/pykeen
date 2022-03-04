# -*- coding: utf-8 -*-

"""Base classes for multi-modal models."""

from typing import Optional

from class_resolver import HintOrType, OptionalKwargs

from ..nbase import ERModel
from ...nn.init import PretrainedInitializer
from ...nn.modules import LiteralInteraction
from ...nn.representation import Embedding, Representation
from ...triples import TriplesNumericLiteralsFactory
from ...typing import HeadRepresentation, OneOrSequence, RelationRepresentation, TailRepresentation
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
        entity_representations: Optional[OneOrSequence[HintOrType[Representation]]] = None,
        entity_representation_kwargs: Optional[OneOrSequence[OptionalKwargs]] = None,
        relation_representations: Optional[OneOrSequence[HintOrType[Representation]]] = None,
        relation_representation_kwargs: Optional[OneOrSequence[OptionalKwargs]] = None,
        **kwargs,
    ):
        literals = triples_factory.get_numeric_literals_tensor()
        max_id, *shape = literals.shape
        entity_representations = tuple(upgrade_to_sequence(entity_representations)) + (Embedding,)
        entity_representation_kwargs = tuple(upgrade_to_sequence(entity_representation_kwargs)) + (
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
            entity_representation_kwargs=entity_representation_kwargs,
            relation_representations=relation_representations,
            relation_representation_kwargs=relation_representation_kwargs,
            **kwargs,
        )
