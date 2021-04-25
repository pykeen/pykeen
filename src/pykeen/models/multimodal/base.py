# -*- coding: utf-8 -*-

"""Base classes for multi-modal models."""

from typing import Sequence, Union

from ..nbase import ERModel, EmbeddingSpecificationHint
from ...nn.emb import EmbeddingSpecification, LiteralRepresentation, RepresentationModule
from ...nn.modules import LiteralInteraction
from ...triples import TriplesNumericLiteralsFactory
from ...typing import HeadRepresentation, RelationRepresentation, TailRepresentation

__all__ = [
    'LiteralModel',
]


class LiteralModel(ERModel[HeadRepresentation, RelationRepresentation, TailRepresentation], autoreset=False):
    """Base class for models with entity literals that uses combinations from :class:`pykeen.nn.combinations`."""

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        interaction: LiteralInteraction,
        entity_representations: Sequence[Union[EmbeddingSpecification, RepresentationModule]],
        relation_representations: EmbeddingSpecificationHint = None,
        **kwargs,
    ):
        literal_representation = LiteralRepresentation(
            numeric_literals=triples_factory.get_numeric_literals_tensor(),
        )
        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            entity_representations=[*entity_representations, literal_representation],
            relation_representations=relation_representations,
            **kwargs,
        )
