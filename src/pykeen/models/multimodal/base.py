# -*- coding: utf-8 -*-

"""Base classes for multi-modal models."""

from typing import Optional, Sequence, Union

from ..nbase import ERModel, EmbeddingSpecificationHint
from ...losses import Loss
from ...nn.emb import EmbeddingSpecification, LiteralRepresentation, RepresentationModule
from ...nn.modules import LiteralInteraction
from ...triples import TriplesNumericLiteralsFactory
from ...typing import DeviceHint, HeadRepresentation, RelationRepresentation, TailRepresentation

__all__ = [
    'LiteralModel',
]


class LiteralModel(ERModel[HeadRepresentation, RelationRepresentation, TailRepresentation], autoreset=False):
    """Base class for models with entity literals."""

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        interaction: LiteralInteraction,
        entity_representations: Sequence[Union[EmbeddingSpecification, RepresentationModule]],
        relation_representations: EmbeddingSpecificationHint = None,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
    ):
        literal_representation = LiteralRepresentation(
            numeric_literals=triples_factory.get_numeric_literals_tensor(),
        )

        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            preferred_device=preferred_device,
            random_seed=random_seed,
            entity_representations=[*entity_representations, literal_representation],
            relation_representations=relation_representations,
        )
