# -*- coding: utf-8 -*-

"""Base classes for multi-modal models."""

import torch
from typing import Optional

from ..nbase import ERModel
from ...losses import Loss
from ...nn.emb import Embedding, EmbeddingSpecification, LiteralRepresentations
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
        entity_specification: Optional[EmbeddingSpecification] = None,
        relation_specification: Optional[EmbeddingSpecification] = None,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
    ):
        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            preferred_device=preferred_device,
            random_seed=random_seed,
            entity_representations=[
                # entity embeddings
                Embedding.from_specification(
                    num_embeddings=triples_factory.num_entities,
                    specification=entity_specification,
                ),
                # Entity literals
                LiteralRepresentations(
                    numeric_literals=torch.as_tensor(triples_factory.numeric_literals, dtype=torch.float32),
                ),
            ],
            relation_representations=Embedding.from_specification(
                num_embeddings=triples_factory.num_relations,
                specification=relation_specification,
            ),
        )
