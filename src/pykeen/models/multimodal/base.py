# -*- coding: utf-8 -*-

"""Base classes for multi-modal models."""

from typing import Optional, TYPE_CHECKING

import torch
from torch import nn

from ..base import ERModel
from ...losses import Loss
from ...nn import Embedding, EmbeddingSpecification, Interaction, LiteralRepresentations
from ...triples import TriplesNumericLiteralsFactory
from ...typing import DeviceHint, HeadRepresentation, RelationRepresentation, TailRepresentation

__all__ = [
    "LiteralModel",
]

if TYPE_CHECKING:
    from ...typing import Representation  # noqa


class LiteralModel(ERModel[HeadRepresentation, RelationRepresentation, TailRepresentation], autoreset=False):
    """Base class for models with entity literals."""

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        interaction: Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation],
        combination: nn.Module,
        entity_specification: Optional[EmbeddingSpecification] = None,
        relation_specification: Optional[EmbeddingSpecification] = None,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        automatic_memory_optimization: Optional[bool] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
    ):
        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            automatic_memory_optimization=automatic_memory_optimization,
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
        self.combination = combination

    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
        slice_size: Optional[int] = None,
        slice_dim: Optional[str] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        h, r, t = self._get_representations(h_indices, r_indices, t_indices)
        # combine entity embeddings + literals
        h, t = [
            self.combination(torch.cat(x, dim=-1))
            for x in (h, t)
        ]
        scores = self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=slice_dim)
        return self._repeat_scores_if_necessary(scores, h_indices, r_indices, t_indices)
