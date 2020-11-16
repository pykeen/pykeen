# -*- coding: utf-8 -*-

"""Implementation of the DistMultLiteral model."""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from .. import ERModel
from ...losses import Loss
from ...nn import Embedding, Interaction
from ...nn.emb import EmbeddingSpecification
from ...nn.modules import DistMultInteraction
from ...regularizers import Regularizer
from ...triples import TriplesNumericLiteralsFactory
from ...typing import DeviceHint, HeadRepresentation, RelationRepresentation, TailRepresentation

__all__ = [
    'DistMultLiteral',
]


class LiteralRepresentations(Embedding):
    """Literal representations."""

    def __init__(
        self,
        numeric_literals: torch.FloatTensor,
    ):
        num_embeddings, embedding_dim = numeric_literals.shape
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            initializer=lambda x: numeric_literals,  # initialize with the literals
        )
        # freeze
        self._embeddings.requires_grad_(False)


class LiteralModel(ERModel):
    """Base class for models with entity literals."""

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        embedding_dim: int,
        interaction: Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation],
        dropout: float = 0.0,
        entity_specification: Optional[EmbeddingSpecification] = None,
        relation_specification: Optional[EmbeddingSpecification] = None,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        automatic_memory_optimization: Optional[bool] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ):
        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            automatic_memory_optimization=automatic_memory_optimization,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            entity_representations=[
                # entity embeddings
                Embedding.from_specification(
                    num_embeddings=triples_factory.num_entities,
                    embedding_dim=embedding_dim,
                    specification=entity_specification,
                ),
                # Entity literals
                LiteralRepresentations(
                    numeric_literals=torch.as_tensor(triples_factory.numeric_literals, dtype=torch.float32),
                ),
            ],
            relation_representations=Embedding.from_specification(
                num_embeddings=triples_factory.num_relations,
                embedding_dim=embedding_dim,
                specification=relation_specification,
            ),
        )
        self.trans = nn.Sequential(
            nn.Linear(embedding_dim + triples_factory.numeric_literals.shape[1], embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
    ) -> torch.FloatTensor:  # noqa: D102
        h, r, t = self._get_representations(h_indices, r_indices, t_indices)
        # combine entity embeddings + literals
        h, t = [
            self.trans(torch.cat(x, dim=-1))
            for x in (h, t)
        ]
        scores = self.interaction(h=h, r=r, t=t)
        return self._repeat_scores_if_necessary(scores, h_indices, r_indices, t_indices)


# TODO: Check entire build of the model
# TODO: There are no tests
class DistMultLiteral(LiteralModel):
    """An implementation of DistMultLiteral from [agustinus2018]_."""

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
        input_dropout=dict(type=float, low=0, high=1.0),
    )
    #: The default parameters for the default loss function class
    loss_default_kwargs = dict(margin=0.0)

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        input_dropout: float = 0.0,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        predict_with_sigmoid: bool = False,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            interaction=DistMultInteraction(),
            dropout=input_dropout,
            entity_specification=EmbeddingSpecification(
                initializer=xavier_normal_,
            ),
            relation_specification=EmbeddingSpecification(
                initializer=xavier_normal_,
            ),
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            automatic_memory_optimization=automatic_memory_optimization,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
