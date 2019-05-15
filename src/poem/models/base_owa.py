# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

from typing import Optional

import torch
from torch import nn

from ..constants import EMBEDDING_DIM, LEARNING_RATE, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS, OWA, PREFERRED_DEVICE
from ..model_config import ModelConfig

__all__ = [
    'BaseOWAModule',
]


class BaseOWAModule(nn.Module):
    """A base class for all of the models."""

    margin_ranking_loss_average: bool = ...
    entity_embedding_max_norm: Optional[int] = None
    entity_embedding_norm_type: int = 2
    hyper_params = [EMBEDDING_DIM, MARGIN_LOSS, LEARNING_RATE]
    kg_assumption = OWA

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()

        self.model_config = model_config
        self.config = self.model_config.config
        # Device selection
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and model_config.config(PREFERRED_DEVICE) else 'cpu')

        # Loss
        self.margin_loss = self.config[MARGIN_LOSS]
        self.criterion = nn.MarginRankingLoss(
            margin=self.margin_loss,
            reduction='mean' if self.margin_ranking_loss_average else None,
        )

        # Entity dimensions
        #: The number of entities in the knowledge graph
        self.num_entities = self.config[NUM_ENTITIES]
        #: The number of unique relation types in the knowledge graph
        self.num_relations = self.config[NUM_RELATIONS]
        #: The dimension of the embeddings to generate
        self.embedding_dim = self.config[EMBEDDING_DIM]

        self.entity_embeddings = nn.Embedding(
            self.num_entities,
            self.embedding_dim,
            norm_type=self.entity_embedding_norm_type,
            max_norm=self.entity_embedding_max_norm,
        )

    def __init_subclass__(cls, **kwargs):  # noqa: D105
        if not getattr(cls, 'model_name', None):
            raise TypeError('missing model_name class attribute')

    def _get_embeddings(self, elements, embedding_module, embedding_dim):
        return embedding_module(elements).view(-1, embedding_dim)
