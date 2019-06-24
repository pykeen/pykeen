# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from .model_config import ModelConfig
from ..constants import EMBEDDING_DIM, GPU, LEARNING_RATE, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS, PREFERRED_DEVICE

__all__ = [
    'BaseCWAModule',
]


@dataclass
class BaseConfig:
    """Configuration for KEEN models."""

    try_gpu: bool
    margin_loss: float
    number_entities: int
    number_relations: int
    embedding_dimension: int

    def get_device(self):
        """Get the Torch device to use."""
        return torch.device('cuda:0' if torch.cuda.is_available() and self.try_gpu else 'cpu')

    @classmethod
    def from_dict(cls, config: Dict) -> 'BaseConfig':
        """Generate an instance from a dictionary."""
        return cls(
            try_gpu=(config.get(PREFERRED_DEVICE) == GPU),
            margin_loss=config[MARGIN_LOSS],
            number_entities=config[NUM_ENTITIES],
            number_relations=config[NUM_RELATIONS],
            embedding_dimension=config[EMBEDDING_DIM],
        )


class BaseCWAModule(nn.Module):
    """A base class for all of the models."""

    margin_ranking_loss_size_average: bool = ...
    entity_embedding_max_norm: Optional[int] = None
    entity_embedding_norm_type: int = 2
    hyper_params = [EMBEDDING_DIM, MARGIN_LOSS, LEARNING_RATE]

    def __init__(self, experimental_setup: ModelConfig) -> None:
        super().__init__()
        config = experimental_setup.config

        if not isinstance(config, BaseConfig):
            config = BaseConfig.from_dict(config)

        # Device selection
        self.device = config.get_device()

        # Loss
        self.margin_loss = config.margin_loss
        self.criterion = nn.MarginRankingLoss(
            margin=self.margin_loss,
            size_average=self.margin_ranking_loss_size_average,
        )

        # Entity dimensions
        #: The number of entities in the knowledge graph
        self.num_entities = config.number_entities
        #: The number of unique relation types in the knowledge graph
        self.num_relations = config.number_relations
        #: The dimension of the embeddings to generate
        self.embedding_dim = config.embedding_dimension

        self.entity_embeddings = nn.Embedding(
            self.num_entities,
            self.embedding_dim,
            norm_type=self.entity_embedding_norm_type,
            max_norm=self.entity_embedding_max_norm,
        )

    def __init_subclass__(cls, **kwargs):  # noqa: D105
        if not getattr(cls, 'model_name', None):
            raise TypeError('missing model_name class attribute')

    def _get_entity_embeddings(self, entities):
        return self.entity_embeddings(entities).view(-1, self.embedding_dim)
