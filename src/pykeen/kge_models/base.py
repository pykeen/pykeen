# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import nn

from pykeen.constants import EMBEDDING_DIM, GPU, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS, PREFERRED_DEVICE

__all__ = [
    'BaseModule',
    'BaseConfig',
]


@dataclass
class BaseConfig:
    """Configuration for KEEN models."""

    try_gpu: bool
    margin_loss: str
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


class BaseModule(nn.Module):
    """A base class for all of the models."""

    margin_ranking_loss_size_average: bool = ...
    entity_embedding_max_norm: Optional[int] = None
    entity_embedding_norm_type: int = 2

    def __init__(self, config: Union[Dict, BaseConfig]) -> None:
        super().__init__()

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
        self.num_entities = config.number_entities
        self.num_relations = config.number_relations
        self.embedding_dim = config.embedding_dimension

        self.entity_embeddings = nn.Embedding(
            self.num_entities,
            self.embedding_dim,
            norm_type=self.entity_embedding_norm_type,
            max_norm=self.entity_embedding_max_norm,
        )

    @property
    def bound(self):
        return 6 / np.sqrt(self.embedding_dim)

    def __init_subclass__(cls, **kwargs):
        if not getattr(cls, 'model_name', None):
            raise TypeError('missing model_name class attribute')
