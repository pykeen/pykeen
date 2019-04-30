# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from torch import nn

from poem.constants import PREFERRED_DEVICE, GPU, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM, \
    LEARNING_RATE, OWA


@dataclass
class BaseOWAConfig:
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
    def from_dict(cls, config: Dict) -> 'BaseOWAConfig':
        """Generate an instance from a dictionary."""
        return cls(
            try_gpu=(config.get(PREFERRED_DEVICE) == GPU),
            margin_loss=config[MARGIN_LOSS],
            number_entities=config[NUM_ENTITIES],
            number_relations=config[NUM_RELATIONS],
            embedding_dimension=config[EMBEDDING_DIM],
        )

class BaseOWAModule(nn.Module):
    """A base class for all of the models."""

    margin_ranking_loss_size_average: bool = ...
    entity_embedding_max_norm: Optional[int] = None
    entity_embedding_norm_type: int = 2
    hyper_params = [EMBEDDING_DIM, MARGIN_LOSS, LEARNING_RATE]
    kg_assumption = OWA

    def __init__(self, config: Union[Dict, BaseOWAConfig]) -> None:
        super().__init__()

        if not isinstance(config, BaseOWAConfig):
            config = BaseOWAConfig.from_dict(config)

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

    def _get_embeddings(self, elements, embedding_module, embedding_dim):
        return embedding_module(elements).view(-1, embedding_dim)


def slice_triples(triples):
    """Get the heads, relations, and tails from a matrix of triples."""
    h = triples[:, 0:1]
    r = triples[:, 1:2]
    t = triples[:, 2:3]
    return h, r, t