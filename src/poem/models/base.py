# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

from typing import Optional, Tuple

import torch
from torch import nn
import numpy as np
import random
import logging

from torch._C import device
from abc import abstractmethod
from operator import attrgetter

from ..constants import EMBEDDING_DIM, GPU, CPU

__all__ = [
    'BaseModule',
]

log = logging.getLogger(__name__)

class BaseModule(nn.Module):
    """A base class for all of the OWA based models."""

    entity_embedding_max_norm: Optional[int] = None
    entity_embedding_norm_type: int = 2
    hyper_params: Tuple[str] = [EMBEDDING_DIM]

    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int = 50,
                 criterion: nn.modules.loss = nn.MarginRankingLoss(),
                 preferred_device: str = GPU,
                 random_seed: Optional[int] = None) -> None:
        super().__init__()

        # Initialize the device
        self._set_device(preferred_device)

        self.random_seed = random_seed

        # Random seeds have to set before the embeddings are initialized
        if self.random_seed is not None:
            np.random.seed(seed=self.random_seed)
            torch.manual_seed(seed=self.random_seed)
            random.seed(self.random_seed)

        # Loss
        self.criterion = criterion
        # TODO: Check loss functions that require 1 and -1 as label but only
        self.compute_mr_loss = isinstance(criterion, nn.MarginRankingLoss)

        # Entity dimensions
        #: The number of entities in the knowledge graph
        self.num_entities = num_entities
        #: The number of unique relation types in the knowledge graph
        self.num_relations = num_relations
        #: The dimension of the embeddings to generate
        self.embedding_dim = embedding_dim

        # The embeddings are first initiated when calling the fit function
        self.entity_embeddings = None


    def _init_embeddings(self):
        self.entity_embeddings = nn.Embedding(
            self.num_entities,
            self.embedding_dim,
            max_norm=self.entity_embedding_max_norm,
            norm_type=self.entity_embedding_norm_type,
        )

    def _set_device(self,
                    device: str = 'cpu',
                    ) -> None:
        """Get the Torch device to use."""
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
                log.info('No cuda devices were available. The model runs on CPU')
        else:
            self.device = torch.device('cpu')

    def _to_cpu(self):
        """Transfer the entire model to CPU"""
        self._set_device('cpu')
        self.to(self.device)
        torch.cuda.empty_cache()

    def _to_gpu(self):
        """Transfer the entire model to GPU"""
        self._set_device('gpu')
        self.to(self.device)
        torch.cuda.empty_cache()

    def predict_scores(self, triples):
        scores = self.forward_owa(triples)
        return scores.detach().cpu().numpy()

    @staticmethod
    def _get_embeddings(elements, embedding_module, embedding_dim):
        """"""
        return embedding_module(elements).view(-1, embedding_dim)

    # TODO: Why this one?
    def compute_probabilities(self, scores):
        """."""
        return self.sigmoid(scores)

    def compute_mr_loss(
            self,
            pos_triple_scores: torch.Tensor,
            neg_triples_scores: torch.Tensor,
    ) -> torch.Tensor:
        assert self.compute_mr_loss, 'The chosen criterion does not allow the calculation of Margin Ranking losses. ' \
                                     'Please use the compute_label_loss method instead'
        y = torch.ones_like(neg_triples_scores, device=self.device)
        loss = self.criterion(pos_triple_scores, neg_triples_scores, y)
        return loss

    def compute_label_loss(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """."""
        assert self.compute_mr_loss == False,\
            'The chosen criterion does not allow the calculation of label losses. Please use the' \
            'compute_mr_loss method instead'
        loss = self.criterion(predictions, labels)
        return loss

    @abstractmethod
    def forward(self, batch):
        pass

    def get_grad_params(self):
        self._init_embeddings()
        return filter(attrgetter('requires_grad'), self.parameters())
