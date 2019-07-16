# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

import logging
import random
from abc import abstractmethod
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

from ..constants import EMBEDDING_DIM, GPU
from ..utils import get_params_requiring_grad

__all__ = [
    'BaseModule',
]

log = logging.getLogger(__name__)


class BaseModule(nn.Module):
    """A base class for all of the OWA based models."""

    entity_embedding_max_norm: Optional[int] = None
    entity_embedding_norm_type: int = 2
    hyper_params: Tuple[str] = (EMBEDDING_DIM,)

    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int = 50,
            criterion: nn.modules.loss._Loss = nn.MarginRankingLoss(),
            preferred_device: str = GPU,
            random_seed: Optional[int] = None,
    ) -> None:
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
        self.is_mr_loss = isinstance(criterion, nn.MarginRankingLoss)

        # Entity dimensions
        #: The number of entities in the knowledge graph
        self.num_entities = num_entities
        #: The number of unique relation types in the knowledge graph
        self.num_relations = num_relations
        #: The dimension of the embeddings to generate
        self.embedding_dim = embedding_dim

        # The embeddings are first initiated when calling the fit function
        self.entity_embeddings = None

        # Marker to check wether the forward constraints of a models has been applied before starting loss calculation
        self.forward_constraint_applied = False

    def _init_embeddings(self):
        self.entity_embeddings = nn.Embedding(
            self.num_entities,
            self.embedding_dim,
            max_norm=self.entity_embedding_max_norm,
            norm_type=self.entity_embedding_norm_type,
        )

    def _set_device(self, device: str = 'cpu') -> None:
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

    # Predicting scores calls the owa forward function, as this
    def predict_scores(self, triples):
        scores = self.forward_owa(triples)
        return scores.detach().cpu().numpy()

    @staticmethod
    def _get_embeddings(elements, embedding_module, embedding_dim):
        return embedding_module(elements).view(-1, embedding_dim)

    # FIXME this will be used in a later commit for the restructuring
    def compute_probabilities(self, scores):
        return self.sigmoid(scores)

    def compute_mr_loss(
            self,
            positive_scores: torch.Tensor,
            negative_scores: torch.Tensor,
    ) -> torch.Tensor:
        assert self.is_mr_loss, 'The chosen criterion does not allow the calculation of label losses. ' \
                                'Please use the compute_mr_loss method instead'
        y = torch.ones_like(negative_scores, device=self.device)
        loss = self.criterion(positive_scores, negative_scores, y)
        return loss

    def compute_label_loss(
            self,
            predictions: torch.Tensor,
            labels: torch.Tensor,
    ) -> torch.Tensor:
        assert not self.is_mr_loss, 'The chosen criterion does not allow the calculation of margin ranking losses. ' \
                                    'Please use the compute_label_loss method instead'
        loss = self.criterion(predictions, labels)
        return loss

    @abstractmethod
    def forward_owa(self, batch):
        raise NotImplementedError

    @abstractmethod
    def forward_cwa(self, batch):
        raise NotImplementedError

    # FIXME this isn't used anywhere
    def get_grad_params(self) -> Iterable[Parameter]:
        """Get the parameters that require gradients."""
        self._init_embeddings()
        return get_params_requiring_grad(self)
