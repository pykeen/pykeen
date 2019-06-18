# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

from typing import Optional

import torch
from torch import nn

from ..constants import EMBEDDING_DIM, GPU, OWA

__all__ = [
    'BaseOWAModule',
]


class BaseOWAModule(nn.Module):
    """A base class for all of the OWA based models."""

    entity_embedding_max_norm: Optional[int] = None
    entity_embedding_norm_type: int = 2
    kg_assumption = OWA
    hyper_params = [EMBEDDING_DIM]

    def __init__(self, num_entities, num_relations, criterion, embedding_dim=50, preferred_device=GPU) -> None:
        super().__init__()

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and preferred_device == GPU else 'cpu')

        # Loss
        self.criterion = criterion
        self.compute_loss_fct = self._compute_mr_loss if type(
            criterion) == nn.MarginRankingLoss else self._compute_label_loss
        self.sigmoid = nn.Sigmoid()

        # Entity dimensions
        #: The number of entities in the knowledge graph
        self.num_entities = num_entities
        #: The number of unique relation types in the knowledge graph
        self.num_relations = num_relations
        #: The dimension of the embeddings to generate
        self.embedding_dim = embedding_dim

        self.entity_embeddings = nn.Embedding(
            self.num_entities,
            self.embedding_dim,
            norm_type=self.entity_embedding_norm_type,
            max_norm=self.entity_embedding_max_norm,
        )

    def predict_scores(self, triples):
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def _get_embeddings(self, elements, embedding_module, embedding_dim):
        """"""
        return embedding_module(elements).view(-1, embedding_dim)

    def apply_forward_constraints(self):
        """"""
        return

    def _compute_label_loss(self, pos_elements, neg_elements):
        """."""

        pos_labels = torch.FloatTensor([1])
        pos_labels = pos_labels.expand(pos_elements.shape[0]).to(self.device)

        neg_labels = torch.FloatTensor([self.neg_label])
        neg_labels = neg_labels.expand(neg_elements.shape[0]).to(self.device)

        scores = torch.cat([pos_elements, neg_elements])
        labels = torch.cat([pos_labels, neg_labels])

        loss = self.criterion(scores, labels)

        return loss

    def compute_probabilities(self, scores):
        """."""
        return self.sigmoid(scores)

    def compute_loss(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        """"""
        loss = self.compute_loss_fct(positive_scores, negative_scores)
        return loss

    def _compute_mr_loss(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        """"""
        y = torch.FloatTensor([1])

        y = y.expand(positive_scores.shape[0]).to(self.device)

        loss = self.criterion(positive_scores, negative_scores, y)

        return loss

    def forward(self, batch):
        scores = self._score_triples(batch)
        return scores


def slice_triples(triples):
    """Get the heads, relations, and tails from a matrix of triples."""
    h = triples[:, 0:1]
    r = triples[:, 1:2]
    t = triples[:, 2:3]
    return h, r, t
