# -*- coding: utf-8 -*-

"""Implementation of ERMLP."""

from typing import Dict

import torch
import torch.autograd
from torch import nn

from pykeen.constants import ERMLP_NAME
from pykeen.kge_models.base import BaseModule, slice_triples

__all__ = ['ERMLP']


class ERMLP(BaseModule):
    """An implementation of ERMLP [dong2014]_.

    This model uses a neural network-based approach.

    .. [dong2014] Dong, X., *et al.* (2014) `Knowledge vault: A web-scale approach to probabilistic knowledge fusion
                  <https://dl.acm.org/citation.cfm?id=2623623>`_. ACM.
    """

    model_name = ERMLP_NAME
    margin_ranking_loss_size_average: bool = True

    def __init__(self, config: Dict) -> None:
        super().__init__(config)

        #: Embeddings for relations in the knowledge graph
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        """The mulit layer perceptron consisting of an input layer with 3 * self.embedding_dim neurons, a  hidden layer
           with self.embedding_dim neurons and output layer with one neuron.
           The input is represented by the concatenation embeddings of the heads, relations and tail embeddings.
        """
        self.mlp = nn.Sequential(
            nn.Linear(3 * self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1),
        )

    def predict(self, triples):
        # triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def forward(self, positives, negatives):
        positive_scores = self._score_triples(positives)
        negative_scores = self._score_triples(negatives)
        loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss

    def _score_triples(self, triples):
        head_embeddings, relation_embeddings, tail_embeddings = self._get_triple_embeddings(triples)
        scores = self._compute_scores(head_embeddings, relation_embeddings, tail_embeddings)
        return scores

    def _compute_scores(self, head_embeddings, relation_embeddings, tail_embeddings):
        x_s = torch.cat([head_embeddings, relation_embeddings, tail_embeddings], 1)
        scores = - self.mlp(x_s)
        return scores

    def _get_triple_embeddings(self, triples):
        heads, relations, tails = slice_triples(triples)
        return (
            self._get_entity_embeddings(heads),
            self._get_relation_embeddings(relations),
            self._get_entity_embeddings(tails),
        )

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.embedding_dim)
