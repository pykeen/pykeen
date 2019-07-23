# -*- coding: utf-8 -*-

"""Implementation of ERMLP."""

from typing import Optional

import torch
import torch.autograd
from torch import nn

from poem.instance_creation_factories.triples_factory import TriplesFactory
from ..base import BaseModule
from ...constants import ERMLP_NAME, GPU
from ...utils import slice_triples

__all__ = ['ERMLP']


class ERMLP(BaseModule):
    """An implementation of ERMLP [dong2014]_.

    This model uses a neural network-based approach.

    .. [dong2014] Dong, X., *et al.* (2014) `Knowledge vault: A web-scale approach to probabilistic knowledge fusion
                  <https://dl.acm.org/citation.cfm?id=2623623>`_. ACM.
    """

    model_name = ERMLP_NAME
    margin_ranking_loss_size_average: bool = True

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 50,
            entity_embeddings: nn.Embedding = None,
            relation_embeddings: nn.Embedding = None,
            criterion: nn.modules.loss = nn.MarginRankingLoss(margin=1., reduction='mean'),
            preferred_device: str = GPU,
            random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            triples_factory = triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings = entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        """The mulit layer perceptron consisting of an input layer with 3 * self.embedding_dim neurons, a  hidden layer
           with self.embedding_dim neurons and output layer with one neuron.
           The input is represented by the concatenation embeddings of the heads, relations and tail embeddings.
        """
        self.mlp = nn.Sequential(
            nn.Linear(3 * self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1),
        )

        self.relation_embeddings = relation_embeddings

        if None in [self.entity_embeddings,self.relation_embeddings]:
            self._init_embeddings()

    def _init_embeddings(self):
        super()._init_embeddings()
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

    def forward_owa(self, triples):
        head_embeddings, relation_embeddings, tail_embeddings = self._get_triple_embeddings(triples)
        x_s = torch.cat([head_embeddings, relation_embeddings, tail_embeddings], 1)
        scores = self.mlp(x_s)
        return scores

    # TODO: Implement forward_cwa

    def _get_triple_embeddings(self, triples):
        heads, relations, tails = slice_triples(triples)
        return (
            self._get_embeddings(
                elements=heads,
                embedding_module=self.entity_embeddings,
                embedding_dim=self.embedding_dim,
            ),
            self._get_embeddings(
                elements=relations,
                embedding_module=self.relation_embeddings,
                embedding_dim=self.embedding_dim,
            ),
            self._get_embeddings(
                elements=tails,
                embedding_module=self.entity_embeddings,
                embedding_dim=self.embedding_dim,
            ),
        )
