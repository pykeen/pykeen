# -*- coding: utf-8 -*-

"""Implementation of DistMult."""

from typing import Dict

import numpy as np
import torch
import torch.autograd
from torch import nn

from pykeen.constants import DISTMULT_NAME
from pykeen.kge_models.base import BaseModule, slice_triples

__all__ = ['DistMult']


class DistMult(BaseModule):
    """An implementation of DistMult [yang2014]_.

    This model simplifies RESCAL by restricting matrices representing relations as diagonal matrices.

    .. [yang2014] Yang, B., Yih, W., He, X., Gao, J., & Deng, L. (2014). `Embedding Entities and Relations for Learning
                  and Inference in Knowledge Bases <https://arxiv.org/pdf/1412.6575.pdf>`_. CoRR, abs/1412.6575.

    .. seealso::

       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/master/models/DistMult.py
    """

    model_name = DISTMULT_NAME
    margin_ranking_loss_size_average: bool = True

    def __init__(self, config: Dict) -> None:
        super().__init__(config)

        # Embeddings
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        self._initialize()

    def _initialize(self):
        # The same bound is used for both entity embeddings and relation embeddings because they have the same dimension
        embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )

        norms = torch.norm(self.relation_embeddings.weight, p=2, dim=1).data
        self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight))

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
        return self._compute_scores(head_embeddings, relation_embeddings, tail_embeddings)

    @staticmethod
    def _compute_scores(head_embeddings, relation_embeddings, tail_embeddings):
        scores = - torch.sum(head_embeddings * relation_embeddings * tail_embeddings, dim=1)
        return scores

    def _get_triple_embeddings(self, triples):
        heads, relations, tails = slice_triples(triples)
        return (
            self._get_entity_embeddings(heads),
            self._get_relation_embeddings(relations),
            self._get_entity_embeddings(tails)
        )

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.embedding_dim)
