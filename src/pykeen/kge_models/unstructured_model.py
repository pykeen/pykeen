# -*- coding: utf-8 -*-

"""Implementation of UM."""

import logging
from typing import Dict

import numpy as np
import torch
import torch.autograd
from torch import nn

from pykeen.constants import NORM_FOR_NORMALIZATION_OF_ENTITIES, SCORING_FUNCTION_NORM, UM_NAME
from pykeen.kge_models.base import BaseModule
from .trans_e import TransEConfig

__all__ = ['UnstructuredModel']

log = logging.getLogger(__name__)


class UnstructuredModel(BaseModule):
    """An implementation of Unstructured Model (UM) [bordes2014]_.

    .. [bordes2014] Bordes, A., *et al.* (2014). `A semantic matching energy function for learning with
                    multi-relational data <https://link.springer.com/content/pdf/10.1007%2Fs10994-013-5363-6.pdf>`_.
                    Machine
    """

    model_name = UM_NAME
    margin_ranking_loss_size_average: bool = True
    hyper_params = BaseModule.hyper_params + [SCORING_FUNCTION_NORM, NORM_FOR_NORMALIZATION_OF_ENTITIES]

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        config = TransEConfig.from_dict(config)

        self.l_p_norm_entities = config.lp_norm
        self.scoring_fct_norm = config.scoring_function_norm

        self._initialize()

    def _initialize(self):
        entity_embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-entity_embeddings_init_bound,
            b=entity_embeddings_init_bound,
        )

    def predict(self, triples):
        # triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def forward(self, batch_positives, batch_negatives):
        # Normalize embeddings of entities
        pos_scores = self._score_triples(batch_positives)
        neg_scores = self._score_triples(batch_negatives)
        loss = self._compute_loss(pos_scores=pos_scores, neg_scores=neg_scores)
        return loss

    def _compute_loss(self, pos_scores, neg_scores):
        y = np.repeat([-1], repeats=pos_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        # Scores for the psotive and negative triples
        pos_scores = torch.tensor(pos_scores, dtype=torch.float, device=self.device)
        neg_scores = torch.tensor(neg_scores, dtype=torch.float, device=self.device)
        # neg_scores_temp = 1 * torch.tensor(neg_scores, dtype=torch.float, device=self.device)

        loss = self.criterion(pos_scores, neg_scores, y)
        return loss

    def _score_triples(self, triples):
        head_embeddings, tail_embeddings = self._get_triple_embeddings(triples)
        scores = self._compute_scores(head_embeddings=head_embeddings, tail_embeddings=tail_embeddings)
        return scores

    def _compute_scores(self, head_embeddings, tail_embeddings):
        # Add the vector element wise
        sum_res = head_embeddings - tail_embeddings
        distances = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        distances = distances ** 2
        return distances

    def _get_triple_embeddings(self, triples):
        heads, tails = self.slice_triples(triples)
        return (
            self._get_entity_embeddings(heads),
            self._get_entity_embeddings(tails),
        )

    def _get_entity_embeddings(self, entities):
        return self.entity_embeddings(entities).view(-1, self.embedding_dim)

    @staticmethod
    def slice_triples(triples):
        return (
            triples[:, 0:1],
            triples[:, 2:3],
        )
