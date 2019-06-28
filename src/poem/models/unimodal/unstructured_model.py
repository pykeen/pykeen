# -*- coding: utf-8 -*-

"""Implementation of UM."""

import logging

import numpy as np
import torch
import torch.autograd
from torch import nn

from poem.constants import GPU, SCORING_FUNCTION_NORM, UM_NAME
from poem.models.base_owa import BaseOWAModule, slice_triples

__all__ = ['UnstructuredModel']

log = logging.getLogger(__name__)


class UnstructuredModel(BaseOWAModule):
    """An implementation of Unstructured Model (UM) [bordes2014]_.

    .. [bordes2014] Bordes, A., *et al.* (2014). `A semantic matching energy function for learning with
                    multi-relational data <https://link.springer.com/content/pdf/10.1007%2Fs10994-013-5363-6.pdf>`_.
                    Machine
    """

    model_name = UM_NAME
    margin_ranking_loss_size_average: bool = True
    hyper_params = BaseOWAModule.hyper_params + [SCORING_FUNCTION_NORM]

    def __init__(self, num_entities, num_relations, embedding_dim=50, scoring_fct_norm=1,
                 criterion=nn.MarginRankingLoss(margin=1., reduction='mean'), preferred_device=GPU) -> None:
        super(UnstructuredModel, self).__init__(num_entities, num_relations, criterion, embedding_dim, preferred_device)

        self.scoring_fct_norm = scoring_fct_norm

        self._initialize()

    def _initialize(self):
        entity_embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-entity_embeddings_init_bound,
            b=entity_embeddings_init_bound,
        )

    def predict_scores(self, triples):
        # triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def _score_triples(self, triples):
        head_embeddings, tail_embeddings = self._get_triple_embeddings(triples)
        scores = self._compute_scores(head_embeddings=head_embeddings, tail_embeddings=tail_embeddings)
        return scores

    def _compute_scores(self, head_embeddings, tail_embeddings):
        # Add the vector element wise
        sum_res = head_embeddings - tail_embeddings
        scores = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        scores = scores ** 2
        return -scores

    def _get_triple_embeddings(self, triples):
        heads, _, tails = slice_triples(triples)
        return (
            self._get_entity_embeddings(heads),
            self._get_entity_embeddings(tails),
        )

    def _get_entity_embeddings(self, entities):
        return self.entity_embeddings(entities).view(-1, self.embedding_dim)
