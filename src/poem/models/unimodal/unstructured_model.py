# -*- coding: utf-8 -*-

"""Implementation of UM."""

import logging
from typing import Optional

import numpy as np
import torch
import torch.autograd
from torch import nn

from poem.constants import GPU, SCORING_FUNCTION_NORM, UM_NAME
from poem.models.base import BaseModule
from poem.utils import slice_triples

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
    hyper_params = BaseModule.hyper_params + (SCORING_FUNCTION_NORM,)

    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int = 50,
            scoring_fct_norm: int = 1,
            criterion: nn.modules.loss = nn.MarginRankingLoss(margin=1., reduction='mean'),
            preferred_device: str = GPU,
            random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
        self.scoring_fct_norm = scoring_fct_norm

    def _init_embeddings(self):
        super()._init_embeddings()
        entity_embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-entity_embeddings_init_bound,
            b=entity_embeddings_init_bound,
        )

    def forward_owa(self, triples):
        head_embeddings, tail_embeddings = self._get_triple_embeddings(triples)
        # Add the vector element wise
        sum_res = head_embeddings - tail_embeddings
        scores = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        scores = scores ** 2
        return scores

    # TODO: Implement forward_cwa

    def _get_triple_embeddings(self, triples):
        heads, _, tails = slice_triples(triples)
        return (
            self._get_entity_embeddings(heads),
            self._get_entity_embeddings(tails),
        )

    def _get_entity_embeddings(self, entities):
        return self.entity_embeddings(entities).view(-1, self.embedding_dim)
