# -*- coding: utf-8 -*-

"""Implementation of UM."""

import logging
from typing import Optional

import numpy as np
import torch
import torch.autograd
from torch import nn

from poem.constants import SCORING_FUNCTION_NORM
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.utils import slice_triples
from ..base import BaseModule
from ...typing import OptionalLoss

__all__ = [
    'UnstructuredModel',
]

log = logging.getLogger(__name__)


class UnstructuredModel(BaseModule):
    """An implementation of Unstructured Model (UM) from [bordes2014]_."""

    margin_ranking_loss_size_average: bool = True
    hyper_params = BaseModule.hyper_params + (SCORING_FUNCTION_NORM,)

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 50,
            entity_embeddings: Optional[nn.Embedding] = None,
            scoring_fct_norm: int = 1,
            criterion: OptionalLoss = None,
            preferred_device: Optional[str] = None,
            random_seed: Optional[int] = None,
    ) -> None:
        if criterion is None:
            criterion = nn.MarginRankingLoss(margin=1., reduction='mean')

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
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
