# -*- coding: utf-8 -*-

"""Implementation of the TransE model."""

import logging
from typing import Optional

import numpy as np
import torch
import torch.autograd
from torch import nn

from poem.models.base import BaseModule
from ...constants import GPU, TRANS_E_NAME, SCORING_FUNCTION_NORM
from ...utils import slice_triples

__all__ = [
    'TransE',
]

log = logging.getLogger(__name__)


class TransE(BaseModule):
    """An implementation of TransE [borders2013]_.

     This model considers a relation as a translation from the head to the tail entity.

    .. [borders2013] Bordes, A., *et al.* (2013). `Translating embeddings for modeling multi-relational data
                     <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_
                     . NIPS.

    .. seealso::

       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py
    """

    model_name = TRANS_E_NAME
    hyper_params = BaseModule.hyper_params + [SCORING_FUNCTION_NORM]

    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int = 50,
                 scoring_fct_norm: int = 1,
                 criterion: nn.modules.loss=nn.MarginRankingLoss(margin=1., reduction='mean'),
                 preferred_device: str = GPU,
                 random_seed: Optional[int] = None,
                 ) -> None:
        super().__init__(num_entities=num_entities, num_relations=num_relations, embedding_dim=embedding_dim,
                         criterion=criterion, preferred_device=preferred_device, random_seed=random_seed)
        self.scoring_fct_norm = scoring_fct_norm
        self.relation_embeddings = None

    def _init_embeddings(self):
        super()._init_embeddings()
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(self.entity_embeddings.weight.data,
                         a=-embeddings_init_bound,
                         b=+embeddings_init_bound,
                         )
        nn.init.uniform_(self.relation_embeddings.weight.data,
                         a=-embeddings_init_bound,
                         b=+embeddings_init_bound,
                         )

        norms = torch.norm(self.relation_embeddings.weight, p=2, dim=1).data
        self.relation_embeddings.weight.data =\
            self.relation_embeddings.weight.data.div(norms.view(self.num_relations, 1)
                                                     .expand_as(self.relation_embeddings.weight))

    def apply_forward_constraints(self):
        """."""
        norms = torch.norm(self.entity_embeddings.weight, p=2, dim=1).data
        self.entity_embeddings.weight.data =\
            self.entity_embeddings.weight.data.div(norms.view(self.num_entities, 1)
                                                   .expand_as(self.entity_embeddings.weight))

    def forward_owa(self, triples):
        head_embeddings, relation_embeddings, tail_embeddings = self._get_triple_embeddings(triples)
        # Add the vector element wise
        sum_res = head_embeddings + relation_embeddings - tail_embeddings
        scores = - torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        return scores

    # TODO: Implement forward_cwa

    def _get_triple_embeddings(self, triples):
        heads, relations, tails = slice_triples(triples)
        return (self._get_embeddings(elements=heads,
                                     embedding_module=self.entity_embeddings,
                                     embedding_dim=self.embedding_dim),
                self._get_embeddings(elements=relations,
                                     embedding_module=self.relation_embeddings,
                                     embedding_dim=self.embedding_dim),
                self._get_embeddings(elements=tails,
                                     embedding_module=self.entity_embeddings,
                                     embedding_dim=self.embedding_dim),
                )
