# -*- coding: utf-8 -*-

"""Implementation of the TransE model."""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.autograd
from torch import nn
from torch.nn import functional

from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.models.base import BaseModule
from poem.utils import slice_triples
from ...typing import OptionalLoss

__all__ = [
    'TransE',
]

log = logging.getLogger(__name__)


class TransE(BaseModule):
    """An implementation of TransE from [bordes2013]_.

     This model considers a relation as a translation from the head to the tail entity.

    .. seealso::

       - OpenKE `implementation of TransE <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py>`_
    """

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 50,
            entity_embeddings: Optional[nn.Embedding] = None,
            relation_embeddings: Optional[nn.Embedding] = None,
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
        self.relation_embeddings = relation_embeddings

        if None in [self.entity_embeddings, self.relation_embeddings]:
            self._init_embeddings()

    def _init_embeddings(self):
        super()._init_embeddings()
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
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

        # Initialise relation embeddings to unit length
        functional.normalize(self.relation_embeddings.weight.data, out=self.relation_embeddings.weight.data)

    def apply_forward_constraints(self):
        functional.normalize(self.entity_embeddings.weight.data, out=self.entity_embeddings.weight.data)
        self.forward_constraint_applied = True

    def forward_owa(self, triples):
        if not self.forward_constraint_applied:
            self.apply_forward_constraints()
        head_embeddings, relation_embeddings, tail_embeddings = self._get_triple_embeddings(triples)
        # Add the vector element wise
        sum_res = head_embeddings + relation_embeddings - tail_embeddings
        scores = - torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
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

    @classmethod
    def get_model_params(cls) -> List:
        """Return model parameters."""
        base_params = BaseModule.get_model_params()
        return base_params + ['scoring_fct_norm']
