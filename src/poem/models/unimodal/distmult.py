# -*- coding: utf-8 -*-

"""Implementation of DistMult."""

from typing import Optional

import numpy as np
import torch
import torch.autograd
from torch import nn
from torch.nn import functional

from poem.constants import DISTMULT_NAME, GPU
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.models.base import BaseModule
from poem.utils import slice_triples

__all__ = ['DistMult']


class DistMult(BaseModule):
    """An implementation of DistMult [yang2014]_.

    This model simplifies RESCAL by restricting matrices representing relations as diagonal matrices.

    .. [yang2014] Yang, B., Yih, W., He, X., Gao, J., & Deng, L. (2014). `Embedding Entities and Relations for Learning
                  and Inference in Knowledge Bases <https://arxiv.org/pdf/1412.6575.pdf>`_. CoRR, abs/1412.6575.

    Note:
      - For FB15k, yang et al. report 2 negatives per each positive.

    .. seealso::

       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/master/models/DistMult.py
    """

    model_name = DISTMULT_NAME
    margin_ranking_loss_size_average: bool = True

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 50,
            entity_embeddings: nn.Embedding = None,
            relation_embeddings:nn.Embedding = None,
            criterion: nn.modules.loss = nn.MarginRankingLoss(margin=1., reduction='mean'),
            preferred_device: str = GPU,
            random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            triples_factory = triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings= entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
        self.relation_embeddings = relation_embeddings

        if None in [self.entity_embeddings, self.relation_embeddings]:
            self._init_embeddings()

    def _init_embeddings(self):
        super()._init_embeddings()
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
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
            norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight),
        )

    def apply_forward_constraints(self):
        # Normalize embeddings of entities
        functional.normalize(self.entity_embeddings.weight.data, out=self.entity_embeddings.weight.data)
        self.forward_constraint_applied = True

    def forward_owa(self, triples):
        if not self.forward_constraint_applied:
            self.apply_forward_constraints()
        head_embeddings, relation_embeddings, tail_embeddings = self._get_triple_embeddings(triples)
        scores = torch.sum(head_embeddings * relation_embeddings * tail_embeddings, dim=1)
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
