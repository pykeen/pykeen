# -*- coding: utf-8 -*-

"""Implementation of TransR."""

from typing import Optional

import numpy as np
import torch
import torch.autograd
from torch import nn
from torch.nn import functional

from poem.constants import RELATION_EMBEDDING_DIM, SCORING_FUNCTION_NORM
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.utils import slice_triples
from ..base import BaseModule
from ...typing import OptionalLoss

__all__ = ['TransR']


class TransR(BaseModule):
    """An implementation of TransR from [lin2015]_.

    This model extends TransE and TransH by considering different vector spaces for entities and relations.

    Constraints:
     * $||h||_2 <= 1$: Done
     * $||r||_2 <= 1$: Done
     * $||t||_2 <= 1$: Done
     * $||h*M_r||_2 <= 1$: Done
     * $||t*M_r||_2 <= 1$: Done

    .. seealso::

       - OpenKE `TensorFlow implementation of TransR <https://github.com/thunlp/OpenKE/blob/master/models/TransR.py>`_
       - OpenKE `PyTorch implementation of TransR <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransR.py>`_
    """

    margin_ranking_loss_size_average: bool = True
    # TODO: max_norm < 1.
    # max_norm = 1 according to the paper
    entity_embedding_max_norm = 1
    entity_embedding_norm_type = 2
    relation_embedding_max_norm = 1
    relation_embedding_norm_type = 2
    hyper_params = BaseModule.hyper_params + (RELATION_EMBEDDING_DIM, SCORING_FUNCTION_NORM)

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 50,
            entity_embeddings: Optional[nn.Embedding] = None,
            relation_dim: int = 30,
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
        self.relation_embedding_dim = relation_dim
        self.scoring_fct_norm = scoring_fct_norm
        self.relation_embeddings = relation_embeddings
        self.projection_matrix_embs = None

        if None in [self.entity_embeddings, self.relation_embeddings]:
            self._init_embeddings()

    def _init_embeddings(self):
        super()._init_embeddings()
        # max_norm = 1 according to the paper
        self.relation_embeddings = nn.Embedding(
            self.num_relations,
            self.relation_embedding_dim,
            norm_type=self.relation_embedding_norm_type,
            max_norm=self.relation_embedding_max_norm,
        )
        self.projection_matrix_embs = nn.Embedding(
            self.num_relations,
            self.relation_embedding_dim * self.embedding_dim,
        )
        entity_embeddings_init_bound = relation_embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-entity_embeddings_init_bound,
            b=entity_embeddings_init_bound,
        )
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=-relation_embeddings_init_bound,
            b=relation_embeddings_init_bound,
        )

        # Initialise relation embeddings to unit length
        functional.normalize(self.relation_embeddings.weight.data, out=self.relation_embeddings.weight.data)

    def _project_entities(self, entity_embs, projection_matrix_embs):
        projected_entity_embs = torch.einsum('nk,nkd->nd', [entity_embs, projection_matrix_embs])
        projected_entity_embs = torch.clamp(projected_entity_embs, max=1.)
        return projected_entity_embs

    def forward_owa(self, triples):
        heads, relations, tails = slice_triples(triples)
        head_embeddings = self._get_embeddings(
            elements=heads, embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim,
        )

        relation_embeddings = self._get_embeddings(
            elements=relations,
            embedding_module=self.relation_embeddings,
            embedding_dim=self.relation_embedding_dim,
        )
        tail_embeddings = self._get_embeddings(
            elements=tails,
            embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim,
        )
        proj_matrix_embs = self._get_embeddings(
            elements=relations,
            embedding_module=self.projection_matrix_embs,
            embedding_dim=self.embedding_dim,
        )
        proj_matrix_embs = proj_matrix_embs.view(-1, self.embedding_dim, self.relation_embedding_dim)
        proj_heads_embs = self._project_entities(head_embeddings, proj_matrix_embs)
        proj_tails_embs = self._project_entities(tail_embeddings, proj_matrix_embs)
        sum_res = proj_heads_embs + relation_embeddings - proj_tails_embs
        scores = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        scores = - torch.mul(scores, scores)
        return scores

    # TODO: Implement forward_cwa
