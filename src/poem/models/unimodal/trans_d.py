# -*- coding: utf-8 -*-

"""Implementation of TransD."""

from typing import Optional

import torch
import torch.autograd
from torch import nn
from torch.nn.init import xavier_normal_

from poem.constants import RELATION_EMBEDDING_DIM, SCORING_FUNCTION_NORM
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.models.base import BaseModule
from poem.utils import slice_triples
from ...typing import OptionalLoss

__all__ = [
    'TransD',
]


class TransD(BaseModule):
    """An implementation of TransD from [ji2015]_.

    This model extends TransR to use fewer parameters.

    .. seealso::

       - OpenKE `implementation of TransD <https://github.com/thunlp/OpenKE/blob/master/models/TransD.py>`_
    """

    margin_ranking_loss_size_average: bool = True
    entity_embedding_max_norm = 1
    hyper_params = BaseModule.hyper_params + (RELATION_EMBEDDING_DIM, SCORING_FUNCTION_NORM)

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 50,
            entity_embeddings: nn.Embedding = None,
            relation_dim: int = 30,
            relation_embeddings: nn.Embedding = None,
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
            entity_embeddings = entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
        self.relation_embedding_dim = relation_dim
        self.scoring_fct_norm = scoring_fct_norm
        self.relation_embeddings = relation_embeddings
        self.entity_projections = None
        self.relation_projections = None

        if None in [self.entity_embeddings, self.relation_embeddings]:
            self._init_embeddings()

    def _init_embeddings(self):
        super()._init_embeddings()
        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_embedding_dim, max_norm=1)
        self.entity_projections = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_projections = nn.Embedding(self.num_relations, self.relation_embedding_dim)
        xavier_normal_(self.entity_embeddings.weight.data)
        xavier_normal_(self.relation_embeddings.weight.data)
        xavier_normal_(self.entity_projections.weight.data)
        xavier_normal_(self.relation_projections.weight.data)

    def forward_owa(self, triples):
        heads, relations, tails = slice_triples(triples)

        h_embs = self._get_embeddings(
            elements=heads,
            embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim,
        )
        r_embs = self._get_embeddings(
            elements=relations,
            embedding_module=self.relation_embeddings,
            embedding_dim=self.relation_embedding_dim,
        )
        t_embs = self._get_embeddings(
            elements=tails,
            embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim,
        )

        h_proj_vec_embs = self._get_embeddings(
            elements=heads,
            embedding_module=self.entity_projections,
            embedding_dim=self.embedding_dim,
        )
        r_projs_embs = self._get_embeddings(
            elements=relations,
            embedding_module=self.relation_projections,
            embedding_dim=self.relation_embedding_dim,
        )
        t_proj_vec_embs = self._get_embeddings(
            elements=tails,
            embedding_module=self.entity_projections,
            embedding_dim=self.embedding_dim,
        )

        proj_heads = self._project_entities(h_embs, h_proj_vec_embs, r_projs_embs)
        proj_tails = self._project_entities(t_embs, t_proj_vec_embs, r_projs_embs)

        # Add the vector element wise
        sum_res = proj_heads + r_embs - proj_tails
        scores = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        scores = - torch.mul(scores, scores)
        return scores

    # TODO: Implement forward_cwa

    def _project_entities(self, entity_embs, entity_proj_vecs, relation_projections):
        relation_projections = relation_projections.unsqueeze(-1)
        entity_proj_vecs = entity_proj_vecs.unsqueeze(-1).permute([0, 2, 1])
        transfer_matrices = torch.matmul(relation_projections, entity_proj_vecs)
        projected_entity_embs = torch.einsum('nmk,nk->nm', [transfer_matrices, entity_embs])
        return projected_entity_embs
