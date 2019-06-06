# -*- coding: utf-8 -*-

"""Implementation of TransD."""

from typing import Dict

import torch
import torch.autograd
from torch import nn

from poem.constants import GPU, TRANS_D_NAME, RELATION_EMBEDDING_DIM, SCORING_FUNCTION_NORM
from poem.models.base_owa import BaseOWAModule, slice_triples

__all__ = [
    'TransD',
]


class TransD(BaseOWAModule):
    """An implementation of TransD [ji2015]_.

    This model extends TransR to use fewer parameters.

    .. [ji2015] Ji, G., *et al.* (2015). `Knowledge graph embedding via dynamic mapping matrix
                <http://www.aclweb.org/anthology/P15-1067>`_. ACL.

    .. seealso::

       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/master/models/TransD.py
    """

    model_name = TRANS_D_NAME
    margin_ranking_loss_size_average: bool = True
    entity_embedding_max_norm = 1
    hyper_params = BaseOWAModule.hyper_params + [RELATION_EMBEDDING_DIM, SCORING_FUNCTION_NORM]

    def __init__(self, num_entities, num_relations, embedding_dim=50, relation_dim=30, scoring_fct_norm=1,
                 criterion=nn.MarginRankingLoss(margin=1., reduction='mean'), preferred_device=GPU) -> None:
        super(TransD, self).__init__(num_entities, num_relations, criterion, embedding_dim, preferred_device)

        # Embeddings
        self.relation_embedding_dim = relation_dim

        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_embedding_dim, max_norm=1)
        self.entity_projections = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_projections = nn.Embedding(self.num_relations, self.relation_embedding_dim)

        # FIXME @mehdi what about initialization?

        self.scoring_fct_norm = scoring_fct_norm

    def predict_scores(self, triples):
        # triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def forward(self, batch_positives, batch_negatives):
        positive_scores = self._score_triples(batch_positives)
        negative_scores = self._score_triples(batch_negatives)
        loss = self.compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss

    def _score_triples(self, triples):
        heads, relations, tails = slice_triples(triples)

        h_embs = self._get_embeddings(
            elements=heads,
            embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim
        )
        r_embs = self._get_embeddings(
            elements=relations,
            embedding_module=self.relation_embeddings,
            embedding_dim=self.relation_embedding_dim
        )
        t_embs = self._get_embeddings(
            elements=tails,
            embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim
        )

        h_proj_vec_embs = self._get_embeddings(
            elements=heads,
            embedding_module=self.entity_projections,
            embedding_dim=self.embedding_dim
        )
        r_projs_embs = self._get_embeddings(
            elements=relations,
            embedding_module=self.relation_projections,
            embedding_dim=self.relation_embedding_dim
        )
        t_proj_vec_embs = self._get_embeddings(
            elements=tails,
            embedding_module=self.entity_projections,
            embedding_dim=self.embedding_dim
        )

        proj_heads = self._project_entities(h_embs, h_proj_vec_embs, r_projs_embs)
        proj_tails = self._project_entities(t_embs, t_proj_vec_embs, r_projs_embs)

        scores = self._compute_scores(h_embs=proj_heads, r_embs=r_embs, t_embs=proj_tails)
        return scores

    def _compute_scores(self, h_embs, r_embs, t_embs):
        # Add the vector element wise
        sum_res = h_embs + r_embs - t_embs
        scores = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        scores = - torch.mul(scores, scores)
        return scores

    def _project_entities(self, entity_embs, entity_proj_vecs, relation_projections):
        """"""
        relation_projections = relation_projections.unsqueeze(-1)
        entity_proj_vecs = entity_proj_vecs.unsqueeze(-1).permute([0, 2, 1])
        transfer_matrices = torch.matmul(relation_projections, entity_proj_vecs)
        projected_entity_embs = torch.einsum('nmk,nk->nm', [transfer_matrices, entity_embs])
        return projected_entity_embs

