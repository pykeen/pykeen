# -*- coding: utf-8 -*-

"""Implementation of TransR."""

import numpy as np
import torch
import torch.autograd
from torch import nn

from poem.constants import GPU, SCORING_FUNCTION_NORM, TRANS_R_NAME, RELATION_EMBEDDING_DIM
from poem.models.base_owa import BaseOWAModule, slice_triples

__all__ = ['TransR']


class TransR(BaseOWAModule):
    """An implementation of TransR [lin2015]_.

    This model extends TransE and TransH by considering different vector spaces for entities and relations.

    .. [lin2015] Lin, Y., *et al.* (2015). `Learning entity and relation embeddings for knowledge graph completion
                 <http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/>`_. AAAI. Vol. 15.

    Constraints:
     * ||h||_2 <= 1: Done
     * ||r||_2 <= 1: Done
     * ||t||_2 <= 1: Done
     * ||h*M_r||_2 <= 1: Done
     * ||t*M_r||_2 <= 1: Done

    .. seealso::

       - Implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/master/models/TransR.py
       - PyTorch implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransR.py
    """

    model_name = TRANS_R_NAME
    margin_ranking_loss_size_average: bool = True
    # TODO: max_norm < 1.
    # max_norm = 1 according to the paper
    entity_embedding_max_norm = 1
    entity_embedding_norm_type = 2
    relation_embedding_max_norm = 1
    relation_embedding_norm_type = 2
    hyper_params = BaseOWAModule.hyper_params + [RELATION_EMBEDDING_DIM, SCORING_FUNCTION_NORM]

    def __init__(self, num_entities, num_relations, embedding_dim=50, relation_dim=30, scoring_fct_norm=1,
                 criterion=nn.MarginRankingLoss(margin=1., reduction='mean'), preferred_device=GPU) -> None:
        super(TransR, self).__init__(num_entities, num_relations, criterion, embedding_dim, preferred_device)

        # Embeddings
        self.relation_embedding_dim = relation_dim

        # max_norm = 1 according to the paper
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_embedding_dim,
                                                norm_type=self.relation_embedding_norm_type,
                                                max_norm=self.relation_embedding_max_norm)
        self.projection_matrix_embs = nn.Embedding(self.num_relations, self.relation_embedding_dim * self.embedding_dim)
        self.scoring_fct_norm = scoring_fct_norm

        self._initialize()

    def _initialize(self):
        """."""
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

        norms = torch.norm(self.relation_embeddings.weight, p=2, dim=1).data
        self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight))

    def _project_entities(self, entity_embs, projection_matrix_embs):
        projected_entity_embs = torch.einsum('nk,nkd->nd', [entity_embs, projection_matrix_embs])
        projected_entity_embs = torch.clamp(projected_entity_embs, max=1.)
        return projected_entity_embs

    def _score_triples(self, triples):
        """"""
        heads, relations, tails = slice_triples(triples)
        head_embeddings = self._get_embeddings(
            elements=heads, embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim
        )

        relation_embeddings = self._get_embeddings(
            elements=relations,
            embedding_module=self.relation_embeddings,
            embedding_dim=self.relation_embedding_dim
        )
        tail_embeddings = self._get_embeddings(
            elements=tails,
            embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim
        )
        proj_matrix_embs = self._get_embeddings(
            elements=relations,
            embedding_module=self.projection_matrix_embs,
            embedding_dim=self.embedding_dim
        )
        scores = self._compute_scores(head_embeddings, relation_embeddings, tail_embeddings, proj_matrix_embs)

        return scores

    def _compute_scores(self, head_embeddings, relation_embeddings, tail_embeddings, proj_matrix_embs):
        """"""

        proj_matrix_embs = proj_matrix_embs.view(-1, self.embedding_dim, self.relation_embedding_dim)
        proj_heads_embs = self._project_entities(head_embeddings, proj_matrix_embs)
        proj_tails_embs = self._project_entities(tail_embeddings, proj_matrix_embs)

        sum_res = proj_heads_embs + relation_embeddings - proj_tails_embs
        scores = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        scores = - torch.mul(scores, scores)

        return scores

    def forward(self, batch_positives, batch_negatives):
        positive_scores = self._score_triples(batch_positives)
        negative_scores = self._score_triples(batch_negatives)

        loss = self.compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss
