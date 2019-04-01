# -*- coding: utf-8 -*-

"""Implementation of TransD."""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.autograd
from torch import nn

from pykeen.constants import RELATION_EMBEDDING_DIM, SCORING_FUNCTION_NORM, TRANS_D_NAME
from pykeen.kge_models.base import BaseModule, slice_triples

__all__ = [
    'TransD',
    'TransDConfig',
]


@dataclass
class TransDConfig:
    relation_embedding_dim: int
    scoring_function_norm: str

    @classmethod
    def from_dict(cls, config: Dict) -> 'TransDConfig':
        """Generate an instance from a dictionary."""
        return cls(
            relation_embedding_dim=config[RELATION_EMBEDDING_DIM],
            scoring_function_norm=config[SCORING_FUNCTION_NORM],
        )


class TransD(BaseModule):
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
    hyper_params = BaseModule.hyper_params + [RELATION_EMBEDDING_DIM, SCORING_FUNCTION_NORM]

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        config = TransDConfig.from_dict(config)

        # Embeddings
        self.relation_embedding_dim = config.relation_embedding_dim

        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_embedding_dim, max_norm=1)
        self.entity_projections = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_projections = nn.Embedding(self.num_relations, self.relation_embedding_dim)

        # FIXME @mehdi what about initialization?

        self.scoring_fct_norm = config.scoring_function_norm

    def predict(self, triples):
        # triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def _compute_loss(self, positive_scores, negative_scores):
        # y == -1 indicates that second input to criterion should get a larger loss
        # y = torch.Tensor([-1]).cuda()
        # NOTE: y = 1 is important
        # y = torch.tensor([-1], dtype=torch.float, device=self.device)
        y = np.repeat([-1], repeats=positive_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        # Scores for the psotive and negative triples
        positive_scores = torch.tensor(positive_scores, dtype=torch.float, device=self.device)
        negative_scores = torch.tensor(negative_scores, dtype=torch.float, device=self.device)

        loss = self.criterion(positive_scores, negative_scores, y)
        return loss

    def forward(self, positives, negatives):
        positive_scores = self._score_triples(positives)
        negative_scores = self._score_triples(negatives)
        loss = self._compute_loss(positive_scores, negative_scores)
        return loss

    def _score_triples(self, triples):
        heads, relations, tails = slice_triples(triples)

        h_embs = self._get_entity_embeddings(heads)
        r_embs = self._get_relation_embeddings(relations)
        t_embs = self._get_entity_embeddings(tails)

        h_proj_vec_embs = self._get_entity_projections(heads)
        r_projs_embs = self._get_relation_projections(relations)
        t_proj_vec_embs = self._get_entity_projections(tails)

        proj_heads = self._project_entities(h_embs, h_proj_vec_embs, r_projs_embs)
        proj_tails = self._project_entities(t_embs, t_proj_vec_embs, r_projs_embs)

        scores = self._compute_scores(h_embs=proj_heads, r_embs=r_embs, t_embs=proj_tails)
        return scores

    def _compute_scores(self, h_embs, r_embs, t_embs):
        # Add the vector element wise
        sum_res = h_embs + r_embs - t_embs
        distances = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        distances = torch.mul(distances, distances)
        return distances

    def _project_entities(self, entity_embs, entity_proj_vecs, relation_projections):
        relation_projections = relation_projections.unsqueeze(-1)
        entity_proj_vecs = entity_proj_vecs.unsqueeze(-1).permute([0, 2, 1])
        transfer_matrices = torch.matmul(relation_projections, entity_proj_vecs)
        projected_entity_embs = torch.einsum('nmk,nk->nm', [transfer_matrices, entity_embs])
        return projected_entity_embs

    def _get_entity_projections(self, entities):
        return self.entity_projections(entities).view(-1, self.embedding_dim)

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.relation_embedding_dim)

    def _get_relation_projections(self, relations):
        return self.relation_projections(relations).view(-1, self.relation_embedding_dim)
