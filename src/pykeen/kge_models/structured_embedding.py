# -*- coding: utf-8 -*-

"""Implementation of structured model (SE)."""

import logging
from typing import Dict

import numpy as np
import torch
import torch.autograd
from torch import nn

from pykeen.constants import NORM_FOR_NORMALIZATION_OF_ENTITIES, SCORING_FUNCTION_NORM, SE_NAME
from .base import BaseModule, slice_triples
from .trans_e import TransEConfig

__all__ = [
    'StructuredEmbedding',
]

log = logging.getLogger(__name__)


class StructuredEmbedding(BaseModule):
    """An implementation of Structured Embedding (SE) [bordes2011]_.

    This model projects different matrices for each relation head and tail entity.

    .. [bordes2011] Bordes, A., *et al.* (2011). `Learning Structured Embeddings of Knowledge Bases
                    <http://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/download/3659/3898>`_. AAAI. Vol. 6. No. 1.
    """

    model_name = SE_NAME
    margin_ranking_loss_size_average: bool = True
    hyper_params = BaseModule.hyper_params + [SCORING_FUNCTION_NORM, NORM_FOR_NORMALIZATION_OF_ENTITIES]

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        config = TransEConfig.from_dict(config)

        # Embeddings
        self.l_p_norm_entities = config.lp_norm
        self.scoring_fct_norm = config.scoring_function_norm

        self.left_relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim * self.embedding_dim)
        self.right_relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim * self.embedding_dim)

        self._initialize()

    def _initialize(self):
        entity_embeddings_init_bound = left_relation_embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-entity_embeddings_init_bound,
            b=+entity_embeddings_init_bound,
        )
        nn.init.uniform_(
            self.left_relation_embeddings.weight.data,
            a=-left_relation_embeddings_init_bound,
            b=+left_relation_embeddings_init_bound,
        )

        # FIXME @mehdi why aren't the right relation embeddings initialized?

        norms = torch.norm(self.left_relation_embeddings.weight, p=2, dim=1).data
        self.left_relation_embeddings.weight.data = self.left_relation_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.left_relation_embeddings.weight))

    def predict(self, triples):
        # triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def forward(self, positives, negatives):
        # Normalise embeddings of entities
        norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))

        positive_scores = self._score_triples(positives)
        negative_scores = self._score_triples(negatives)

        loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss

    def _score_triples(self, triples):
        heads, relations, tails = slice_triples(triples)

        head_embeddings = self._get_entity_embeddings(heads)
        left_relation_embeddings = self._get_left_relation_embeddings(relations)
        projected_head_embeddings = self._project_entities(
            entity_embeddings=head_embeddings,
            relation_embeddings=left_relation_embeddings,
        )

        tail_embeddings = self._get_entity_embeddings(tails)
        right_relation_embeddings = self._get_right_relation_embeddings(relations)
        projected_tails_embeddings = self._project_entities(
            entity_embeddings=tail_embeddings,
            relation_embeddings=right_relation_embeddings,
        )

        scores = self._compute_scores(projected_head_embeddings, projected_tails_embeddings)
        return scores

    def _compute_scores(self, projected_head_embeddings, projected_tail_embeddings):
        difference = projected_head_embeddings - projected_tail_embeddings
        distances = torch.norm(difference, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        return distances

    def _project_entities(self, entity_embeddings, relation_embeddings):
        entity_embeddings = entity_embeddings.unsqueeze(-1)
        projected_entity_embs = torch.matmul(relation_embeddings, entity_embeddings)
        return projected_entity_embs

    def _get_left_relation_embeddings(self, relations):
        return self.left_relation_embeddings(relations).view(-1, self.embedding_dim, self.embedding_dim)

    def _get_right_relation_embeddings(self, relations):
        return self.right_relation_embeddings(relations).view(-1, self.embedding_dim, self.embedding_dim)
