# -*- coding: utf-8 -*-

"""Implementation of structured model (SE)."""

import logging
from typing import Optional

import numpy as np
import torch
import torch.autograd
from torch import nn
from torch.nn import functional

from poem.constants import GPU, SCORING_FUNCTION_NORM, SE_NAME
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.models.base import BaseModule
from poem.utils import slice_triples

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
    hyper_params = BaseModule.hyper_params + (SCORING_FUNCTION_NORM,)

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 50,
            left_relation_embeddings: nn.Embedding = None,
            right_relation_embeddings: nn.Embedding = None,
            scoring_fct_norm: int = 1,
            criterion: nn.modules.loss = nn.MarginRankingLoss(margin=1., reduction='mean'),
            preferred_device: str = GPU,
            random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            triples_factory = triples_factory,
            embedding_dim=embedding_dim,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        # Embeddings
        self.scoring_fct_norm = scoring_fct_norm

        self.left_relation_embeddings = left_relation_embeddings
        self.right_relation_embeddings = right_relation_embeddings

        if None in [self.left_relation_embeddings, self.right_relation_embeddings]:
            self._init_embeddings()

    def _init_embeddings(self):
        super()._init_embeddings()
        self.left_relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim * self.embedding_dim)
        self.right_relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim * self.embedding_dim)

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
            norms.view(self.num_relations, 1).expand_as(self.left_relation_embeddings.weight),
        )

    def apply_forward_constraints(self):
        # Normalise embeddings of entities
        functional.normalize(self.entity_embeddings.weight.data, out=self.entity_embeddings.weight.data)
        self.forward_constraint_applied = True

    def forward_owa(self, triples):
        if not self.forward_constraint_applied:
            self.apply_forward_constraints()
        heads, relations, tails = slice_triples(triples)

        head_embeddings = self._get_embeddings(
            elements=heads,
            embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim,
        )
        left_relation_embeddings = self._get_left_relation_embeddings(relations)
        projected_head_embeddings = self._project_entities(
            entity_embeddings=head_embeddings,
            relation_embeddings=left_relation_embeddings,
        )

        tail_embeddings = self._get_embeddings(
            elements=tails,
            embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim,
        )

        right_relation_embeddings = self._get_right_relation_embeddings(relations)
        projected_tails_embeddings = self._project_entities(
            entity_embeddings=tail_embeddings,
            relation_embeddings=right_relation_embeddings,
        )
        difference = projected_head_embeddings - projected_tails_embeddings
        scores = - torch.norm(difference, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        return scores

    # TODO: Implement forward_cwa

    def _project_entities(self, entity_embeddings, relation_embeddings):
        entity_embeddings = entity_embeddings.unsqueeze(-1)
        projected_entity_embs = torch.matmul(relation_embeddings, entity_embeddings)
        return projected_entity_embs

    def _get_left_relation_embeddings(self, relations):
        return self.left_relation_embeddings(relations).view(-1, self.embedding_dim, self.embedding_dim)

    def _get_right_relation_embeddings(self, relations):
        return self.right_relation_embeddings(relations).view(-1, self.embedding_dim, self.embedding_dim)
