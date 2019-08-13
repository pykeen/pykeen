# -*- coding: utf-8 -*-

"""Implementation of TransR."""

from typing import Optional

import torch
import torch.autograd
from torch import nn
from torch.nn import functional

from ..base import BaseModule
from ..init import embedding_xavier_uniform_
from ...instance_creation_factories import TriplesFactory
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

       - OpenKE `TensorFlow implementation of TransR
         <https://github.com/thunlp/OpenKE/blob/master/models/TransR.py>`_
       - OpenKE `PyTorch implementation of TransR
         <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransR.py>`_
    """

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 50,
            entity_embeddings: Optional[nn.Embedding] = None,
            relation_dim: int = 30,
            relation_embeddings: Optional[nn.Embedding] = None,
            relation_projections: Optional[nn.Embedding] = None,
            scoring_fct_norm: int = 1,
            criterion: OptionalLoss = None,
            preferred_device: Optional[str] = None,
            random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the model."""
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
        self.relation_projections = relation_projections

        self._init_embeddings()

    def _init_embeddings(self) -> None:
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim, max_norm=1)
            embedding_xavier_uniform_(self.entity_embeddings)
        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_embedding_dim, max_norm=1)
            embedding_xavier_uniform_(self.relation_embeddings)
            # Initialise relation embeddings to unit length
            functional.normalize(self.relation_embeddings.weight.data, out=self.relation_embeddings.weight.data)
        if self.relation_projections is None:
            self.relation_projections = nn.Embedding(
                self.num_relations,
                self.relation_embedding_dim * self.embedding_dim,
            )

    def _apply_forward_constraints_if_necessary(self) -> None:
        # Normalize embeddings of entities
        if not self.forward_constraint_applied:
            functional.normalize(self.entity_embeddings.weight.data, out=self.entity_embeddings.weight.data)
            self.forward_constraint_applied = True

    def forward_owa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Guarantee forward constraints
        self._apply_forward_constraints_if_necessary()

        # Get embeddings
        h = self.entity_embeddings(batch[:, 0]).view(-1, 1, self.embedding_dim)
        r = self.relation_embeddings(batch[:, 1])
        t = self.entity_embeddings(batch[:, 2]).view(-1, 1, self.embedding_dim)
        m_r = self.relation_projections(batch[:, 1]).view(-1, self.embedding_dim, self.relation_embedding_dim)

        # Project entities
        h_bot = torch.renorm(h @ m_r, p=2, dim=-1, maxnorm=1.).view(-1, self.relation_embedding_dim)
        t_bot = torch.renorm(t @ m_r, p=2, dim=-1, maxnorm=1.).view(-1, self.relation_embedding_dim)

        score = -torch.norm(h_bot + r - t_bot, dim=-1, keepdim=True) ** 2
        return score

    def forward_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Guarantee forward constraints
        self._apply_forward_constraints_if_necessary()

        # Get embeddings
        h = self.entity_embeddings(batch[:, 0]).view(-1, 1, self.embedding_dim)
        r = self.relation_embeddings(batch[:, 1]).view(-1, 1, self.relation_embedding_dim)
        t = self.entity_embeddings.weight.view(1, -1, self.embedding_dim)
        m_r = self.relation_projections(batch[:, 1]).view(-1, self.embedding_dim, self.relation_embedding_dim)

        # Project entities
        h_bot = torch.renorm(h @ m_r, p=2, dim=-1, maxnorm=1.).view(-1, 1, self.relation_embedding_dim)
        t_bot = torch.renorm(t @ m_r, p=2, dim=-1, maxnorm=1.).view(-1, self.num_entities, self.relation_embedding_dim)

        score = -torch.norm(h_bot + r - t_bot, dim=-1) ** 2
        return score

    def forward_inverse_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Guarantee forward constraints
        self._apply_forward_constraints_if_necessary()

        # Get embeddings
        h = self.entity_embeddings.weight.view(1, -1, self.embedding_dim)
        r = self.relation_embeddings(batch[:, 0]).view(-1, 1, self.relation_embedding_dim)
        t = self.entity_embeddings(batch[:, 1]).view(-1, 1, self.embedding_dim)
        m_r = self.relation_projections(batch[:, 0]).view(-1, self.embedding_dim, self.relation_embedding_dim)

        # Project entities
        h_bot = torch.renorm(h @ m_r, p=2, dim=-1, maxnorm=1.).view(-1, self.num_entities, self.relation_embedding_dim)
        t_bot = torch.renorm(t @ m_r, p=2, dim=-1, maxnorm=1.).view(-1, 1, self.relation_embedding_dim)

        score = -torch.norm(h_bot + r - t_bot, dim=-1) ** 2
        return score
