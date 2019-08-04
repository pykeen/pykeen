# -*- coding: utf-8 -*-

"""Implementation of TransD."""

from typing import Optional

import torch
import torch.autograd
from torch import nn
from torch.nn.init import xavier_normal_

from poem.instance_creation_factories.triples_factory import TriplesFactory
from ..base import BaseModule
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

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 50,
            entity_embeddings: Optional[nn.Embedding] = None,
            relation_dim: int = 30,
            relation_embeddings: Optional[nn.Embedding] = None,
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
        # The dimensions affected by h_bot
        self.change_dim = min(self.embedding_dim, self.relation_embedding_dim)
        self.relation_embeddings = relation_embeddings
        self.entity_projections = None
        self.relation_projections = None

        if None in [self.entity_embeddings, self.relation_embeddings]:
            self._init_embeddings()

    def _init_embeddings(self):
        super()._init_embeddings()
        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_embedding_dim, max_norm=1.)
        self.entity_projections = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_projections = nn.Embedding(self.num_relations, self.relation_embedding_dim)
        xavier_normal_(self.entity_embeddings.weight.data)
        xavier_normal_(self.relation_embeddings.weight.data)
        xavier_normal_(self.entity_projections.weight.data)
        xavier_normal_(self.relation_projections.weight.data)

    def forward_owa(self, batch: torch.tensor) -> torch.tensor:
        """Forward pass for training with the OWA."""
        # Get embeddings
        h = self.entity_embeddings(batch[:, 0])
        r = self.relation_embeddings(batch[:, 1])
        t = self.entity_embeddings(batch[:, 2])

        # Get projection vectors
        hp = self.entity_projections(batch[:, 0])
        rp = self.relation_projections(batch[:, 1])
        tp = self.entity_projections(batch[:, 2])

        # Project entities
        # h_bot = M_rh h
        #       = (r_p h_p.T + I) h
        #       = r_p h_p.T h + h
        h_bot = rp * torch.sum(hp * h, dim=-1, keepdim=True)
        h_bot[:, :self.change_dim] += h[:, :self.change_dim]
        t_bot = rp * torch.sum(tp * t, dim=-1, keepdim=True)
        t_bot[:, :self.change_dim] += t[:, :self.change_dim]

        # Enforce constraints
        h_bot = torch.renorm(h_bot, p=2, dim=-1, maxnorm=1)
        t_bot = torch.renorm(t_bot, p=2, dim=-1, maxnorm=1)

        # score = -||h_bot + r - t_bot||_2^2
        return -torch.norm(h_bot + r - t_bot, dim=-1, keepdim=True, p=2) ** 2

    def forward_cwa(self, batch: torch.tensor) -> torch.tensor:
        """Forward pass using right side (object) prediction for training with the CWA."""
        # Get embeddings
        h = self.entity_embeddings(batch[:, 0])
        r = self.relation_embeddings(batch[:, 1])
        t = self.entity_embeddings.weight

        # Get projection vectors
        hp = self.entity_projections(batch[:, 0])
        rp = self.relation_projections(batch[:, 1])
        tp = self.entity_projections.weight

        # Project entities
        # h_bot = M_rh h
        #       = (r_p h_p.T + I) h
        #       = r_p h_p.T h + h
        h_bot = rp * torch.sum(hp * h, dim=-1, keepdim=True)
        h_bot[:, :self.change_dim] += h[:, :self.change_dim]
        t_bot = rp[:, None, :] * torch.sum(tp * t, dim=-1)[None, :, None]
        t_bot[:, :, :self.change_dim] += t[None, :, :self.change_dim]

        # Enforce constraints
        h_bot = torch.renorm(h_bot, p=2, dim=-1, maxnorm=1)
        t_bot = torch.renorm(t_bot, p=2, dim=-1, maxnorm=1)

        # score = -||h_bot + r - t_bot||_2^2
        return -torch.norm(h_bot[:, None, :] + r[:, None, :] - t_bot, dim=-1, p=2) ** 2

    def forward_inverse_cwa(self, batch: torch.tensor) -> torch.tensor:
        """Forward pass using left side (subject) prediction for training with the CWA."""
        # Get embeddings
        h = self.entity_embeddings.weight
        r = self.relation_embeddings(batch[:, 0])
        t = self.entity_embeddings(batch[:, 1])

        # Get projection vectors
        hp = self.entity_projections.weight
        rp = self.relation_projections(batch[:, 0])
        tp = self.entity_projections(batch[:, 1])

        # Project entities
        # h_bot = M_rh h
        #       = (r_p h_p.T + I) h
        #       = r_p h_p.T h + h
        h_bot = rp[:, None, :] * torch.sum(hp * h, dim=-1)[None, :, None]
        h_bot[:, :, :self.change_dim] += h[None, :, :self.change_dim]
        t_bot = rp * torch.sum(tp * t, dim=-1, keepdim=True)
        t_bot[:, :self.change_dim] += t[:, :self.change_dim]

        # Enforce constraints
        h_bot = torch.renorm(h_bot, p=2, dim=-1, maxnorm=1)
        t_bot = torch.renorm(t_bot, p=2, dim=-1, maxnorm=1)

        # score = -||h_bot + r - t_bot||_2^2
        return -torch.norm(h_bot + r[:, None, :] - t_bot[:, None, :], dim=-1, p=2) ** 2
