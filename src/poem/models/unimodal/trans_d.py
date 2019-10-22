# -*- coding: utf-8 -*-

"""Implementation of TransD."""

from typing import Optional

import torch
import torch.autograd
from torch import nn

from ..base import BaseModule
from ..init import embedding_xavier_normal_
from ...triples import TriplesFactory
from ...typing import Loss

__all__ = [
    'TransD',
]


class TransD(BaseModule):
    """An implementation of TransD from [ji2015]_.

    This model extends TransR to use fewer parameters.

    .. seealso::

       - OpenKE `implementation of TransD <https://github.com/thunlp/OpenKE/blob/master/models/TransD.py>`_
    """

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        entity_embeddings: Optional[nn.Embedding] = None,
        entity_projections: Optional[nn.Embedding] = None,
        relation_dim: int = 30,
        relation_embeddings: Optional[nn.Embedding] = None,
        relation_projections: Optional[nn.Embedding] = None,
        criterion: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        init: bool = True,
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
        self.entity_projections = entity_projections
        self.relation_projections = relation_projections

        if init:
            self.init_empty_weights_()

    def init_empty_weights_(self):  # noqa: D102
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim, max_norm=1)
            embedding_xavier_normal_(self.entity_embeddings)
        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_embedding_dim, max_norm=1.)
            embedding_xavier_normal_(self.relation_embeddings)
        if self.entity_projections is None:
            self.entity_projections = nn.Embedding(self.num_entities, self.embedding_dim)
            embedding_xavier_normal_(self.entity_projections)
        if self.relation_projections is None:
            self.relation_projections = nn.Embedding(self.num_relations, self.relation_embedding_dim)
            embedding_xavier_normal_(self.relation_projections)

        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.entity_projections = None
        self.relation_embeddings = None
        self.relation_projections = None
        return self

    def forward_owa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
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

    def forward_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
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

    def forward_inverse_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
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
