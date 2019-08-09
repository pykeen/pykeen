# -*- coding: utf-8 -*-

"""Implementation of ProjE."""

from typing import Optional

import numpy as np
import torch
import torch.autograd
from torch import nn

from ..base import BaseModule
from ...instance_creation_factories.triples_factory import TriplesFactory
from ...typing import OptionalLoss

__all__ = ['ProjE']


class ProjE(BaseModule):
    """An implementation of ProjE from [shi2017]_.

    .. seealso::

       - Official Implementation: <https://github.com/nddsg/ProjE>`_
    """

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 50,
            entity_embeddings: Optional[nn.Embedding] = None,
            relation_embeddings: Optional[nn.Embedding] = None,
            criterion: OptionalLoss = None,
            preferred_device: Optional[str] = None,
            random_seed: Optional[int] = None,
            inner_non_linearity: Optional[nn.Module] = None,
    ) -> None:
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss(reduction='mean')

        if inner_non_linearity is None:
            inner_non_linearity = nn.Tanh()

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
        self.relation_embeddings = relation_embeddings
        self.inner_non_linearity = inner_non_linearity

        init_bound = 6 / np.sqrt(self.embedding_dim)

        # Global entity projection
        self.d_e = (torch.rand(self.embedding_dim, requires_grad=True) - 2.0) * init_bound

        # Global relation projection
        self.d_r = (torch.rand(self.embedding_dim, requires_grad=True) - 2.0) * init_bound

        # Global combination bias
        self.b_c = (torch.rand(self.embedding_dim, requires_grad=True) - 2.0) * init_bound

        # Global combination bias
        self.b_p = (torch.rand(1, requires_grad=True) - 2.0) * init_bound

        if None in [self.entity_embeddings, self.relation_embeddings]:
            self._init_embeddings()

    def _init_embeddings(self):
        super()._init_embeddings()
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        # The same bound is used for both entity embeddings and relation embeddings because they have the same dimension
        embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )

    def forward_owa(  # noqa: D102
            self,
            batch: torch.tensor,
    ) -> torch.tensor:

        # Get embeddings
        h = self.entity_embeddings(batch[:, 0])
        r = self.relation_embeddings(batch[:, 1])
        t = self.entity_embeddings(batch[:, 2])

        # Compute score
        hidden = self.inner_non_linearity(self.d_e[None, :] * h + self.d_r[None, :] * r + self.b_c[None, :])
        scores = torch.sum(hidden * t, dim=-1, keepdim=True) + self.b_p

        return scores

    def forward_cwa(  # noqa: D102
            self,
            batch: torch.tensor,
    ) -> torch.tensor:
        # Get embeddings
        h = self.entity_embeddings(batch[:, 0])
        r = self.relation_embeddings(batch[:, 1])
        t = self.entity_embeddings.weight

        # Rank against all entities
        hidden = self.inner_non_linearity(self.d_e[None, :] * h + self.d_r[None, :] * r + self.b_c[None, :])
        scores = torch.sum(hidden[:, None, :] * t[None, :, :], dim=-1) + self.b_p

        return scores

    def forward_inverse_cwa(  # noqa: D102
            self,
            batch: torch.tensor,
    ) -> torch.tensor:
        # Get embeddings
        h = self.entity_embeddings.weight
        r = self.relation_embeddings(batch[:, 0])
        t = self.entity_embeddings(batch[:, 1])

        # Rank against all entities
        hidden = self.inner_non_linearity(self.d_e[None, None, :] * h[None, :, :] + (self.d_r[None, None, :] * r[:, None, :] + self.b_c[None, None, :]))
        scores = torch.sum(hidden * t[:, None, :], dim=-1) + self.b_p

        return scores
