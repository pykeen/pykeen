# -*- coding: utf-8 -*-

"""Implementation of UM."""

import logging
from typing import Optional

import numpy as np
import torch
import torch.autograd
from torch import nn

from ..base import BaseModule
from ...instance_creation_factories.triples_factory import TriplesFactory
from ...typing import OptionalLoss

__all__ = [
    'UnstructuredModel',
]

log = logging.getLogger(__name__)


class UnstructuredModel(BaseModule):
    """An implementation of Unstructured Model (UM) from [bordes2014]_."""

    margin_ranking_loss_size_average: bool = True

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 50,
            entity_embeddings: Optional[nn.Embedding] = None,
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
            entity_embeddings=entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
        self.scoring_fct_norm = scoring_fct_norm

        if None in [self.entity_embeddings]:
            self._init_embeddings()

    def _init_embeddings(self):
        super()._init_embeddings()
        entity_embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-entity_embeddings_init_bound,
            b=entity_embeddings_init_bound,
        )

    def forward_owa(self, batch: torch.tensor) -> torch.tensor:
        """Forward pass for training with the OWA."""
        h = self.entity_embeddings(batch[:, 0])
        t = self.entity_embeddings(batch[:, 2])

        return -torch.norm(h - t, dim=-1, p=self.scoring_fct_norm, keepdim=True) ** 2

    def forward_cwa(self, batch: torch.tensor) -> torch.tensor:
        """Forward pass using right side (object) prediction for training with the CWA."""
        h = self.entity_embeddings(batch[:, 0]).view(-1, 1, self.embedding_dim)
        t = self.entity_embeddings.weight.view(1, -1, self.embedding_dim)

        return -torch.norm(h - t, dim=-1, p=self.scoring_fct_norm) ** 2

    def forward_inverse_cwa(self, batch: torch.tensor) -> torch.tensor:
        """Forward pass using left side (subject) prediction for training with the CWA."""
        h = self.entity_embeddings.weight.view(1, -1, self.embedding_dim)
        t = self.entity_embeddings(batch[:, 1]).view(-1, 1, self.embedding_dim)

        return -torch.norm(h - t, dim=-1, p=self.scoring_fct_norm) ** 2
