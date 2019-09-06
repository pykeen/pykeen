# -*- coding: utf-8 -*-

"""Implementation of UM."""

from typing import Optional

import torch
import torch.autograd
from torch import nn

from ..base import BaseModule
from ..init import embedding_xavier_uniform_
from ...triples import TriplesFactory
from ...typing import OptionalLoss

__all__ = [
    'UnstructuredModel',
]


class UnstructuredModel(BaseModule):
    """An implementation of Unstructured Model (UM) from [bordes2014]_."""

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 50,
            entity_embeddings: Optional[nn.Embedding] = None,
            scoring_fct_norm: int = 1,
            criterion: OptionalLoss = None,
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
        self.scoring_fct_norm = scoring_fct_norm
        if init:
            self.init_empty_weights_()

    def init_empty_weights_(self):  # noqa: D102
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
            embedding_xavier_uniform_(self.entity_embeddings)
        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        return self

    def forward_owa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(batch[:, 0])
        t = self.entity_embeddings(batch[:, 2])

        return -torch.norm(h - t, dim=-1, p=self.scoring_fct_norm, keepdim=True) ** 2

    def forward_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(batch[:, 0]).view(-1, 1, self.embedding_dim)
        t = self.entity_embeddings.weight.view(1, -1, self.embedding_dim)

        return -torch.norm(h - t, dim=-1, p=self.scoring_fct_norm) ** 2

    def forward_inverse_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings.weight.view(1, -1, self.embedding_dim)
        t = self.entity_embeddings(batch[:, 1]).view(-1, 1, self.embedding_dim)

        return -torch.norm(h - t, dim=-1, p=self.scoring_fct_norm) ** 2
