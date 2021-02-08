# -*- coding: utf-8 -*-

"""Mocks for tests."""

from typing import Optional

import torch
from torch import nn

from pykeen.nn import RepresentationModule


class CustomRepresentations(RepresentationModule):
    """A custom representation module with minimal implementation."""

    def __init__(self, num_entities: int, embedding_dim: int = 2):
        super().__init__()
        self.num_embeddings = num_entities
        self.embedding_dim = embedding_dim
        self.x = nn.Parameter(torch.rand(embedding_dim))

    def forward(self, indices: Optional[torch.LongTensor] = None) -> torch.FloatTensor:  # noqa:D102
        n = self.num_embeddings if indices is None else indices.shape[0]
        return self.x.unsqueeze(dim=0).repeat(n, 1)

    def get_in_canonical_shape(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa:D102
        x = self(indices=indices)
        if indices is None:
            return x.unsqueeze(dim=0)
        return x.unsqueeze(dim=1)
