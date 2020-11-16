# -*- coding: utf-8 -*-

"""Mocks for testing PyKEEN."""

from typing import Optional

import torch

from pykeen.models.base import Model
from pykeen.triples import TriplesFactory

__all__ = [
    'MockModel',
]


class MockModel(Model):
    """A mock model returning fake scores."""

    def __init__(self, triples_factory: TriplesFactory, automatic_memory_optimization: bool = True):
        super().__init__(
            triples_factory=triples_factory,
            automatic_memory_optimization=automatic_memory_optimization,
        )

    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
        slice_size: Optional[int] = None,
        slice_dim: Optional[str] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # (batch_size, num_heads, num_relations, num_tails)
        scores = torch.zeros(1, 1, 1, 1, requires_grad=True)  # for requires_grad
        # reproducible scores
        for i, (ind, num) in enumerate((
            (h_indices, self.num_entities),
            (r_indices, self.num_relations),
            (t_indices, self.num_entities),
        )):
            shape = [1, 1, 1, 1]
            if ind is None:
                shape[i + 1] = num
                delta = torch.arange(num)
            else:
                shape[0] = len(ind)
                delta = ind
            scores = scores + delta.float().view(*shape)
        return scores
