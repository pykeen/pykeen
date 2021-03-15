# -*- coding: utf-8 -*-

"""Mocks for tests."""

from typing import Optional, Tuple

import torch
from torch import nn

from pykeen.models import EntityRelationEmbeddingModel, Model
from pykeen.nn import EmbeddingSpecification, RepresentationModule
from pykeen.triples import TriplesFactory

__all__ = [
    'CustomRepresentations',
    'MockModel',
]


class CustomRepresentations(RepresentationModule):
    """A custom representation module with minimal implementation."""

    def __init__(self, num_entities: int, shape: Tuple[int, ...] = (2,)):
        super().__init__(max_id=num_entities, shape=shape)
        self.x = nn.Parameter(torch.rand(*shape))

    def forward(self, indices: Optional[torch.LongTensor] = None) -> torch.FloatTensor:  # noqa:D102
        n = self.max_id if indices is None else indices.shape[0]
        return self.x.unsqueeze(dim=0).repeat(n, *(1 for _ in self.shape))


class MockModel(EntityRelationEmbeddingModel):
    """A mock model returning fake scores."""

    def __init__(self, triples_factory: TriplesFactory):
        super().__init__(
            triples_factory=triples_factory,
            entity_representations=EmbeddingSpecification(embedding_dim=50),
            relation_representations=EmbeddingSpecification(embedding_dim=50),
        )
        num_entities = self.num_entities
        self.scores = torch.arange(num_entities, dtype=torch.float)

    def _generate_fake_scores(self, batch: torch.LongTensor) -> torch.FloatTensor:
        """Generate fake scores s[b, i] = i of size (batch_size, num_entities)."""
        batch_size = batch.shape[0]
        batch_scores = self.scores.view(1, -1).repeat(batch_size, 1)
        assert batch_scores.shape == (batch_size, self.num_entities)
        return batch_scores

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._generate_fake_scores(batch=hrt_batch)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._generate_fake_scores(batch=hr_batch)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._generate_fake_scores(batch=rt_batch)

    def reset_parameters_(self) -> Model:  # noqa: D102
        pass  # Not needed for unittest
