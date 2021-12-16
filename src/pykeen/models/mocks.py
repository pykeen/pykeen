# -*- coding: utf-8 -*-

"""Mock models that return random results.

These are useful for baselines.
"""

import torch

from pykeen.models import EntityRelationEmbeddingModel, Model
from pykeen.nn import EmbeddingSpecification
from pykeen.triples import CoreTriplesFactory

__all__ = [
    "MockModel",
]


class MockModel(EntityRelationEmbeddingModel):
    """A mock model returning fake scores."""

    def __init__(self, *, triples_factory: CoreTriplesFactory):
        super().__init__(
            triples_factory=triples_factory,
            entity_representations=EmbeddingSpecification(embedding_dim=50),
            relation_representations=EmbeddingSpecification(embedding_dim=50),
        )
        num_entities = self.num_entities
        self.scores = torch.arange(num_entities, dtype=torch.float, requires_grad=True)
        self.num_backward_propagations = 0

    def _generate_fake_scores(self, batch: torch.LongTensor) -> torch.FloatTensor:
        """Generate fake scores s[b, i] = i of size (batch_size, num_entities)."""
        batch_size = batch.shape[0]
        batch_scores = self.scores.view(1, -1).repeat(batch_size, 1)
        assert batch_scores.shape == (batch_size, self.num_entities)
        return batch_scores

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self.scores[torch.randint(high=self.num_entities, size=hrt_batch.shape[:-1])]

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._generate_fake_scores(batch=hr_batch)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._generate_fake_scores(batch=rt_batch)

    def reset_parameters_(self) -> Model:  # noqa: D102
        pass  # Not needed for unittest
