# -*- coding: utf-8 -*-

"""Mock models that return fixed scores.

These are useful for baselines.
"""

import torch

from .base import EntityRelationEmbeddingModel, Model
from ..triples import CoreTriplesFactory

__all__ = [
    "MockModel",
]


class MockModel(EntityRelationEmbeddingModel):
    """A mock model returning fake scores."""

    hpo_default = {}

    def __init__(self, *, triples_factory: CoreTriplesFactory, **_kwargs):
        super().__init__(
            triples_factory=triples_factory,
            entity_representations=[],
            relation_representations=[],
        )
        self.num_entities = triples_factory.num_entities
        self.num_relations = triples_factory.num_relations

    def _generate_fake_scores(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Generate fake scores."""
        return (h * (self.num_entities * self.num_relations) + r * self.num_entities + t).requires_grad_(True)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._generate_fake_scores(*hrt_batch.t()).unsqueeze(dim=-1)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._generate_fake_scores(
            h=hr_batch[:, 0:1],
            r=hr_batch[:, 1:2],
            t=torch.arange(self.num_entities).unsqueeze(dim=0),
        )

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._generate_fake_scores(
            h=torch.arange(self.num_entities).unsqueeze(dim=0),
            r=rt_batch[:, 0:1],
            t=rt_batch[:, 1:2],
        )

    def reset_parameters_(self) -> Model:  # noqa: D102
        pass  # Not needed for unittest
