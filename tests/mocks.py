# -*- coding: utf-8 -*-

"""Mocks for tests."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from pykeen.evaluation import Evaluator, MetricResults, RankBasedMetricResults
from pykeen.evaluation.rank_based_evaluator import RANK_REALISTIC, RANK_TYPES, SIDES
from pykeen.models import EntityRelationEmbeddingModel, Model
from pykeen.nn.emb import EmbeddingSpecification, RepresentationModule
from pykeen.triples import CoreTriplesFactory
from pykeen.typing import MappedTriples

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


class MockEvaluator(Evaluator):
    """A mock evaluator for testing early stopping."""

    def __init__(self, losses: Iterable[float], automatic_memory_optimization: bool = True) -> None:
        super().__init__(automatic_memory_optimization=automatic_memory_optimization)
        self.losses = tuple(losses)
        self.losses_iter = iter(self.losses)

    def process_tail_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        pass

    def process_head_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        pass

    def finalize(self) -> MetricResults:  # noqa: D102
        hits = next(self.losses_iter)
        dummy_1 = {
            side: {
                rank_type: 10.0
                for rank_type in RANK_TYPES
            }
            for side in SIDES
        }
        dummy_2 = {
            side: {
                rank_type: 1.0
                for rank_type in RANK_TYPES
            }
            for side in SIDES
        }
        return RankBasedMetricResults(
            arithmetic_mean_rank=dummy_1,
            geometric_mean_rank=dummy_1,
            harmonic_mean_rank=dummy_1,
            median_rank=dummy_1,
            inverse_arithmetic_mean_rank=dummy_2,
            inverse_harmonic_mean_rank=dummy_2,
            inverse_geometric_mean_rank=dummy_2,
            inverse_median_rank=dummy_2,
            adjusted_arithmetic_mean_rank=dummy_2,
            adjusted_arithmetic_mean_rank_index={
                side: {
                    RANK_REALISTIC: 0.0,
                }
                for side in SIDES
            },
            rank_std=dummy_1,
            rank_var=dummy_1,
            rank_mad=dummy_1,
            hits_at_k={
                side: {
                    rank_type: {
                        10: hits,
                    } for rank_type in RANK_TYPES
                }
                for side in SIDES
            },
        )

    def __repr__(self):  # noqa: D105
        return f'{self.__class__.__name__}(losses={self.losses})'
