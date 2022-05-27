# -*- coding: utf-8 -*-

"""Mocks for tests."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from pykeen.evaluation import Evaluator, MetricResults, RankBasedMetricResults
from pykeen.evaluation.ranking_metric_lookup import MetricKey
from pykeen.nn import Representation
from pykeen.typing import ExtendedTarget, MappedTriples, RankType, Target

__all__ = [
    "CustomRepresentation",
]


class CustomRepresentation(Representation):
    """A custom representation module with minimal implementation."""

    def __init__(self, num_entities: int, shape: Tuple[int, ...] = (2,)):
        super().__init__(max_id=num_entities, shape=shape)
        self.x = nn.Parameter(torch.rand(*shape))

    def _plain_forward(self, indices: Optional[torch.LongTensor] = None) -> torch.FloatTensor:  # noqa: D102
        n = self.max_id if indices is None else indices.shape[0]
        return self.x.unsqueeze(dim=0).repeat(n, *(1 for _ in self.shape))


class MockEvaluator(Evaluator):
    """A mock evaluator for testing early stopping."""

    def __init__(
        self,
        key: Optional[Tuple[str, ExtendedTarget, RankType]] = None,
        values: Optional[Iterable[float]] = None,
        automatic_memory_optimization: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(automatic_memory_optimization=automatic_memory_optimization)
        self.key = MetricKey.lookup(key)
        self.random_state = random_state
        if values is None:
            self.values = self.values_iter = None
        else:
            self.values = tuple(values)
            self.values_iter = iter(self.values)

    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        pass

    def finalize(self) -> MetricResults:  # noqa: D102
        result = RankBasedMetricResults.create_random(self.random_state)
        assert self.values_iter is not None
        if self.key not in result.data:
            raise KeyError(self.key)
        result.data[self.key] = next(self.values_iter)
        return result

    def __repr__(self):  # noqa: D105
        return f"{self.__class__.__name__}(values={self.values})"
