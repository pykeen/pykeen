"""Mocks for tests."""

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from pykeen.evaluation import Evaluator, MetricResults, RankBasedMetricResults
from pykeen.nn import Representation
from pykeen.typing import ExtendedTarget, MappedTriples, RankType, Target

__all__ = [
    "CustomRepresentation",
]


class CustomRepresentation(Representation):
    """A custom representation module with minimal implementation."""

    def __init__(self, num_entities: int, shape: tuple[int, ...] = (2,)):  # noqa:D107
        super().__init__(max_id=num_entities, shape=shape)
        self.x = nn.Parameter(torch.rand(*shape))

    def _plain_forward(self, indices: Optional[torch.LongTensor] = None) -> torch.FloatTensor:  # noqa: D102
        n = self.max_id if indices is None else indices.shape[0]
        return self.x.unsqueeze(dim=0).repeat(n, *(1 for _ in self.shape))


class MockEvaluator(Evaluator):
    """A mock evaluator for testing early stopping."""

    def __init__(
        self,
        key: Optional[tuple[str, ExtendedTarget, RankType]] = None,
        values: Optional[Iterable[float]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.key = RankBasedMetricResults.key_from_string(s=None if key is None else ".".join((*key[1:], key[0])))
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

    def clear(self):  # noqa: D102
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
