# -*- coding: utf-8 -*-

"""Basic structure of a evaluator."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from dataclasses_json import dataclass_json

from ..models.base import BaseModule
from ..typing import MappedTriples

__all__ = [
    'Evaluator',
    'MetricResults',
]


@dataclass_json
@dataclass
class MetricResults:
    """Results from computing metrics."""

    mean_rank: float
    mean_reciprocal_rank: float
    adjusted_mean_rank: float
    adjusted_mean_reciprocal_rank: float
    hits_at_k: Dict[int, float]


class Evaluator(ABC):
    """A rank-based evaluator for KGE models."""

    def __init__(self, model: Optional[BaseModule] = None) -> None:
        self.model = model

    @property
    def device(self):  # noqa: D401
        """The device used by the model."""
        return self.model.device

    @abstractmethod
    def evaluate(self, mapped_triples: MappedTriples, **kwargs) -> MetricResults:
        """Evaluate the mapped triples."""
        raise NotImplementedError
