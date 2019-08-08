# -*- coding: utf-8 -*-

"""Basic structure of a evaluator."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
from dataclasses_json import dataclass_json

from ..models.base import BaseModule

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
    def entity_to_id(self) -> Mapping[str, int]:  # noqa: D401
        """A mapping from entities to their numeric identifiers."""
        return self.model.triples_factory.entity_to_id

    @property
    def relation_to_id(self) -> Mapping[str, int]:  # noqa: D401
        """A mapping from entities to their numeric identifiers."""
        return self.model.triples_factory.relation_to_id

    @property
    def device(self):  # noqa: D401
        """The device used by the model."""
        return self.model.device

    @abstractmethod
    def evaluate(self, triples: np.ndarray) -> MetricResults:
        """Evaluate the triples."""
        raise NotImplementedError
