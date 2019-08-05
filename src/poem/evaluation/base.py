# -*- coding: utf-8 -*-

"""Basic structure of a evaluator."""

from abc import ABC, abstractmethod
from typing import Mapping, Optional

import numpy as np

from ..models.base import BaseModule

__all__ = [
    'Evaluator',
]


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
    def evaluate(self, triples: np.ndarray):
        """Evaluate the triples."""
        raise NotImplementedError
