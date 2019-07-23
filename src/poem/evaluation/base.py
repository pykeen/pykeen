# -*- coding: utf-8 -*-

"""Basic structure of a evaluator."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..models.base import BaseModule

__all__ = [
    'Evaluator',
]


class Evaluator(ABC):

    def __init__(self, model: Optional[BaseModule] = None) -> None:
        self.model = model

    @property
    def entity_to_id(self):
        return self.model.triples_factory.entity_to_id

    def relation_to_id(self):
        return self.model.triples_factory.relation_to_id

    @property
    def device(self):
        return self.model.device

    @abstractmethod
    def evaluate(self, triples: np.ndarray):
        raise NotImplementedError
