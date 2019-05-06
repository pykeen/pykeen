# -*- coding: utf-8 -*-

"""Basic structure of a evaluator."""

from abc import ABC

from dataclasses import dataclass
from typing import Dict
import torch.nn as nn
import numpy as np

@dataclass
class EvaluatorConfig:
    """."""
    config: Dict
    kge_model: nn.Module
    entity_to_id: Dict[str, int]
    relation_to_id: Dict[str, int]
    training_triples: np.ndarray = None

class AbstractEvalutor(ABC):
    """."""

    def __init__(self, evaluator_config: EvaluatorConfig):
        self.evaluator_config = evaluator_config

    def evaluate(self, test_triples: np.ndarray):
        """."""
        pass