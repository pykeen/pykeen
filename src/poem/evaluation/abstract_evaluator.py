# -*- coding: utf-8 -*-

"""Basic structure of a evaluator."""

from abc import ABC

from poem.pipeline import EvaluatorConfig
import numpy as np


class AbstractEvalutor(ABC):
    """."""

    def __init__(self, evaluator_config: EvaluatorConfig):
        self.evaluator_config = evaluator_config

    def evaluate(self, test_triples: np.ndarray):
        """."""
        pass