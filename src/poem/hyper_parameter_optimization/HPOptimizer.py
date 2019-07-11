# -*- coding: utf-8 -*-

"""Utilities for hyper-parameter optimization."""

import random
from abc import ABC, abstractmethod
from torch.nn import Module
from typing import Any, Iterable, List, Mapping, Tuple

__all__ = [
    'HPOptimizerResult',
    'HPOptimizer',
]

HPOptimizerResult = Tuple[Module, List[float], Any, Any, Any, Any]


class HPOptimizer(ABC):
    """An abstract class from which all hyper-parameter optimizers should inherit."""

    @abstractmethod
    def optimize_hyperparams(self, *args, **kwargs) -> HPOptimizerResult:
        """Run the optimizer."""

