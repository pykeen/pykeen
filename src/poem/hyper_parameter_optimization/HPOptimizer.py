# -*- coding: utf-8 -*-

"""Utilities for hyper-parameter optimization."""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from torch.nn import Module

__all__ = [
    'HPOptimizerResult',
    'HPOptimizer',
]

HPOptimizerResult = Tuple[Module, List[float], Any, Any]


class HPOptimizer(ABC):
    """An abstract class from which all hyper-parameter optimizers should inherit."""

    @abstractmethod
    def optimize_hyperparams(self, *args, **kwargs) -> HPOptimizerResult:
        """Run the optimizer."""
