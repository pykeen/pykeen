# -*- coding: utf-8 -*-

"""Utilities for hyper-parameter optimization."""

import random
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping

__all__ = ['AbstractHPOptimizer']


class AbstractHPOptimizer(ABC):
    """An abstract class from which all hyper-parameter optimizers should inherit."""

    @abstractmethod
    def optimize_hyperparams(self, config, path_to_kg, device, seed):
        """Run the optimizer."""

    @staticmethod
    def _sample_parameter_value(parameter_values: Mapping[int, Iterable[Any]]) -> Mapping[int, Any]:
        """Randomly subsample a dictionary whose values are iterable."""
        return {
            parameter: (
                random.choice(values)
                if isinstance(values, list) else
                values
            )
            for parameter, values in parameter_values.items()
        }
