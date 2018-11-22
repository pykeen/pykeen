# -*- coding: utf-8 -*-

import random
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping

__all__ = ['AbstractHPOptimizer']


class AbstractHPOptimizer(ABC):

    @abstractmethod
    def optimize_hyperparams(self, config, path_to_kg, device, seed):
        pass

    @staticmethod
    def _sample_parameter_value(parameter_values: Mapping[int, Iterable[Any]]) -> Mapping[int, Any]:
        return {
            parameter: (
                random.choice(values)
                if isinstance(values, list) else
                values
            )
            for parameter, values in parameter_values.items()
        }
