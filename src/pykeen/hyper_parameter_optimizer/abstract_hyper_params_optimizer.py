# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

__all__ = ['AbstractHPOptimizer']


class AbstractHPOptimizer(ABC):

    @abstractmethod
    def optimize_hyperparams(self, config, path_to_kg, device, seed):
        pass
