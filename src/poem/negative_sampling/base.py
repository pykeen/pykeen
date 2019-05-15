# -*- coding: utf-8 -*-

"""Basic structure for a negative sampler."""

from abc import ABC, abstractmethod

import numpy as np


class NegativeSampler(ABC):
    @abstractmethod
    def sample(self, positive_batch) -> np.ndarray:
        pass
