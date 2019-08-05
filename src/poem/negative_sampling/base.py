# -*- coding: utf-8 -*-

"""Basic structure for a negative sampler."""

from abc import ABC, abstractmethod

import numpy as np


class NegativeSampler(ABC):
    """A negative sampler."""

    def __init__(self, all_entities) -> None:
        """Initialize the negative sampler with the given entities."""
        self.all_entities = all_entities

    @property
    def num_entities(self) -> int:  # noqa: D401
        """The number of entities to sample from."""
        return self.all_entities.shape[0]

    @abstractmethod
    def sample(self, positive_batch) -> np.ndarray:
        """Generate negative samples from the positive batch."""
        raise NotImplementedError
