# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work of Bordes *et al.*."""

from typing import Mapping

import numpy as np
import torch

from .base import NegativeSampler

__all__ = [
    'BernNegativeSampler',
]


class BernNegativeSampler(NegativeSampler):
    """A negative sampler using the algorithm from Bordes *et al.*."""

    def __init__(
            self,
            all_entities: np.ndarray,
            relations_t_p_h: Mapping,
            relations_h_p_t: Mapping,
    ) -> None:
        """Initialize the negative sampler with the given entities.

        :param all_entities:
        :param relations_t_p_h:
        :param relations_h_p_t:
        """
        super().__init__(all_entities=all_entities)
        # TODO: Set probs
        self.bern = torch.distributions.bernoulli.Bernoulli()

    def sample(self, positive_batch) -> np.ndarray:
        """Sample a negative batched based on the bern approach."""
        raise NotImplementedError
