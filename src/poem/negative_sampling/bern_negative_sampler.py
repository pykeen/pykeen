# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work og of Bordes *et al.*."""

import numpy as np

import torch
from typing import Mapping
from .base import NegativeSampler
from ..utils import slice_triples

__all__ = [
    'BernNegativeSampler',
]


class BernNegativeSampler(NegativeSampler):
    def __init__(self, all_entities: np.ndarray, relations_t_p_h: Mapping, relations_h_p_t: Mapping):
        self.all_entities = all_entities
        self.num_entities = self.all_entities.shape[0]
        # TODO: Set probs
        self.bern = torch.distributions.bernoulli.Bernoulli()

    def sample(self, positive_batch) -> np.ndarray:
       """Sample a negative batched based on the bern approach."""
       pass
