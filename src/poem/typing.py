# -*- coding: utf-8 -*-

"""Type hints for POEM."""

from typing import Callable, Mapping

import numpy as np
import torch

__all__ = [
    'LabeledTriples',
    'MappedTriples',
    'EntityMapping',
    'RelationMapping',
    'InteractionFunction',
    'SklearnMetric',
]

LabeledTriples = np.ndarray
MappedTriples = torch.LongTensor
EntityMapping = Mapping[str, int]
RelationMapping = Mapping[str, int]

InteractionFunction = Callable[[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]
SklearnMetric = Callable[[np.ndarray, np.ndarray], float]
