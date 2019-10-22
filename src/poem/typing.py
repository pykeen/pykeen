# -*- coding: utf-8 -*-

"""Type hints for POEM."""

from typing import Mapping

import numpy as np
import torch
from torch import nn

__all__ = [
    'LabeledTriples',
    'MappedTriples',
    'Loss',
    'EntityMapping',
    'RelationMapping',
]

LabeledTriples = np.ndarray
MappedTriples = torch.LongTensor
Loss = nn.modules.loss._Loss
EntityMapping = Mapping[str, int]
RelationMapping = Mapping[str, int]
