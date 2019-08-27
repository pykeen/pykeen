# -*- coding: utf-8 -*-

"""Type hints for POEM."""

from typing import Mapping, Optional

import numpy as np
import torch
from torch import nn

__all__ = [
    'LabeledTriples',
    'MappedTriples',
    'Loss',
    'OptionalLoss',
    'EntityMapping',
    'RelationMapping',
]

LabeledTriples = np.ndarray
MappedTriples = torch.LongTensor
Loss = nn.modules.loss._Loss
OptionalLoss = Optional[Loss]
EntityMapping = Mapping[str, int]
RelationMapping = Mapping[str, int]
