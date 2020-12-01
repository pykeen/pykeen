# -*- coding: utf-8 -*-

"""Type hints for PyKEEN."""

from typing import Callable, Mapping, TypeVar, Union

import numpy as np
import torch

__all__ = [
    'LabeledTriples',
    'MappedTriples',
    'EntityMapping',
    'RelationMapping',
    'InteractionFunction',
    'DeviceHint',
]

LabeledTriples = np.ndarray
MappedTriples = torch.LongTensor
EntityMapping = Mapping[str, int]
RelationMapping = Mapping[str, int]

# comment: TypeVar expects none, or at least two super-classes
TensorType = TypeVar("TensorType", torch.Tensor, torch.FloatTensor)
InteractionFunction = Callable[[TensorType, TensorType, TensorType], TensorType]
Initializer = Callable[[TensorType], TensorType]
Normalizer = Callable[[TensorType], TensorType]
Constrainer = Callable[[TensorType], TensorType]

DeviceHint = Union[None, str, torch.device]
