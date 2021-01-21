# -*- coding: utf-8 -*-

"""Type hints for PyKEEN."""

from typing import Callable, Mapping, NamedTuple, TypeVar, Union

import numpy as np
import torch

__all__ = [
    'LabeledTriples',
    'MappedTriples',
    'EntityMapping',
    'RelationMapping',
    'Initializer',
    'Normalizer',
    'Constrainer',
    'InteractionFunction',
    'DeviceHint',
    'TorchRandomHint',
    'GaussianDistribution',
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
TorchRandomHint = Union[None, int, torch.Generator]


class GaussianDistribution(NamedTuple):
    """A gaussian distribution with diagonal covariance matrix."""

    mean: torch.FloatTensor
    diagonal_covariance: torch.FloatTensor
