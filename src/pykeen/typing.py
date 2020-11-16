# -*- coding: utf-8 -*-

"""Type hints for PyKEEN."""

from typing import Callable, Mapping, NamedTuple, Sequence, TypeVar, Union

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
    'DeviceHint',
    'HeadRepresentation',
    'RelationRepresentation',
    'Representation',
    'TailRepresentation',
    'GaussianDistribution',
]

LabeledTriples = np.ndarray
MappedTriples = torch.LongTensor
EntityMapping = Mapping[str, int]
RelationMapping = Mapping[str, int]

# comment: TypeVar expects none, or at least two super-classes
TensorType = TypeVar("TensorType", torch.Tensor, torch.FloatTensor)
Initializer = Callable[[TensorType], TensorType]
Normalizer = Callable[[TensorType], TensorType]
Constrainer = Callable[[TensorType], TensorType]

DeviceHint = Union[None, str, torch.device]

Representation = torch.FloatTensor
# TODO upgrade to use bound=...
# HeadRepresentation = TypeVar("HeadRepresentation", bound=Union[Representation, Sequence[Representation]])
HeadRepresentation = TypeVar("HeadRepresentation", Representation, Sequence[Representation])  # type: ignore
RelationRepresentation = TypeVar("RelationRepresentation", Representation, Sequence[Representation])  # type: ignore
TailRepresentation = TypeVar("TailRepresentation", Representation, Sequence[Representation])  # type: ignore


class GaussianDistribution(NamedTuple):
    """A gaussian distribution with diagonal covariance matrix."""

    mean: torch.FloatTensor
    diagonal_covariance: torch.FloatTensor
