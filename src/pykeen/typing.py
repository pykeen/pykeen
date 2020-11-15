# -*- coding: utf-8 -*-

"""Type hints for PyKEEN."""

from typing import Callable, Mapping, Sequence, TypeVar, Union

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
HeadRepresentation = TypeVar("HeadRepresentation", Representation, Sequence[Representation])  # type: ignore
RelationRepresentation = TypeVar("RelationRepresentation", Representation, Sequence[Representation])  # type: ignore
TailRepresentation = TypeVar("TailRepresentation", Representation, Sequence[Representation])  # type: ignore
