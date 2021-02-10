# -*- coding: utf-8 -*-

"""Type hints for PyKEEN."""

from typing import Callable, Mapping, NamedTuple, Sequence, TypeVar, Union, cast

import numpy as np
import torch

__all__ = [
    # General types
    'Hint',
    'Mutation',
    'OneOrSequence',
    # Others
    'LabeledTriples',
    'MappedTriples',
    'EntityMapping',
    'RelationMapping',
    'Initializer',
    'Normalizer',
    'Constrainer',
    'cast_constrainer',
    'InteractionFunction',
    'DeviceHint',
    'TorchRandomHint',
    'HeadRepresentation',
    'RelationRepresentation',
    'TailRepresentation',
    # Dataclasses
    'GaussianDistribution',
    'ScorePack',
]

X = TypeVar('X')
Hint = Union[None, str, X]
Mutation = Callable[[X], X]
OneOrSequence = Union[X, Sequence[X]]

LabeledTriples = np.ndarray
MappedTriples = torch.LongTensor
EntityMapping = Mapping[str, int]
RelationMapping = Mapping[str, int]

# comment: TypeVar expects none, or at least two super-classes
TensorType = TypeVar("TensorType", torch.Tensor, torch.FloatTensor)
InteractionFunction = Callable[[TensorType, TensorType, TensorType], TensorType]

Initializer = Mutation[TensorType]
Normalizer = Mutation[TensorType]
Constrainer = Mutation[TensorType]


def cast_constrainer(f) -> Constrainer:
    """Cast a constrainer function with :func:`typing.cast`."""
    return cast(Constrainer, f)


DeviceHint = Hint[torch.device]
TorchRandomHint = Hint[torch.Generator]

HeadRepresentation = TypeVar("HeadRepresentation", bound=OneOrSequence[torch.FloatTensor])
RelationRepresentation = TypeVar("RelationRepresentation", bound=OneOrSequence[torch.FloatTensor])
TailRepresentation = TypeVar("TailRepresentation", bound=OneOrSequence[torch.FloatTensor])


class GaussianDistribution(NamedTuple):
    """A gaussian distribution with diagonal covariance matrix."""

    mean: torch.FloatTensor
    diagonal_covariance: torch.FloatTensor


class ScorePack(NamedTuple):
    """A pair of result triples and scores."""

    result: torch.LongTensor
    scores: torch.FloatTensor
