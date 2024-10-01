"""Type hints for PyKEEN."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Callable, Literal, NamedTuple, TypeVar, Union, cast

import numpy as np
import torch
from class_resolver import Hint, HintOrType, HintType
from typing_extensions import TypeAlias  # Python <=3.10

__all__ = [
    # General types
    "Hint",
    "HintType",
    "HintOrType",
    "Mutation",
    "OneOrSequence",
    # Triples
    "LabeledTriples",
    "MappedTriples",
    "EntityMapping",
    "RelationMapping",
    # Others
    "DeviceHint",
    "TorchRandomHint",
    # Tensor Functions
    "Initializer",
    "Normalizer",
    "Constrainer",
    "cast_constrainer",
    # Tensors
    "HeadRepresentation",
    "RelationRepresentation",
    "TailRepresentation",
    # Dataclasses
    "GaussianDistribution",
    # prediction targets
    "Target",
    "LABEL_HEAD",
    "LABEL_RELATION",
    "LABEL_TAIL",
    "TargetColumn",
    "COLUMN_HEAD",
    "COLUMN_RELATION",
    "COLUMN_TAIL",
    # modes
    "InductiveMode",
    "TRAINING",
    "TESTING",
    "VALIDATION",
    # entity alignment sides
    "EASide",
    "EA_SIDE_LEFT",
    "EA_SIDE_RIGHT",
    "EA_SIDES",
    # utils
    "normalize_rank_type",
    "normalize_target",
]

X = TypeVar("X")

# some PyTorch functions to not properly propagate types (e.g., float() does not return FloatTensor but Tensor)
# however it is still useful to distinguish float tensors from long ones
# to make the switch easier once PyTorch improves typing, we use a global type alias inside PyKEEN.
BoolTensor: TypeAlias = torch.Tensor  # replace by torch.BoolTensor
FloatTensor: TypeAlias = torch.Tensor  # replace by torch.FloatTensor
LongTensor: TypeAlias = torch.Tensor  # replace by torch.LongTensor

#: A function that mutates the input and returns a new object of the same type as output
Mutation = Callable[[X], X]
OneOrSequence = Union[X, Sequence[X]]

LabeledTriples = np.ndarray
MappedTriples = LongTensor
EntityMapping = Mapping[str, int]
RelationMapping = Mapping[str, int]

#: A function that can be applied to a tensor to initialize it
Initializer = Mutation[FloatTensor]
#: A function that can be applied to a tensor to normalize it
Normalizer = Mutation[FloatTensor]
#: A function that can be applied to a tensor to constrain it
Constrainer = Mutation[FloatTensor]


def cast_constrainer(f) -> Constrainer:
    """Cast a constrainer function with :func:`typing.cast`."""
    return cast(Constrainer, f)


#: A hint for a :class:`torch.device`
DeviceHint = Hint[torch.device]
#: A hint for a :class:`torch.Generator`
TorchRandomHint = Union[None, int, torch.Generator]

Representation = TypeVar("Representation", bound=OneOrSequence[FloatTensor])
#: A type variable for head representations used in :class:`pykeen.models.Model`,
#: :class:`pykeen.nn.modules.Interaction`, etc.
HeadRepresentation = TypeVar("HeadRepresentation", bound=OneOrSequence[FloatTensor])
#: A type variable for relation representations used in :class:`pykeen.models.Model`,
#: :class:`pykeen.nn.modules.Interaction`, etc.
RelationRepresentation = TypeVar("RelationRepresentation", bound=OneOrSequence[FloatTensor])
#: A type variable for tail representations used in :class:`pykeen.models.Model`,
#: :class:`pykeen.nn.modules.Interaction`, etc.
TailRepresentation = TypeVar("TailRepresentation", bound=OneOrSequence[FloatTensor])


class GaussianDistribution(NamedTuple):
    """A gaussian distribution with diagonal covariance matrix."""

    mean: FloatTensor
    diagonal_covariance: FloatTensor


Sign = Literal[-1, 1]

#: the inductive prediction and training mode
InductiveMode = Literal["training", "validation", "testing"]
TRAINING: InductiveMode = "training"
VALIDATION: InductiveMode = "validation"
TESTING: InductiveMode = "testing"

#: the prediction target
Target = Literal["head", "relation", "tail"]
LABEL_HEAD: Target = "head"
LABEL_RELATION: Target = "relation"
LABEL_TAIL: Target = "tail"


#: the prediction target index
TargetColumn = Literal[0, 1, 2]
COLUMN_HEAD: TargetColumn = 0
COLUMN_RELATION: TargetColumn = 1
COLUMN_TAIL: TargetColumn = 2

#: the rank types
RankType = Literal["optimistic", "realistic", "pessimistic"]
RANK_OPTIMISTIC: RankType = "optimistic"
RANK_REALISTIC: RankType = "realistic"
RANK_PESSIMISTIC: RankType = "pessimistic"
# RANK_TYPES: Tuple[RankType, ...] = typing.get_args(RankType) # Python >= 3.8
RANK_TYPES: tuple[RankType, ...] = (RANK_OPTIMISTIC, RANK_REALISTIC, RANK_PESSIMISTIC)
RANK_TYPE_SYNONYMS: Mapping[str, RankType] = {
    "best": RANK_OPTIMISTIC,
    "worst": RANK_PESSIMISTIC,
    "avg": RANK_REALISTIC,
    "average": RANK_REALISTIC,
}


def normalize_rank_type(rank: str | None) -> RankType:
    """Normalize a rank type."""
    if rank is None:
        return RANK_REALISTIC
    rank = rank.lower()
    rank = RANK_TYPE_SYNONYMS.get(rank, rank)
    if rank not in RANK_TYPES:
        raise ValueError(f"Invalid target={rank}. Possible values: {RANK_TYPES}")
    return cast(RankType, rank)


TargetBoth = Literal["both"]
SIDE_BOTH: TargetBoth = "both"
ExtendedTarget = Union[Target, TargetBoth]
SIDES: Collection[ExtendedTarget] = {LABEL_HEAD, LABEL_TAIL, SIDE_BOTH}
SIDE_MAPPING = {LABEL_HEAD: [LABEL_HEAD], LABEL_TAIL: [LABEL_TAIL], SIDE_BOTH: [LABEL_HEAD, LABEL_TAIL]}


def normalize_target(target: str | None) -> ExtendedTarget:
    """Normalize a prediction target side."""
    if target is None:
        return SIDE_BOTH
    if target not in SIDES:
        raise ValueError(f"Invalid target={target}. Possible values: {SIDES}")
    return cast(ExtendedTarget, target)


# entity alignment
EASide = Literal["left", "right"]
EA_SIDE_LEFT: EASide = "left"
EA_SIDE_RIGHT: EASide = "right"
EA_SIDES: tuple[EASide, EASide] = (EA_SIDE_LEFT, EA_SIDE_RIGHT)
