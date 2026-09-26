"""Inductive models in PyKEEN."""

from class_resolver import ClassResolver

from .base import (
    DisjointInductivePathDataset,
    DisjointInductiveSourceDataset,
    EagerInductiveDataset,
    InductiveDataset,
    LazyInductiveDataset,
    UnpackedRemoteDisjointInductiveDataset,
    Version,
)
from .ilp_teru import InductiveFB15k237, InductiveNELL, InductiveWN18RR
from .ilpc2022 import ILPC2022Large, ILPC2022Small

__all__ = [
    "DisjointInductivePathDataset",
    "DisjointInductiveSourceDataset",
    "EagerInductiveDataset",
    "ILPC2022Large",
    "ILPC2022Small",
    "InductiveDataset",
    "InductiveFB15k237",
    "InductiveNELL",
    "InductiveWN18RR",
    "LazyInductiveDataset",
    "UnpackedRemoteDisjointInductiveDataset",
    "Version",
]

#: A resolver for inductive datasets
inductive_dataset_resolver: ClassResolver[InductiveDataset] = ClassResolver.from_subclasses(
    InductiveDataset,
    skip={
        EagerInductiveDataset,
        LazyInductiveDataset,
        DisjointInductivePathDataset,
        UnpackedRemoteDisjointInductiveDataset,
        DisjointInductiveSourceDataset,
    },
)
