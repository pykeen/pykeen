# -*- coding: utf-8 -*-

"""Inductive models in PyKEEN."""

from class_resolver import ClassResolver

from .base import (
    DisjointInductivePathDataset,
    EagerInductiveDataset,
    InductiveDataset,
    LazyInductiveDataset,
    UnpackedRemoteDisjointInductiveDataset,
)
from .ilp_teru import InductiveFB15k237, InductiveNELL, InductiveWN18RR
from .ilpc2022 import ILPC2022Large, ILPC2022Small

__all__ = [
    # Base class
    "InductiveDataset",
    # Mid-level classes
    "EagerInductiveDataset",
    "LazyInductiveDataset",
    "DisjointInductivePathDataset",
    "UnpackedRemoteDisjointInductiveDataset",
    # Datasets
    "InductiveFB15k237",
    "InductiveWN18RR",
    "InductiveNELL",
    "ILPC2022Large",
    "ILPC2022Small",
]

inductive_dataset_resolver: ClassResolver[InductiveDataset] = ClassResolver.from_subclasses(
    InductiveDataset,
    skip={
        EagerInductiveDataset,
        LazyInductiveDataset,
        DisjointInductivePathDataset,
        UnpackedRemoteDisjointInductiveDataset,
    },
)
