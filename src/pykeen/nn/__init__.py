# -*- coding: utf-8 -*-

"""PyKEEN internal "nn" module."""

from class_resolver import ClassResolver

from . import init
from .representation import Embedding, EmbeddingSpecification, Representation, SubsetRepresentation
from .modules import (
    AutoSFInteraction,
    BoxEInteraction,
    ComplExInteraction,
    ConvEInteraction,
    ConvKBInteraction,
    CrossEInteraction,
    DistMAInteraction,
    DistMultInteraction,
    ERMLPEInteraction,
    ERMLPInteraction,
    HolEInteraction,
    Interaction,
    KG2EInteraction,
    MonotonicAffineTransformationInteraction,
    MuREInteraction,
    NTNInteraction,
    PairREInteraction,
    ProjEInteraction,
    RESCALInteraction,
    RotatEInteraction,
    SEInteraction,
    SimplEInteraction,
    TorusEInteraction,
    TransDInteraction,
    TransEInteraction,
    TransFInteraction,
    TransHInteraction,
    TransRInteraction,
    TripleREInteraction,
    TuckerInteraction,
    UMInteraction,
    interaction_resolver,
)
from .node_piece import NodePieceRepresentation, TokenizationRepresentation, tokenizer_resolver

__all__ = [
    "Embedding",
    "EmbeddingSpecification",
    "NodePieceRepresentation",
    "Representation",
    "SubsetRepresentation",
    "TokenizationRepresentation",
    "init",
    "Interaction",
    "interaction_resolver",
    "tokenizer_resolver",
    "representation_resolver",
    # Adapter classes
    "MonotonicAffineTransformationInteraction",
    # Concrete Classes
    "AutoSFInteraction",
    "BoxEInteraction",
    "ComplExInteraction",
    "ConvEInteraction",
    "ConvKBInteraction",
    "CrossEInteraction",
    "DistMultInteraction",
    "DistMAInteraction",
    "ERMLPInteraction",
    "ERMLPEInteraction",
    "HolEInteraction",
    "KG2EInteraction",
    "MuREInteraction",
    "NTNInteraction",
    "PairREInteraction",
    "ProjEInteraction",
    "RESCALInteraction",
    "RotatEInteraction",
    "SimplEInteraction",
    "SEInteraction",
    "TorusEInteraction",
    "TransDInteraction",
    "TransEInteraction",
    "TransFInteraction",
    "TransHInteraction",
    "TransRInteraction",
    "TripleREInteraction",
    "TuckerInteraction",
    "UMInteraction",
]


representation_resolver: ClassResolver[Representation] = ClassResolver.from_subclasses(
    base=Representation,
    default=Embedding,
)
