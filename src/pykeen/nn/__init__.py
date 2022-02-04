# -*- coding: utf-8 -*-

"""PyKEEN internal "nn" module."""

from class_resolver import Resolver

from . import init
from .emb import Embedding, EmbeddingSpecification, RepresentationModule, SubsetRepresentationModule
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
    Interaction,
    interaction_resolver,
)
from .node_piece import NodePieceRepresentation, TokenizationRepresentationModule, tokenizer_resolver

__all__ = [
    "Embedding",
    "EmbeddingSpecification",
    "NodePieceRepresentation",
    "RepresentationModule",
    "SubsetRepresentationModule",
    "TokenizationRepresentationModule",
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


representation_resolver: Resolver[RepresentationModule] = Resolver.from_subclasses(
    base=RepresentationModule,
    default=Embedding,
)
