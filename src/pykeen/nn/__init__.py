# -*- coding: utf-8 -*-

"""PyKEEN internal "nn" module."""

from class_resolver import ClassResolver

from . import init
from .message_passing import RGCNRepresentation
from .modules import (
    AutoSFInteraction,
    BoxEInteraction,
    ComplExInteraction,
    ConvEInteraction,
    ConvKBInteraction,
    CPInteraction,
    CrossEInteraction,
    DistMAInteraction,
    DistMultInteraction,
    ERMLPEInteraction,
    ERMLPInteraction,
    HolEInteraction,
    Interaction,
    KG2EInteraction,
    LineaREInteraction,
    MonotonicAffineTransformationInteraction,
    MultiLinearTuckerInteraction,
    MuREInteraction,
    NTNInteraction,
    PairREInteraction,
    ProjEInteraction,
    QuatEInteraction,
    RESCALInteraction,
    RotatEInteraction,
    SEInteraction,
    SimplEInteraction,
    TorusEInteraction,
    TransDInteraction,
    TransEInteraction,
    TransFInteraction,
    TransformerInteraction,
    TransHInteraction,
    TransRInteraction,
    TripleREInteraction,
    TuckerInteraction,
    UMInteraction,
    interaction_resolver,
)
from .node_piece import NodePieceRepresentation, TokenizationRepresentation, tokenizer_resolver
from .pyg import (
    FeaturizedMessagePassingRepresentation,
    SimpleMessagePassingRepresentation,
    TypedMessagePassingRepresentation,
)
from .representation import (
    Embedding,
    LowRankRepresentation,
    Representation,
    SubsetRepresentation,
    TextRepresentation,
    WikidataTextRepresentation,
)
from .vision import VisualRepresentation, WikidataVisualRepresentation

__all__ = [
    # REPRESENTATION
    # base
    "Representation",
    # concrete
    "Embedding",
    "FeaturizedMessagePassingRepresentation",
    "LowRankRepresentation",
    "NodePieceRepresentation",
    "RGCNRepresentation",
    "SimpleMessagePassingRepresentation",
    "SubsetRepresentation",
    "TokenizationRepresentation",
    "TypedMessagePassingRepresentation",
    "FeaturizedMessagePassingRepresentation",
    "TextRepresentation",
    "WikidataTextRepresentation",
    "VisualRepresentation",
    "WikidataVisualRepresentation",
    "tokenizer_resolver",
    "representation_resolver",
    # INITIALIZER
    "init",
    # INTERACTIONS
    "Interaction",
    # Adapter classes
    "MonotonicAffineTransformationInteraction",
    # Concrete Classes
    "AutoSFInteraction",
    "BoxEInteraction",
    "ComplExInteraction",
    "ConvEInteraction",
    "ConvKBInteraction",
    "CPInteraction",
    "CrossEInteraction",
    "DistMAInteraction",
    "DistMultInteraction",
    "ERMLPEInteraction",
    "ERMLPInteraction",
    "HolEInteraction",
    "KG2EInteraction",
    "LineaREInteraction",
    "MultiLinearTuckerInteraction",
    "MuREInteraction",
    "NTNInteraction",
    "PairREInteraction",
    "ProjEInteraction",
    "QuatEInteraction",
    "RESCALInteraction",
    "RotatEInteraction",
    "SEInteraction",
    "SimplEInteraction",
    "TorusEInteraction",
    "TransDInteraction",
    "TransEInteraction",
    "TransFInteraction",
    "TransformerInteraction",
    "TransHInteraction",
    "TransRInteraction",
    "TripleREInteraction",
    "TuckerInteraction",
    "UMInteraction",
    "interaction_resolver",
]

representation_resolver: ClassResolver[Representation] = ClassResolver.from_subclasses(
    base=Representation,
    default=Embedding,
)
