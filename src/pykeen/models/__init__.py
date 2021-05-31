# -*- coding: utf-8 -*-

r"""An interaction model $f:\mathcal{E} \times \mathcal{R} \times \mathcal{E} \rightarrow \mathbb{R}$ computes a
real-valued score representing the plausibility of a triple $(h,r,t) \in \mathbb{K}$ given the embeddings for the
entities and relations. In general, a larger score indicates a higher plausibility. The interpretation of the
score value is model-dependent, and usually it cannot be directly interpreted as a probability.
"""  # noqa: D205, D400

from class_resolver import Resolver

from .base import EntityEmbeddingModel, EntityRelationEmbeddingModel, Model, _OldAbstractModel
from .multimodal import ComplExLiteral, DistMultLiteral, LiteralModel
from .nbase import ERModel, _NewAbstractModel
from .resolve import make_model, make_model_cls
from .unimodal import (
    CompGCN,
    ComplEx,
    ConvE,
    ConvKB,
    CrossE,
    DistMult,
    ERMLP,
    ERMLPE,
    HolE,
    KG2E,
    MuRE,
    NTN,
    PairRE,
    ProjE,
    QuatE,
    RESCAL,
    RGCN,
    RotatE,
    SimplE,
    StructuredEmbedding,
    TransD,
    TransE,
    TransH,
    TransR,
    TuckER,
    UnstructuredModel,
)

__all__ = [
    # Base Models
    'Model',
    '_OldAbstractModel',
    'EntityEmbeddingModel',
    'EntityRelationEmbeddingModel',
    '_NewAbstractModel',
    'ERModel',
    'LiteralModel',
    # Concrete Models
    'CompGCN',
    'ComplEx',
    'ComplExLiteral',
    'ConvE',
    'ConvKB',
    'CrossE',
    'DistMult',
    'DistMultLiteral',
    'ERMLP',
    'ERMLPE',
    'HolE',
    'KG2E',
    'MuRE',
    'NTN',
    'PairRE',
    'ProjE',
    'QuatE',
    'RESCAL',
    'RGCN',
    'RotatE',
    'SimplE',
    'StructuredEmbedding',
    'TransD',
    'TransE',
    'TransH',
    'TransR',
    'TuckER',
    'UnstructuredModel',
    # Utils
    'model_resolver',
    'make_model',
    'make_model_cls',
]

model_resolver = Resolver.from_subclasses(
    base=Model,
    skip={
        _NewAbstractModel,
        # We might be able to relax this later
        ERModel,
        LiteralModel,
        # Old style models should never be looked up
        _OldAbstractModel,
        EntityEmbeddingModel,
        EntityRelationEmbeddingModel,
    },
)
