# -*- coding: utf-8 -*-

r"""An interaction model $f:\mathcal{E} \times \mathcal{R} \times \mathcal{E} \rightarrow \mathbb{R}$ computes a
real-valued score representing the plausibility of a triple $(h,r,t) \in \mathbb{K}$ given the embeddings for the
entities and relations. In general, a larger score indicates a higher plausibility. The interpretation of the
score value is model-dependent, and usually it cannot be directly interpreted as a probability.
"""  # noqa: D205, D400

from typing import Set, Type

from class_resolver import Resolver, get_subclasses

from .base import EntityEmbeddingModel, EntityRelationEmbeddingModel, Model, MultimodalModel, _OldAbstractModel
from .multimodal import ComplExLiteral, DistMultLiteral
from .nbase import ERModel, _NewAbstractModel
from .resolve import make_model, make_model_cls
from .unimodal import (
    ComplEx,
    ConvE,
    ConvKB,
    DistMult,
    ERMLP,
    ERMLPE,
    HolE,
    KG2E,
    MuRE,
    NTN,
    PairRE,
    ProjE,
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
    'MultimodalModel',
    # Concrete Models
    'ComplEx',
    'ComplExLiteral',
    'ConvE',
    'ConvKB',
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

_MODELS: Set[Type[Model]] = {
    subcls
    for subcls in get_subclasses(Model)  # type: ignore
    if not subcls._is_base_model
}
model_resolver = Resolver(classes=_MODELS, base=Model)  # type: ignore
