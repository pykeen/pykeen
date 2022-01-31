# -*- coding: utf-8 -*-

r"""An interaction model $f:\mathcal{E} \times \mathcal{R} \times \mathcal{E} \rightarrow \mathbb{R}$ computes a
real-valued score representing the plausibility of a triple $(h,r,t) \in \mathbb{K}$ given the embeddings for the
entities and relations. In general, a larger score indicates a higher plausibility. The interpretation of the
score value is model-dependent, and usually it cannot be directly interpreted as a probability.
"""  # noqa: D205, D400

from class_resolver import Resolver, get_subclasses

from .base import EntityRelationEmbeddingModel, Model, _OldAbstractModel
from .baseline import EvaluationOnlyModel, MarginalDistributionBaseline
from .mocks import FixedModel
from .multimodal import ComplExLiteral, DistMultLiteral, DistMultLiteralGated, LiteralModel
from .nbase import ERModel, _NewAbstractModel
from .resolve import make_model, make_model_cls
from .unimodal import (
    CP,
    ERMLP,
    ERMLPE,
    KG2E,
    NTN,
    RESCAL,
    RGCN,
    SE,
    UM,
    AutoSF,
    BoxE,
    CompGCN,
    ComplEx,
    ConvE,
    ConvKB,
    CrossE,
    DistMA,
    DistMult,
    HolE,
    MuRE,
    NodePiece,
    PairRE,
    ProjE,
    QuatE,
    RotatE,
    SimplE,
    TorusE,
    TransD,
    TransE,
    TransF,
    TransH,
    TransR,
    TuckER,
)

__all__ = [
    # Base Models
    "Model",
    "_OldAbstractModel",
    "EntityRelationEmbeddingModel",
    "_NewAbstractModel",
    "ERModel",
    "LiteralModel",
    "EvaluationOnlyModel",
    # Concrete Models
    "AutoSF",
    "BoxE",
    "CompGCN",
    "ComplEx",
    "ComplExLiteral",
    "ConvE",
    "ConvKB",
    "CP",
    "CrossE",
    "DistMA",
    "DistMult",
    "DistMultLiteral",
    "DistMultLiteralGated",
    "ERMLP",
    "ERMLPE",
    "HolE",
    "KG2E",
    "FixedModel",
    "MuRE",
    "NodePiece",
    "NTN",
    "PairRE",
    "ProjE",
    "QuatE",
    "RESCAL",
    "RGCN",
    "RotatE",
    "SimplE",
    "SE",
    "TorusE",
    "TransD",
    "TransE",
    "TransF",
    "TransH",
    "TransR",
    "TuckER",
    "UM",
    # Evaluation-only models
    "MarginalDistributionBaseline",
    # Utils
    "model_resolver",
    "make_model",
    "make_model_cls",
]

model_resolver = Resolver.from_subclasses(
    base=Model,
    skip={
        # Abstract Models
        _NewAbstractModel,
        # We might be able to relax this later
        ERModel,
        LiteralModel,
        # baseline models behave differently
        EvaluationOnlyModel,
        *get_subclasses(EvaluationOnlyModel),
        # Old style models should never be looked up
        _OldAbstractModel,
        EntityRelationEmbeddingModel,
    },
)
