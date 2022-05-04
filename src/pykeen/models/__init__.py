# -*- coding: utf-8 -*-

r"""
A knowledge graph embedding model is capable of computing real-valued scores representing the plausibility
of a triple $(h,r,t) \in \mathbb{K}$, where a larger score indicates a higher plausibility. The interpretation
of the score value is model-dependent, and usually it cannot be directly interpreted as a probability.

In PyKEEN, the API of a model is defined in :class:`Model`, where the scoring function is exposed as
:meth:`Model.score_hrt`, which can be used to compute plausability scores for (a batch of) triples.
In addition, the :class:`Model` class also offers additional scoring methods, which can be used to
(efficiently) compute scores for a large number of triples sharing some parts, e.g., to compute scores
for triples `(h, r, e)` for a given `(h, r)` pair and all available entities $e \in \mathcal{E}$.

.. note ::

    The implementations of the knowledge graph embedding models provided here all operate on entity / relation
    indices rather than string representations, cf. `here <../tutorial/performance.html#entity-and-relation-ids>`_.

On top of these scoring methods, there are also corresponding prediction methods, e.g.,
:meth:`Model.predict_hrt`. These methods extend the scoring ones, by ensuring the model is in evaluation
mode, cf. :meth:`torch.nn.Module.eval`, and optionally applying a sigmoid activation on the scores to
ensure a value range of $[0, 1]$.

.. warning ::

    Depending on the model at hand, directly applying sigmoid might not always be sensible. For instance,
    distance-based interaction functions, such as :class:`pykeen.nn.modules.TransEInteraction`, result in non-positive
    scores (since they use the *negative* distance as scoring function), and thus the output of the sigmoid
    only covers the interval $[0.5, 1]$.

Most models derive from :class:`ERModel`, which is a generic implementation of a knowledge graph embedding model.
It combines a variable number of *representations* for entities and relations, cf.
:class:`pykeen.nn.representation.Representation`, and an interaction function, cf.
:class:`pykeen.nn.modules.Interaction`. The representation modules convert integer entity or relation indices to
numeric representations, e.g., vectors. The interaction function takes the representations of the head entities,
relations and tail entities as input and computes a scalar plausability score for triples.

.. note ::

    An in-depth discussion of representation modules can be found in
    `the corresponding tutorial <../tutorial/representations.html>`_.

.. note ::

    The specific models from this module, e.g., :class:`RESCAL`, package given specific entity and relation
    representations with an interaction function. For more flexible combinations, consider using
    :class:`ERModel` directly.
"""  # noqa: D205, D400

from class_resolver import ClassResolver, get_subclasses

from .base import EntityRelationEmbeddingModel, Model, _OldAbstractModel
from .baseline import EvaluationOnlyModel, MarginalDistributionBaseline, SoftInverseTripleBaseline
from .inductive import InductiveNodePiece, InductiveNodePieceGNN
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
    # Inductive Models
    "InductiveNodePiece",
    "InductiveNodePieceGNN",
    # Evaluation-only models
    "SoftInverseTripleBaseline",
    "MarginalDistributionBaseline",
    # Utils
    "model_resolver",
    "make_model",
    "make_model_cls",
]

model_resolver: ClassResolver[Model] = ClassResolver.from_subclasses(
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
