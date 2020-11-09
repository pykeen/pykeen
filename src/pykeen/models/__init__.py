# -*- coding: utf-8 -*-

r"""An interaction model $f:\mathcal{E} \times \mathcal{R} \times \mathcal{E} \rightarrow \mathbb{R}$ computes a
real-valued score representing the plausibility of a triple $(h,r,t) \in \mathbb{K}$ given the embeddings for the
entities and relations. In general, a larger score indicates a higher plausibility. The interpretation of the
score value is model-dependent, and usually it cannot be directly interpreted as a probability.
"""  # noqa: D205, D400

import inspect
from typing import Mapping, Set, Type, Union

from .base import EntityEmbeddingModel, EntityRelationEmbeddingModel, Model, MultimodalModel  # noqa:F401
from .multimodal import ComplExLiteral, DistMultLiteral
from .unimodal import (
    ComplEx,
    ConvE,
    ConvKB,
    DistMult,
    ERMLP,
    ERMLPE,
    HolE,
    KG2E,
    NTN,
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
from ..utils import get_cls, normalize_string

__all__ = [
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
    'NTN',
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
    'models',
    'get_model_cls',
]


def _concrete_subclasses(cls):
    for subcls in cls.__subclasses__():
        if not inspect.isabstract(subcls):
            yield subcls
        yield from _concrete_subclasses(subcls)


_MODELS: Set[Type[Model]] = set(_concrete_subclasses(Model))

#: A mapping of models' names to their implementations
models: Mapping[str, Type[Model]] = {
    normalize_string(cls.__name__): cls
    for cls in _MODELS
}


def get_model_cls(query: Union[str, Type[Model]]) -> Type[Model]:
    """Look up a model class by name (case/punctuation insensitive) in :data:`pykeen.models.models`.

    :param query: The name of the model (case insensitive, punctuation insensitive).
    :return: The model class
    """
    return get_cls(
        query,
        base=Model,
        lookup_dict=models,
    )
