# -*- coding: utf-8 -*-

"""Implementations of various knowledge graph embedding models.

===================  ==========================================  ====================
Name                 Reference                                   Citation
===================  ==========================================  ====================
ComplEx              :class:`pykeen.models.ComplEx`              [trouillon2016]_
ComplExLiteral       :class:`pykeen.models.ComplExLiteral`       [agustinus2018]_
ConvE                :class:`pykeen.models.ConvE`                [dettmers2018]_
ConvKB               :class:`pykeen.models.ConvKB`               [nguyen2018]_
DistMult             :class:`pykeen.models.DistMult`             [yang2014]_
DistMultLiteral      :class:`pykeen.models.DistMultLiteral`      [agustinus2018]_
ERMLP                :class:`pykeen.models.ERMLP`                [dong2014]_
ERMLPE               :class:`pykeen.models.ERMLPE`               [sharifzadeh2019]_
HolE                 :class:`pykeen.models.HolE`                 [nickel2016]_
KG2E                 :class:`pykeen.models.KG2E`                 [he2015]_
NTN                  :class:`pykeen.models.NTN`                  [socher2013]_
ProjE                :class:`pykeen.models.ProjE`                [shi2017]_
RESCAL               :class:`pykeen.models.RESCAL`               [nickel2011]_
RGCN                 :class:`pykeen.models.RGCN`                 [schlichtkrull2018]_
RotatE               :class:`pykeen.models.RotatE`               [sun2019]_
SimplE               :class:`pykeen.models.SimplE`               [kazemi2018]_
StructuredEmbedding  :class:`pykeen.models.StructuredEmbedding`  [bordes2011]_
TransD               :class:`pykeen.models.TransD`               [ji2015]_
TransE               :class:`pykeen.models.TransE`               [bordes2013]_
TransH               :class:`pykeen.models.TransH`               [wang2014]_
TransR               :class:`pykeen.models.TransR`               [lin2015]_
TuckER               :class:`pykeen.models.TuckER`               [balazevic2019]_
UnstructuredModel    :class:`pykeen.models.UnstructuredModel`    [bordes2014]_
===================  ==========================================  ====================

.. note:: This table can be re-generated with ``pykeen ls models -f rst``
"""

from typing import Mapping, Set, Type, Union

from .base import EntityEmbeddingModel, EntityRelationEmbeddingModel, Model
from .multimodal import ComplExLiteral, DistMultLiteral, MultimodalModel
from .unimodal import (
    ComplEx,
    ConvE,
    ConvKB,
    DistMult,
    ERMLP,
    ERMLPE,
    HAKE,
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
    'HAKE',
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


def _recur(c):
    for sc in c.__subclasses__():
        yield sc
        yield from _recur(sc)


_MODELS: Set[Type[Model]] = {
    cls
    for cls in _recur(Model)
    if cls not in {Model, MultimodalModel, EntityRelationEmbeddingModel, EntityEmbeddingModel}
}

#: A mapping of models' names to their implementations
models: Mapping[str, Type[Model]] = {
    normalize_string(cls.__name__): cls
    for cls in _MODELS
}


def get_model_cls(query: Union[str, Type[Model]]) -> Type[Model]:
    """Get the model class."""
    return get_cls(
        query,
        base=Model,
        lookup_dict=models,
    )
