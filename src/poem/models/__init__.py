# -*- coding: utf-8 -*-

"""Implementations of various knowledge graph embedding models.

===================  ========================================  ==================
Name                 Reference                                 Citation
===================  ========================================  ==================
ComplEx              :class:`poem.models.ComplEx`              [trouillon2016]_
ComplExLiteral       :class:`poem.models.ComplExLiteral`       [agustinus2018]_
ConvE                :class:`poem.models.ConvE`                [dettmers2018]_
ConvKB               :class:`poem.models.ConvKB`               [nguyen2018]_
DistMult             :class:`poem.models.DistMult`             [yang2014]_
DistMultLiteral      :class:`poem.models.DistMultLiteral`      [agustinus2018]_
ERMLP                :class:`poem.models.ERMLP`                [dong2014]_
ERMLPE               :class:`poem.models.ERMLPE`               [sharifzadeh2019]_
HolE                 :class:`poem.models.HolE`                 [nickel2016]_
KG2E                 :class:`poem.models.KG2E`                 [he2015]_
NTN                  :class:`poem.models.NTN`                  [socher2013]_
ProjE                :class:`poem.models.ProjE`                [shi2017]_
RESCAL               :class:`poem.models.RESCAL`               [nickel2011]_
R-GCN                :class:`poem.models.RGCN`                 [schlichtkrull2018]_
RotatE               :class:`poem.models.RotatE`               [sun2019]_
SimplE               :class:`poem.models.SimplE`               [kazemi2018]_
StructuredEmbedding  :class:`poem.models.StructuredEmbedding`  [bordes2011]_
TransD               :class:`poem.models.TransD`               [ji2015]_
TransE               :class:`poem.models.TransE`               [bordes2013]_
TransH               :class:`poem.models.TransH`               [wang2014]_
TransR               :class:`poem.models.TransR`               [lin2015]_
TuckER               :class:`poem.models.TuckER`               [balazevic2019]_
UnstructuredModel    :class:`poem.models.UnstructuredModel`    [bordes2014]_
===================  ========================================  ==================

.. note:: This table can be re-generated with ``poem ls models -f rst``
"""

import itertools as itt
from typing import Mapping, Type, Union

from .base import BaseModule, RegularizedModel
from .multimodal import ComplExLiteral, DistMultLiteral, MultimodalBaseModule
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

#: A mapping of models' names to their implementations
models: Mapping[str, Type[BaseModule]] = {
    cls.__name__: cls
    for cls in itt.chain(
        BaseModule.__subclasses__(),
        RegularizedModel.__subclasses__(),
        MultimodalBaseModule.__subclasses__(),
    )
    if cls not in {BaseModule, RegularizedModel, MultimodalBaseModule}
}

_MODELS = {
    normalize_string(model_name): model_cls
    for model_name, model_cls in models.items()
}


def get_model_cls(query: Union[str, Type[BaseModule]]) -> Type[BaseModule]:
    """Get the model class."""
    return get_cls(
        query,
        base=BaseModule,
        lookup_dict=_MODELS,
    )
