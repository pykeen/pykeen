# -*- coding: utf-8 -*-

"""Implementations of various knowledge graph embedding models.

====  ===================  ========================================  ================
  ..  Name                 Reference                                 Citation
====  ===================  ========================================  ================
   1  ComplEx              :class:`poem.models.ComplEx`              [trouillon2016]_
   2  ComplExLiteralCWA    :class:`poem.models.ComplExLiteralCWA`    [agustinus2018]_
   3  ConvE                :class:`poem.models.ConvE`                [dettmers2018]_
   4  ConvKB               :class:`poem.models.ConvKB`               [nguyen2018]_
   5  DistMult             :class:`poem.models.DistMult`             [yang2014]_
   6  DistMultLiteral      :class:`poem.models.DistMultLiteral`      [agustinus2018]_
   7  ERMLP                :class:`poem.models.ERMLP`                [dong2014]_
   8  HolE                 :class:`poem.models.HolE`                 [nickel2016]_
   9  KG2E                 :class:`poem.models.KG2E`                 [he2015]_
  10  NTN                  :class:`poem.models.NTN`                  [socher2013]_
  11  ProjE                :class:`poem.models.ProjE`                [shi2017]_
  12  RESCAL               :class:`poem.models.RESCAL`               [nickel2011]_
  13  RotatE               :class:`poem.models.RotatE`               [sun2019]_
  14  SimplE               :class:`poem.models.SimplE`               [kazemi2018]_
  15  StructuredEmbedding  :class:`poem.models.StructuredEmbedding`  [bordes2011]_
  16  TransD               :class:`poem.models.TransD`               [ji2015]_
  17  TransE               :class:`poem.models.TransE`               [bordes2013]_
  18  TransH               :class:`poem.models.TransH`               [wang2014]_
  19  TransR               :class:`poem.models.TransR`               [lin2015]_
  20  TuckEr               :class:`poem.models.TuckEr`               [balazevic2019]_
  21  UnstructuredModel    :class:`poem.models.UnstructuredModel`    [bordes2014]_
====  ===================  ========================================  ================

.. note:: This table can be re-generated with ``poem ls models -f rst``
"""

import itertools as itt
from typing import Mapping, Type, Union

from .base import BaseModule, RegularizedModel
from .multimodal import ComplExLiteralCWA, DistMultLiteral, MultimodalBaseModule
from .unimodal import (
    ComplEx,
    ConvE,
    ConvKB,
    DistMult,
    ERMLP,
    HolE,
    KG2E,
    NTN,
    ProjE,
    RESCAL,
    RotatE,
    SimplE,
    StructuredEmbedding,
    TransD,
    TransE,
    TransH,
    TransR,
    TuckEr,
    UnstructuredModel,
)
from ..utils import get_cls, normalize_string

__all__ = [
    'ComplEx',
    'ComplExLiteralCWA',
    'ConvE',
    'ConvKB',
    'DistMult',
    'DistMultLiteral',
    'ERMLP',
    'HolE',
    'KG2E',
    'NTN',
    'ProjE',
    'RESCAL',
    'RotatE',
    'SimplE',
    'StructuredEmbedding',
    'TransD',
    'TransE',
    'TransH',
    'TransR',
    'TuckEr',
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
