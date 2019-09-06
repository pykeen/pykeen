# -*- coding: utf-8 -*-

"""Implementations of various knowledge graph embedding models.

+---------------------------+---------------------------------------------+
| Model Name                | Reference                                   |
+===========================+=============================================+
| ComplEx                   | :py:class:`poem.models.ComplEx`             |
+---------------------------+---------------------------------------------+
| ComplExLiteral            | :py:class:`poem.models.ComplexLiteralCWA`   |
+---------------------------+---------------------------------------------+
| ConvE                     | :py:class:`poem.models.ConvE`               |
+---------------------------+---------------------------------------------+
| ConvKB                    | :py:class:`poem.models.ConvKB`              |
+---------------------------+---------------------------------------------+
| DistMult                  | :py:class:`poem.models.DistMult`            |
+---------------------------+---------------------------------------------+
| DistmultLiteral           | :py:class:`poem.models.DistMultLiteral`     |
+---------------------------+---------------------------------------------+
| ER-MLP                    | :py:class:`poem.models.ERMLP`               |
+---------------------------+---------------------------------------------+
| HolE                      | :py:class:`poem.models.HolE`                |
+---------------------------+---------------------------------------------+
| KG2E                      | :py:class:`poem.models.KG2E`                |
+---------------------------+---------------------------------------------+
| NTN                       | :py:class:`poem.models.NTN`                 |
+---------------------------+---------------------------------------------+
| ProjE                     | :py:class:`poem.models.ProjE`               |
+---------------------------+---------------------------------------------+
| RESCAL                    | :py:class:`poem.models.RESCAL`              |
+---------------------------+---------------------------------------------+
| RotatE                    | :py:class:`poem.models.RotatE`              |
+---------------------------+---------------------------------------------+
| Structured Embedding (SE) | :py:class:`poem.models.StructuredEmbedding` |
+---------------------------+---------------------------------------------+
| TransD                    | :py:class:`poem.models.TransD`              |
+---------------------------+---------------------------------------------+
| TransE                    | :py:class:`poem.models.TransE`              |
+---------------------------+---------------------------------------------+
| TransH                    | :py:class:`poem.models.TransH`              |
+---------------------------+---------------------------------------------+
| TransR                    | :py:class:`poem.models.TransR`              |
+---------------------------+---------------------------------------------+
| TuckEr                    | :py:class:`poem.models.TuckEr`              |
+---------------------------+---------------------------------------------+
| SimplE                    | :py:class:`poem.models.SimplE`              |
+---------------------------+---------------------------------------------+
| Unstructured Model (UM)   | :py:class:`poem.models.UnstructuredModel`   |
+---------------------------+---------------------------------------------+
"""

from .base import BaseModule
from .multimodal import ComplexLiteralCWA, DistMultLiteral
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

__all__ = [
    'BaseModule',
    'ComplEx',
    'ComplexLiteralCWA',
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
]
