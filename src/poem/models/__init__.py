# -*- coding: utf-8 -*-

"""Implementations of various knowledge graph embedding models.

+---------------------------+---------------------------------------------+
| Model Name                | Reference                                   |
|                           |                                             |
+===========================+=============================================+
| ComplEx                   | :py:class:`poem.models.ComplEx`             |
+---------------------------+---------------------------------------------+
| ComplExLiteral            | :py:class:`poem.models.ComplexLiteralCWA`   |
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
| NTN                       | :py:class:`poem.models.NTN`                 |
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
| SimplE                    | :py:class:`poem.models.SimplE`              |
+---------------------------+---------------------------------------------+
| Unstructured Model (UM)   | :py:class:`poem.models.UnstructuredModel`   |
+---------------------------+---------------------------------------------+
"""

from .base import BaseModule
from .multimodal import ComplexLiteralCWA, DistMultLiteral
from .unimodal import (
    ComplEx, ConvKB, DistMult, ERMLP, HolE, NTN, RESCAL, RotatE, SimplE, StructuredEmbedding, TransD, TransE, TransH, TransR,
    UnstructuredModel,
)

__all__ = [
    'BaseModule',
    'ComplEx',
    'ComplexLiteralCWA',
    'ConvKB',
    'DistMult',
    'DistMultLiteral',
    'ERMLP',
    'HolE',
    'NTN',
    'RESCAL',
    'RotatE',
    'SimplE',
    'StructuredEmbedding',
    'TransD',
    'TransE',
    'TransH',
    'TransR',
    'UnstructuredModel',
]
