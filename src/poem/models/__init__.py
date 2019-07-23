# -*- coding: utf-8 -*-

"""Implementations of various knowledge graph embedding models.

+---------------------------+---------------------------------------------+
| Model Name                | Reference                                   |
|                           |                                             |
+===========================+=============================================+
| ComplEx                   | :py:class:`poem.models.ComplEx`             |
+---------------------------+---------------------------------------------+
| ComplEx (CWA)             | :py:class:`poem.models.ComplexCWA`          |
+---------------------------+---------------------------------------------+
| ComplExLiteral            | :py:class:`poem.models.ComplexLiteralCWA`   |
+---------------------------+---------------------------------------------+
| DistmultLiteral           | :py:class:`poem.models.DistMultLiteral`     |
+---------------------------+---------------------------------------------+
| ConvKB                    | :py:class:`poem.models.ConvKB`              |
+---------------------------+---------------------------------------------+
| DistMult                  | :py:class:`poem.models.DistMult`            |
+---------------------------+---------------------------------------------+
| ERMLP                     | :py:class:`poem.models.ERMLP`               |
+---------------------------+---------------------------------------------+
| HolE                      | :py:class:`poem.models.HolE`                |
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
| Unstructured Model (UM)   | :py:class:`poem.models.UnstructuredModel`   |
+---------------------------+---------------------------------------------+



"""

from . import multimodal, unimodal
from .base import BaseModule
from .multimodal import *
from .unimodal import *

__all__ = ['BaseModule'] + multimodal.__all__ + unimodal.__all__
