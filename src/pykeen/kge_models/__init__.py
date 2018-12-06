# -*- coding: utf-8 -*-

"""Implementations of various knowledge graph embedding models.

+---------------------------+---------------------------------------------------+
| Model Name                | Reference                                         |
|                           |                                                   |
+===========================+===================================================+
| TransE                    | :py:class:`pykeen.kge_models.TransE`              |
+---------------------------+---------------------------------------------------+
| TransH                    | :py:class:`pykeen.kge_models.TransH`              |
+---------------------------+---------------------------------------------------+
| TransR                    | :py:class:`pykeen.kge_models.TransR`              |
+---------------------------+---------------------------------------------------+
| TransD                    | :py:class:`pykeen.kge_models.TransD`              |
+---------------------------+---------------------------------------------------+
| ConvE                     | :py:class:`pykeen.kge_models.ConvE`               |
+---------------------------+---------------------------------------------------+
| Structured Embedding (SE) | :py:class:`pykeen.kge_models.StructuredEmbedding` |
+---------------------------+---------------------------------------------------+
| Unstructured Model (UM)   | :py:class:`pykeen.kge_models.UnstructuredModel`   |
+---------------------------+---------------------------------------------------+
| RESCAL                    | :py:class:`pykeen.kge_models.RESCAL`              |
+---------------------------+---------------------------------------------------+
| ERMLP                     | :py:class:`pykeen.kge_models.ERMLP`               |
+---------------------------+---------------------------------------------------+
| DistMult                  | :py:class:`pykeen.kge_models.DistMult`            |
+---------------------------+---------------------------------------------------+
"""

from pykeen.kge_models.conv_e import ConvE  # noqa: F401
from pykeen.kge_models.distmult import DistMult  # noqa: F401
from pykeen.kge_models.ermlp import ERMLP  # noqa: F401
from pykeen.kge_models.rescal import RESCAL  # noqa: F401
from pykeen.kge_models.structured_embedding import StructuredEmbedding  # noqa: F401
from pykeen.kge_models.trans_d import TransD  # noqa: F401
from pykeen.kge_models.trans_e import TransE  # noqa: F401
from pykeen.kge_models.trans_h import TransH  # noqa: F401
from pykeen.kge_models.trans_r import TransR  # noqa: F401
from pykeen.kge_models.unstructured_model import UnstructuredModel  # noqa: F401
from pykeen.kge_models.utils import get_kge_model  # noqa: F401

__all__ = [
    'TransE',
    'TransH',
    'TransR',
    'TransD',
    'ConvE',
    'StructuredEmbedding',
    'UnstructuredModel',
    'RESCAL',
    'ERMLP',
    'DistMult',
    'get_kge_model',
]
