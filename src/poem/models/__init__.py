# -*- coding: utf-8 -*-

"""KGE Models."""

from .base import BaseModule
from .multimodal import ComplexLiteralCWA, DistMultLiteral
from .unimodal import (
    ComplEx, ComplexCWA, DistMult, ERMLP, RESCAL, StructuredEmbedding, TransD, TransE, TransH, TransR,
    UnstructuredModel,
)

__all__ = [
    'BaseModule',
    'ComplEx',
    'ComplexCWA',
    'ComplexLiteralCWA',
    'DistMult',
    'DistMultLiteral',
    'ERMLP',
    'RESCAL',
    'StructuredEmbedding',
    'TransD',
    'TransE',
    'TransH',
    'TransR',
    'UnstructuredModel',
]
