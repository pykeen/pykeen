# -*- coding: utf-8 -*-

"""KGE Models."""

from .multimodal import ComplexLiteralCWA, DistMultLiteral
from .unimodal import ComplexCWA, TransE

__all__ = [
    'ComplexLiteralCWA',
    'DistMultLiteral',
    'ComplexCWA',
    'TransE',
]
