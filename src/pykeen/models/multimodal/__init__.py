"""Multimodal KGE Models."""

from .base import LiteralModel
from .complex_literal import ComplExLiteral
from .distmult_literal import DistMultLiteral
from .distmult_literal_gated import DistMultLiteralGated

__all__ = [
    "LiteralModel",
    "ComplExLiteral",
    "DistMultLiteral",
    "DistMultLiteralGated",
]
