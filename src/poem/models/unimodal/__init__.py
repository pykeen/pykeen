# -*- coding: utf-8 -*-

"""KGE models that only use triples."""

from .complex import ComplEx
from .complex_cwa import ComplexCWA
from .distmult import DistMult
from .hole import HolE
from .ermlp import ERMLP
from .rescal import RESCAL
from .rotate import RotatE
from .structured_embedding import StructuredEmbedding
from .trans_d import TransD
from .trans_e import TransE
from .trans_h import TransH
from .trans_r import TransR
from .unstructured_model import UnstructuredModel

__all__ = [
    'ComplEx',
    'ComplexCWA',
    'DistMult',
    'ERMLP',
    'HolE',
    'RESCAL',
    'StructuredEmbedding',
    'TransD',
    'TransE',
    'TransH',
    'TransR',
    'RotatE',
    'UnstructuredModel',
]
