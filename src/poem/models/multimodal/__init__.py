# -*- coding: utf-8 -*-

"""Multimodal KGE Models.

.. [agustinus2018] Agustinus, K., *et al.* (2018) `Incorporating literals into knowledge graph embeddings.
                   <https://arxiv.org/pdf/1802.00934.pdf>`_ arXiv preprint arXiv:1802.00934.
"""

from .base_module import MultimodalBaseModule
from .complex_literal_cwa import ComplExLiteralCWA
from .distmult_literal_e_owa import DistMultLiteral

__all__ = [
    'MultimodalBaseModule',
    'ComplExLiteralCWA',
    'DistMultLiteral',
]
