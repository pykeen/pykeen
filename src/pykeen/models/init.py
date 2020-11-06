# -*- coding: utf-8 -*-

"""Deprecated wrapper for :mod:`pykeen.nn.init`."""

import warnings

from ..nn.init import embedding_xavier_normal_, embedding_xavier_uniform_, xavier_normal_, xavier_uniform_

__all__ = [
    'xavier_uniform_',
    'embedding_xavier_uniform_',
    'xavier_normal_',
    'embedding_xavier_normal_',
]

warnings.warn('use pykeen.nn.init instead of pykeen.models.init', DeprecationWarning)
