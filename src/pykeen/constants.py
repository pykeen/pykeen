# -*- coding: utf-8 -*-

"""Constants for PyKEEN."""

import os

__all__ = [
    'PYKEEN_HOME',
]

PYKEEN_HOME = os.environ.get('PYKEEN_HOME') or os.path.join(os.path.expanduser('~'), '.pykeen')
DEFAULT_DROPOUT_HPO_RANGE = dict(type=float, low=0.0, high=0.5, q=0.1)
DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE = dict(type=int, low=16, high=256, q=16)
