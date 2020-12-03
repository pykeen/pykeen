# -*- coding: utf-8 -*-

"""Constants for PyKEEN."""

import pystow

__all__ = [
    'PYKEEN_HOME',
    'PYKEEN_EXPERIMENTS',
]

PYKEEN_HOME = pystow.get('pykeen')
PYKEEN_EXPERIMENTS = pystow.get('pykeen', 'experiments')
