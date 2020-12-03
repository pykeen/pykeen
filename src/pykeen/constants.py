# -*- coding: utf-8 -*-

"""Constants for PyKEEN."""

import pystow

__all__ = [
    'PYKEEN_HOME',
    'PYKEEN_BENCHMARK_HOME',
]

PYKEEN_HOME = pystow.get('pykeen')
PYKEEN_BENCHMARK_HOME = pystow.get('pykeen', 'benchmarking')
