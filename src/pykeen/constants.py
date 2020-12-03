# -*- coding: utf-8 -*-

"""Constants for PyKEEN."""

import os
import pathlib

__all__ = [
    'PYKEEN_HOME',
    'PYKEEN_BENCHMARK_HOME',
]

PYKEEN_DEFAULT_PATH = pathlib.Path.home() / '.pykeen'
PYKEEN_HOME = pathlib.Path(os.environ.get('PYKEEN_HOME') or PYKEEN_DEFAULT_PATH)
PYKEEN_HOME.mkdir(exist_ok=True, parents=True)

PYKEEN_BENCHMARK_HOME = PYKEEN_HOME / 'benchmarking'
PYKEEN_BENCHMARK_HOME.mkdir(exist_ok=True, parents=True)
