# -*- coding: utf-8 -*-

"""Constants for PyKEEN."""

import os

__all__ = [
    'PYKEEN_HOME',
]

PYKEEN_HOME = os.environ.get('PYKEEN_HOME') or os.path.join(os.path.expanduser('~'), '.pykeen')
