# -*- coding: utf-8 -*-

"""Constants for PyKEEN testing."""

import os

__all__ = [
    'HERE',
    'RESOURCES',
]

HERE = os.path.abspath(os.path.dirname(__file__))
RESOURCES = os.path.join(HERE, 'resources')
