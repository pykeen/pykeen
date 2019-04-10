# -*- coding: utf-8 -*-

"""Testing constants for PyKEEN."""

import os

__all__ = [
    'RESOURCES_DIRECTORY',
]

HERE = os.path.abspath(os.path.dirname(__file__))
RESOURCES_DIRECTORY = os.path.join(HERE, 'resources')
