# -*- coding: utf-8 -*-

"""Testing constants for PyKEEN."""

import logging
import os

__all__ = [
    'HERE',
    'RESOURCES_DIRECTORY',
    'TEST_DATA',
]

logging.basicConfig(level=logging.INFO)
logging.getLogger('poem').setLevel(logging.INFO)

HERE = os.path.abspath(os.path.dirname(__file__))
RESOURCES_DIRECTORY = os.path.join(HERE, 'resources')
TEST_DATA = os.path.join(RESOURCES_DIRECTORY, 'test.txt')
