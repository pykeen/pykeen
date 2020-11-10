# -*- coding: utf-8 -*-

"""Constants for PyKEEN."""

import os
import pathlib

__all__ = [
    'PYKEEN_HOME',
]

PYKEEN_HOME = pathlib.Path(os.environ.get('PYKEEN_HOME')) or pathlib.Path("~", ".pykeen").expanduser()
