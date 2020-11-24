# -*- coding: utf-8 -*-

"""Constants for PyKEEN."""

import os
import pathlib

__all__ = [
    'PYKEEN_HOME',
    'PYKEEN_DEFAULT_CHECKPOINT_DIR',
]

PYKEEN_HOME = os.environ.get('PYKEEN_HOME') or os.path.join(os.path.expanduser('~'), '.pykeen')
PYKEEN_DEFAULT_CHECKPOINT = "PyKEEN_just_saved_my_day.pt"

PYKEEN_DEFAULT_CHECKPOINT_DIR = pathlib.Path(PYKEEN_HOME).joinpath("checkpoints")
PYKEEN_DEFAULT_CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
