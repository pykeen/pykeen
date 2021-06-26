# -*- coding: utf-8 -*-

"""Result trackers in PyKEEN."""

from class_resolver import Resolver, get_subclasses

from .base import ConsoleResultTracker, ResultTracker
from .file import CSVResultTracker, FileResultTracker, JSONResultTracker
from .mlflow import MLFlowResultTracker
from .neptune import NeptuneResultTracker
from .tensorboard import TensorBoardResultTracker
from .wandb import WANDBResultTracker

__all__ = [
    # Base classes
    'ResultTracker',
    'FileResultTracker',
    # Concrete classes
    'MLFlowResultTracker',
    'NeptuneResultTracker',
    'WANDBResultTracker',
    'JSONResultTracker',
    'CSVResultTracker',
    'TensorBoardResultTracker',
    'ConsoleResultTracker',
    # Utilities
    'tracker_resolver',
]

_RESULT_TRACKER_SUFFIX = 'ResultTracker'
_TRACKERS = [
    tracker
    for tracker in get_subclasses(ResultTracker)
    if tracker not in {FileResultTracker}
]
tracker_resolver = Resolver(_TRACKERS, base=ResultTracker, default=ResultTracker, suffix=_RESULT_TRACKER_SUFFIX)
