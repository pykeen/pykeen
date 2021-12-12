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
    "ResultTracker",
    "FileResultTracker",
    # Concrete classes
    "MLFlowResultTracker",
    "NeptuneResultTracker",
    "WANDBResultTracker",
    "JSONResultTracker",
    "CSVResultTracker",
    "PythonResultTracker",
    "TensorBoardResultTracker",
    "ConsoleResultTracker",
    # Utilities
    "tracker_resolver",
]

tracker_resolver = Resolver.from_subclasses(
    base=ResultTracker,
    default=ResultTracker,
    skip={FileResultTracker},
)
