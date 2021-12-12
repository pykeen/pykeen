# -*- coding: utf-8 -*-

"""Result trackers in PyKEEN."""

from class_resolver import Resolver

from .base import ConsoleResultTracker, MultiResultTracker, ResultTracker, TrackerHint
from .file import CSVResultTracker, FileResultTracker, JSONResultTracker
from .mlflow import MLFlowResultTracker
from .neptune import NeptuneResultTracker
from .tensorboard import TensorBoardResultTracker
from .wandb import WANDBResultTracker

__all__ = [
    # Base classes
    "ResultTracker",
    "FileResultTracker",
    "MultiResultTracker",
    # Concrete classes
    "MLFlowResultTracker",
    "NeptuneResultTracker",
    "WANDBResultTracker",
    "JSONResultTracker",
    "CSVResultTracker",
    "TensorBoardResultTracker",
    "ConsoleResultTracker",
    # Utilities
    "tracker_resolver",
    "TrackerHint",
]

tracker_resolver = Resolver.from_subclasses(
    base=ResultTracker,
    skip={FileResultTracker, MultiResultTracker},
)
