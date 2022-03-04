# -*- coding: utf-8 -*-

"""Result trackers in PyKEEN."""

from class_resolver import ClassResolver
from class_resolver.utils import OneOrManyHintOrType, OneOrManyOptionalKwargs

from .base import ConsoleResultTracker, MultiResultTracker, PythonResultTracker, ResultTracker, TrackerHint
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
    "PythonResultTracker",
    "TensorBoardResultTracker",
    "ConsoleResultTracker",
    # Utilities
    "tracker_resolver",
    "TrackerHint",
    "resolve_result_trackers",
]

tracker_resolver: ClassResolver[ResultTracker] = ClassResolver.from_subclasses(
    base=ResultTracker,
    default=ResultTracker,
    skip={FileResultTracker, MultiResultTracker},
)


def resolve_result_trackers(
    result_tracker: OneOrManyHintOrType[ResultTracker] = None,
    result_tracker_kwargs: OneOrManyOptionalKwargs = None,
) -> MultiResultTracker:
    """Resolve and compose result trackers.

    :param result_tracker: Either none (will result in a Python result tracker),
        a single tracker (as either a class, instance, or string for class name), or a list
        of trackers (as either a class, instance, or string for class name
    :param result_tracker_kwargs: Either none (will use all defaults), a single dictionary
        (will be used for all trackers), or a list of dictionaries with the same length
        as the result trackers
    :returns: A multi-result trackers that offloads to all contained result trackers
    """
    if result_tracker is None:
        result_tracker = []
    trackers = tracker_resolver.make_many(queries=result_tracker, kwargs=result_tracker_kwargs)
    # always add a Python result tracker for storing the configuration
    trackers.append(PythonResultTracker(store_metrics=False))
    return MultiResultTracker(trackers=trackers)
