# -*- coding: utf-8 -*-

"""Result trackers in PyKEEN."""

from typing import Any, Mapping, Optional, Sequence

from class_resolver import HintType, Resolver

from .base import ConsoleResultTracker, MultiResultTracker, PythonResultTracker, ResultTracker, TrackerHint
from .file import CSVResultTracker, FileResultTracker, JSONResultTracker
from .mlflow import MLFlowResultTracker
from .neptune import NeptuneResultTracker
from .tensorboard import TensorBoardResultTracker
from .wandb import WANDBResultTracker
from ..typing import OneOrSequence
from ..utils import upgrade_to_sequence

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

tracker_resolver = Resolver.from_subclasses(
    base=ResultTracker,
    default=ResultTracker,
    skip={FileResultTracker, MultiResultTracker},
)


def resolve_result_trackers(
    result_tracker: Optional[OneOrSequence[HintType[ResultTracker]]] = None,
    result_tracker_kwargs: Optional[OneOrSequence[Optional[Mapping[str, Any]]]] = None,
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
    result_trackers: Sequence[HintType[ResultTracker]]
    if result_tracker is None:
        result_trackers = []
    else:
        result_trackers = upgrade_to_sequence(result_tracker)

    result_tracker_kwargses: Sequence[Optional[Mapping[str, Any]]]
    if result_tracker_kwargs is None:
        result_tracker_kwargses = [None] * len(result_trackers)
    else:
        result_tracker_kwargses = upgrade_to_sequence(result_tracker_kwargs)

    if 0 < len(result_tracker_kwargses) and 0 == len(result_trackers):
        raise ValueError("Kwargs were given but no result trackers")
    elif 1 == len(result_tracker_kwargses) == 1 and 1 < len(result_trackers):
        result_tracker_kwargses = list(result_tracker_kwargses) * len(result_trackers)
    elif len(result_tracker_kwargses) != len(result_trackers):
        raise ValueError("Mismatch in number number of trackers and kwargs")

    trackers = [
        tracker_resolver.make(query=_result_tracker, pos_kwargs=_result_tracker_kwargs)
        for _result_tracker, _result_tracker_kwargs in zip(result_trackers, result_tracker_kwargses)
    ]
    # always add a Python result tracker for storing the configuration
    trackers.append(PythonResultTracker(store_metrics=False))

    return MultiResultTracker(trackers=trackers)
