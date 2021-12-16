# -*- coding: utf-8 -*-

"""Result trackers in PyKEEN."""

from typing import Any, Mapping, Optional, Union

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
    """Resolve and compose result trackers."""
    if result_tracker is None:
        result_trackers = []
    else:
        result_trackers = upgrade_to_sequence(result_tracker)

    if result_tracker_kwargs is None:
        result_tracker_kwargses = [None] * len(result_trackers)
    else:
        result_tracker_kwargses = upgrade_to_sequence(result_tracker_kwargs)

    if len(result_tracker_kwargses) == 1 and len(result_trackers) == 0:
        raise ValueError
    elif len(result_tracker_kwargses) == 1 and len(result_trackers) > 1:
        result_tracker_kwargses = list(result_tracker_kwargses) * len(result_trackers)
    elif len(result_tracker_kwargses) != len(result_trackers):
        raise ValueError("Mismatch in number number of trackers and kwarg")

    trackers = [
        tracker_resolver.make(query=_result_tracker, pos_kwargs=_result_tracker_kwargs)
        for _result_tracker, _result_tracker_kwargs in zip(result_trackers, result_tracker_kwargses)
    ]
    # always add a Python result tracker for storing the configuration
    trackers.append(PythonResultTracker(store_metrics=False))

    return MultiResultTracker(trackers=trackers)
