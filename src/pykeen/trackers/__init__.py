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
    result_tracker: Union[None, OneOrSequence[HintType[ResultTracker]]] = None,
    result_tracker_kwargs: Optional[OneOrSequence[Optional[Mapping[str, Any]]]] = None,
) -> MultiResultTracker:
    """Resolve and compose result trackers."""
    if result_tracker is None:
        result_tracker = []
    result_tracker = upgrade_to_sequence(result_tracker)
    if result_tracker_kwargs is None:
        result_tracker_kwargs = [None] * len(result_tracker)
    result_tracker_kwargs = upgrade_to_sequence(result_tracker_kwargs)
    if len(result_tracker_kwargs) == 1 and len(result_tracker) > 1:
        result_tracker_kwargs = list(result_tracker_kwargs) * len(result_tracker)
    return MultiResultTracker(
        trackers=[
            tracker_resolver.make(query=_result_tracker, pos_kwargs=_result_tracker_kwargs)
            for _result_tracker, _result_tracker_kwargs in zip(result_tracker, result_tracker_kwargs)
        ]
        + [PythonResultTracker(store_metrics=False)]  # always add a Python result tracker for storing the configuration
    )
