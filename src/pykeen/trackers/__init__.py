# -*- coding: utf-8 -*-

"""Result trackers in PyKEEN."""

from typing import Mapping, Type, Union

from .base import ResultTracker
from .mlflow import MLFlowResultTracker
from .wandb import WANDBResultTracker
from ..utils import get_cls, normalize_string

__all__ = [
    'ResultTracker',
    'get_result_tracker_cls',
    'MLFlowResultTracker',
    'WANDBResultTracker',
]

_RESULT_TRACKER_SUFFIX = 'ResultTracker'

#: A mapping of trackers' names to their implementations
trackers: Mapping[str, Type[ResultTracker]] = {
    normalize_string(tracker.__name__, suffix=_RESULT_TRACKER_SUFFIX): tracker
    for tracker in ResultTracker.__subclasses__()
}


def get_result_tracker_cls(query: Union[None, str, Type[ResultTracker]]) -> Type[ResultTracker]:
    """Get the tracker class."""
    return get_cls(
        query,
        base=ResultTracker,
        lookup_dict=trackers,
        default=ResultTracker,
        suffix=_RESULT_TRACKER_SUFFIX,
    )
