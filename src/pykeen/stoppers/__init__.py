# -*- coding: utf-8 -*-

"""Early stoppers."""

from typing import Collection, Mapping, Type, Union

from .early_stopping import EarlyStopper  # noqa: F401
from .stopper import NopStopper, Stopper
from ..utils import get_cls, normalize_string

__all__ = [
    'Stopper',
    'NopStopper',
    'EarlyStopper',
    'stoppers',
    'get_stopper_cls',
]

_STOPPER_SUFFIX = 'Stopper'
_STOPPERS: Collection[Type[Stopper]] = {
    NopStopper,
    EarlyStopper,
}

#: A mapping of training loops' names to their implementations
stoppers: Mapping[str, Type[Stopper]] = {
    normalize_string(cls.__name__, suffix=_STOPPER_SUFFIX): cls
    for cls in _STOPPERS
}


def get_stopper_cls(query: Union[None, str, Type[Stopper]]) -> Type[Stopper]:
    """Get the training loop class."""
    return get_cls(
        query,
        base=Stopper,
        lookup_dict=stoppers,
        default=NopStopper,
        suffix=_STOPPER_SUFFIX,
    )
