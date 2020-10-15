# -*- coding: utf-8 -*-

"""A wrapper for looking up pruners from :mod:`optuna`."""

from typing import Mapping, Set, Type, Union

from optuna.pruners import BasePruner, MedianPruner, NopPruner, PercentilePruner, SuccessiveHalvingPruner

from ..utils import get_cls, normalize_string

__all__ = [
    'pruners',
    'get_pruner_cls',
]

_PRUNER_SUFFIX = 'Pruner'
_PRUNERS: Set[Type[BasePruner]] = {
    MedianPruner,
    NopPruner,
    PercentilePruner,
    SuccessiveHalvingPruner,
}

#: A mapping of HPO pruners' names to their implementations
pruners: Mapping[str, Type[BasePruner]] = {
    normalize_string(cls.__name__, suffix=_PRUNER_SUFFIX): cls
    for cls in _PRUNERS
}


def get_pruner_cls(query: Union[None, str, Type[BasePruner]]) -> Type[BasePruner]:
    """Get the pruner class."""
    return get_cls(
        query,
        base=BasePruner,
        lookup_dict=pruners,
        default=MedianPruner,
        suffix=_PRUNER_SUFFIX,
    )
