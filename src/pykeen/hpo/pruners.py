# -*- coding: utf-8 -*-

"""A wrapper for looking up pruners from :mod:`optuna`."""

from typing import Set, Type

from class_resolver import Resolver
from optuna.pruners import BasePruner, MedianPruner, NopPruner, PercentilePruner, SuccessiveHalvingPruner

__all__ = [
    'pruner_resolver',
]

_PRUNER_SUFFIX = 'Pruner'
_PRUNERS: Set[Type[BasePruner]] = {
    MedianPruner,
    NopPruner,
    PercentilePruner,
    SuccessiveHalvingPruner,
}
pruner_resolver = Resolver(
    _PRUNERS,
    default=MedianPruner,
    suffix=_PRUNER_SUFFIX,
    base=BasePruner,
)
