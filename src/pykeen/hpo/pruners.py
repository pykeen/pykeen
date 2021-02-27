# -*- coding: utf-8 -*-

"""A wrapper for looking up pruners from :mod:`optuna`."""

from typing import Set, Type

from optuna.pruners import BasePruner, MedianPruner, NopPruner, PercentilePruner, SuccessiveHalvingPruner

from ..utils import Resolver

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
