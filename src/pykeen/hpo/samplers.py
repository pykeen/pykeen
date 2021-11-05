# -*- coding: utf-8 -*-

"""A wrapper for looking up samplers from :mod:`optuna`.

======  ======================================
Name    Reference
======  ======================================
grid    :class:`optuna.samplers.GridSampler`
random  :class:`optuna.samplers.RandomSampler`
tpe     :class:`optuna.samplers.TPESampler`
======  ======================================

.. note:: This table can be re-generated with ``pykeen ls hpo-samplers -f rst``
"""

# TODO update docs with table and CLI wtih generator

from typing import Set, Type

from class_resolver import Resolver
from optuna.samplers import BaseSampler, GridSampler, RandomSampler, TPESampler

__all__ = [
    "sampler_resolver",
]

_SAMPLER_SUFFIX = "Sampler"
_SAMPLERS: Set[Type[BaseSampler]] = {
    RandomSampler,
    TPESampler,
    GridSampler,
}
sampler_resolver = Resolver(
    _SAMPLERS,
    base=BaseSampler,
    default=TPESampler,
    suffix=_SAMPLER_SUFFIX,
)
