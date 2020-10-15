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

from typing import Mapping, Set, Type, Union

from optuna.samplers import BaseSampler, GridSampler, RandomSampler, TPESampler

from ..utils import get_cls, normalize_string

__all__ = [
    'samplers',
    'get_sampler_cls',
]

_SAMPLER_SUFFIX = 'Sampler'
_SAMPLERS: Set[Type[BaseSampler]] = {
    RandomSampler,
    TPESampler,
    GridSampler,
}

#: A mapping of HPO samplers' names to their implementations
samplers: Mapping[str, Type[BaseSampler]] = {
    normalize_string(cls.__name__, suffix=_SAMPLER_SUFFIX): cls
    for cls in _SAMPLERS
}


def get_sampler_cls(query: Union[None, str, Type[BaseSampler]]) -> Type[BaseSampler]:
    """Get the sampler class."""
    return get_cls(
        query,
        base=BaseSampler,
        lookup_dict=samplers,
        default=TPESampler,
        suffix=_SAMPLER_SUFFIX,
    )
