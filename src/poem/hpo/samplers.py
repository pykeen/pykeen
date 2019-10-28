# -*- coding: utf-8 -*-

"""A wrapper for looking up samplers from :mod:`optuna`."""

from typing import Type, Union

from optuna.samplers import BaseSampler, RandomSampler, TPESampler

from ..utils import get_cls

__all__ = [
    'samplers',
    'get_sampler_cls',
]

samplers = {
    'random': RandomSampler,
    'tpe': TPESampler,
}


def get_sampler_cls(query: Union[None, str, Type[BaseSampler]]) -> Type[BaseSampler]:
    """Get the sampler class."""
    return get_cls(
        query,
        base=BaseSampler,
        lookup_dict=samplers,
        default=TPESampler,
    )
