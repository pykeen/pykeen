# -*- coding: utf-8 -*-

"""Negative sampling.

=========  =================================================
Name       Reference
=========  =================================================
basic      :class:`pykeen.sampling.BasicNegativeSampler`
bernoulli  :class:`pykeen.sampling.BernoulliNegativeSampler`
=========  =================================================

.. note:: This table can be re-generated with ``pykeen ls samplers -f rst``
"""

from typing import Mapping, Set, Type, Union

from .basic_negative_sampler import BasicNegativeSampler
from .bernoulli_negative_sampler import BernoulliNegativeSampler
from .negative_sampler import NegativeSampler
from ..utils import get_cls, normalize_string

__all__ = [
    'NegativeSampler',
    'BasicNegativeSampler',
    'BernoulliNegativeSampler',
    'negative_samplers',
    'get_negative_sampler_cls',
]

_NEGATIVE_SAMPLER_SUFFIX = 'NegativeSampler'
_NEGATIVE_SAMPLERS: Set[Type[NegativeSampler]] = {
    BasicNegativeSampler,
    BernoulliNegativeSampler,
}

#: A mapping of negative samplers' names to their implementations
negative_samplers: Mapping[str, Type[NegativeSampler]] = {
    normalize_string(cls.__name__, suffix=_NEGATIVE_SAMPLER_SUFFIX): cls
    for cls in _NEGATIVE_SAMPLERS
}


def get_negative_sampler_cls(query: Union[None, str, Type[NegativeSampler]]) -> Type[NegativeSampler]:
    """Get the negative sampler class."""
    return get_cls(
        query,
        base=NegativeSampler,
        lookup_dict=negative_samplers,
        default=BasicNegativeSampler,
        suffix=_NEGATIVE_SAMPLER_SUFFIX,
    )
