# -*- coding: utf-8 -*-

"""Negative sampling.

=========  ================================
Name       Reference
=========  ================================
basic      :class:`poem.sampling.basic`
bernoulli  :class:`poem.sampling.bernoulli`
=========  ================================

.. note:: This table can be re-generated with ``poem ls samplers -f rst``
"""

from typing import Type, Union

from .basic_negative_sampler import BasicNegativeSampler
from .bern_negative_sampler import BernoulliNegativeSampler
from .negative_sampler import NegativeSampler
from ..utils import get_cls

__all__ = [
    'NegativeSampler',
    'BasicNegativeSampler',
    'BernoulliNegativeSampler',
    'negative_samplers',
    'get_negative_sampler_cls',
]

#: A mapping of negative samplers' names to their implementations
negative_samplers = {
    'basic': BasicNegativeSampler,
    'bernoulli': BernoulliNegativeSampler,
}


def get_negative_sampler_cls(query: Union[None, str, Type[NegativeSampler]]) -> Type[NegativeSampler]:
    """Get the negative sampler class."""
    return get_cls(
        query,
        base=NegativeSampler,
        lookup_dict=negative_samplers,
        default=BasicNegativeSampler,
    )
