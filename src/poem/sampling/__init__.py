# -*- coding: utf-8 -*-

"""Negative sampling.

+-----------------------+----------------------------------------------------+
| Negative Sampler Name | Reference                                          |
+=======================+====================================================+
| Basic                 | :py:class:`poem.sampling.BasicNegativeSampler`     |
+-----------------------+----------------------------------------------------+
| Bernoulli             | :py:class:`poem.datasets.BernoulliNegativeSampler` |
+-----------------------+----------------------------------------------------+
"""

from .basic_negative_sampler import BasicNegativeSampler
from .bern_negative_sampler import BernoulliNegativeSampler
from .negative_sampler import NegativeSampler

__all__ = [
    'NegativeSampler',
    'BasicNegativeSampler',
    'BernoulliNegativeSampler',
    'negative_samplers',
]

negative_samplers = {
    'basic': BasicNegativeSampler,
    'bernoulli': BernoulliNegativeSampler,
}
