# -*- coding: utf-8 -*-

"""Evaluators.

====  =========  ==================================
  ..  Name       Reference
====  =========  ==================================
   1  rankbased  :class:`poem.evaluators.rankbased`
====  =========  ==================================

.. note:: This table can be re-generated with ``poem ls evaluators -f rst``
"""

from typing import Type, Union

from .evaluator import Evaluator, MetricResults
from .rank_based_evaluator import RankBasedEvaluator
from ..utils import get_cls

__all__ = [
    'Evaluator',
    'MetricResults',
    'RankBasedEvaluator',
    'metrics',
    'evaluators',
    'get_evaluator_cls',
]

#: A mapping of results' names to their implementations
metrics = {
    'metric': MetricResults,
}

#: A mapping of evaluators' names to their implementations
evaluators = {
    'rankbased': RankBasedEvaluator,
}


def get_evaluator_cls(query: Union[None, str, Type[Evaluator]]) -> Type[Evaluator]:
    """Get the evaluator class."""
    return get_cls(
        query,
        base=Evaluator,
        lookup_dict=evaluators,
        default=RankBasedEvaluator,
    )
