# -*- coding: utf-8 -*-

"""Evaluators.

=========  ===========================================
Name       Reference
=========  ===========================================
rankbased  :class:`poem.evaluators.RankBasedEvaluator`
=========  ===========================================

.. note:: This table can be re-generated with ``poem ls evaluators -f rst``

=========  ===============================================
Name       Reference
=========  ===============================================
rankbased  :class:`poem.evaluation.RankBasedMetricResults`
=========  ===============================================

.. note:: This table can be re-generated with ``poem ls metrics -f rst``
"""

import dataclasses
from typing import Mapping, Set, Type, Union

from .evaluator import Evaluator, MetricResults, evaluate
from .rank_based_evaluator import RankBasedEvaluator, RankBasedMetricResults
from ..utils import get_cls, normalize_string

__all__ = [
    'evaluate',
    'Evaluator',
    'MetricResults',
    'RankBasedEvaluator',
    'RankBasedMetricResults',
    'metrics',
    'evaluators',
    'get_evaluator_cls',
    'get_metric_list',
]

_EVALUATOR_SUFFIX = 'Evaluator'
_EVALUATORS: Set[Type[Evaluator]] = {
    RankBasedEvaluator,
}

#: A mapping of evaluators' names to their implementations
evaluators: Mapping[str, Type[Evaluator]] = {
    normalize_string(cls.__name__, suffix=_EVALUATOR_SUFFIX): cls
    for cls in _EVALUATORS
}


def get_evaluator_cls(query: Union[None, str, Type[Evaluator]]) -> Type[Evaluator]:
    """Get the evaluator class."""
    return get_cls(
        query,
        base=Evaluator,
        lookup_dict=evaluators,
        default=RankBasedEvaluator,
        suffix=_EVALUATOR_SUFFIX,
    )


_METRICS_SUFFIX = 'MetricResults'
_METRICS: Set[Type[MetricResults]] = {
    RankBasedMetricResults,
}

#: A mapping of results' names to their implementations
metrics: Mapping[str, Type[MetricResults]] = {
    normalize_string(cls.__name__, suffix=_METRICS_SUFFIX): cls
    for cls in _METRICS
}


def get_metric_list():
    """Get info about all metrics across all evaluators."""
    return [
        (field, name, value)
        for name, value in metrics.items()
        for field in dataclasses.fields(value)
    ]
