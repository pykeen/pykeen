# -*- coding: utf-8 -*-

"""Evaluators.

=========  =============================================
Name       Reference
=========  =============================================
rankbased  :class:`pykeen.evaluation.RankBasedEvaluator`
sklearn    :class:`pykeen.evaluation.SklearnEvaluator`
=========  =============================================

.. note:: This table can be re-generated with ``pykeen ls evaluators -f rst``

=========  =================================================
Name       Reference
=========  =================================================
rankbased  :class:`pykeen.evaluation.RankBasedMetricResults`
sklearn    :class:`pykeen.evaluation.SklearnMetricResults`
=========  =================================================

.. note:: This table can be re-generated with ``pykeen ls metrics -f rst``

References
----------
.. [berrendorf2020] Max Berrendorf, Evgeniy Faerman, Laurent Vermue, Volker Tresp (2020) `Interpretable and Fair
   Comparison of Link Prediction or Entity Alignment Methods with Adjusted Mean Rank
   <https://arxiv.org/abs/2002.06914>`_.
"""

import dataclasses
from typing import Set, Type

from class_resolver import Resolver

from .evaluator import Evaluator, MetricResults, evaluate
from .rank_based_evaluator import RankBasedEvaluator, RankBasedMetricResults
from .sklearn import SklearnEvaluator, SklearnMetricResults

__all__ = [
    'evaluate',
    'Evaluator',
    'MetricResults',
    'RankBasedEvaluator',
    'RankBasedMetricResults',
    'SklearnEvaluator',
    'SklearnMetricResults',
    'evaluator_resolver',
    'metric_resolver',
    'get_metric_list',
]

_EVALUATOR_SUFFIX = 'Evaluator'
_EVALUATORS: Set[Type[Evaluator]] = {
    RankBasedEvaluator,
    SklearnEvaluator,
}
evaluator_resolver = Resolver(
    _EVALUATORS,
    base=Evaluator,  # type: ignore
    suffix=_EVALUATOR_SUFFIX,
    default=RankBasedEvaluator,
)

_METRICS_SUFFIX = 'MetricResults'
_METRICS: Set[Type[MetricResults]] = {
    RankBasedMetricResults,
    SklearnMetricResults,
}
metric_resolver = Resolver(
    _METRICS,
    suffix=_METRICS_SUFFIX,
    base=MetricResults,
)


def get_metric_list():
    """Get info about all metrics across all evaluators."""
    return [
        (field, name, value)
        for name, value in metric_resolver.lookup_dict.items()
        for field in dataclasses.fields(value)
    ]
