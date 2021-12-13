# -*- coding: utf-8 -*-

"""Evaluation."""

import dataclasses

from class_resolver import Resolver

from .classification import ClassificationEvaluator, ClassificationMetricResults, ClassificationMetricResultsBase
from .evaluator import Evaluator, MetricResults, evaluate
from .rank_based_evaluator import RankBasedEvaluator, RankBasedMetricResults

__all__ = [
    "evaluate",
    "Evaluator",
    "MetricResults",
    "RankBasedEvaluator",
    "RankBasedMetricResults",
    "ClassificationEvaluator",
    "ClassificationMetricResults",
    "evaluator_resolver",
    "metric_resolver",
    "get_metric_list",
]

evaluator_resolver = Resolver.from_subclasses(
    base=Evaluator,  # type: ignore
    default=RankBasedEvaluator,
)

metric_resolver = Resolver.from_subclasses(
    base=MetricResults,
    default=RankBasedMetricResults,
    skip={ClassificationMetricResultsBase},
)


def get_metric_list():
    """Get info about all metrics across all evaluators."""
    return [
        (field, name, value)
        for name, value in metric_resolver.lookup_dict.items()
        for field in dataclasses.fields(value)
    ]
