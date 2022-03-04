# -*- coding: utf-8 -*-

"""Evaluation."""

from typing import List, Tuple, Type

from class_resolver import ClassResolver

from .classification_evaluator import ClassificationEvaluator, ClassificationMetricResults
from .evaluator import Evaluator, MetricResults, evaluate
from .rank_based_evaluator import RankBasedEvaluator, RankBasedMetricResults
from ..metrics.utils import Metric, ValueRange

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
    "ValueRange",
]

evaluator_resolver: ClassResolver[Evaluator] = ClassResolver.from_subclasses(
    base=Evaluator,
    default=RankBasedEvaluator,
)

metric_resolver: ClassResolver[MetricResults] = ClassResolver.from_subclasses(MetricResults)


def get_metric_list() -> List[Tuple[str, Type[Metric], Type[MetricResults]]]:
    """Get info about all metrics across all evaluators."""
    return [
        (metric_key, metric_cls, resolver_cls)
        for resolver_cls in metric_resolver
        for metric_key, metric_cls in resolver_cls.metrics.items()
    ]
