# -*- coding: utf-8 -*-

"""Evaluation."""

from typing import List, Tuple, Type

from class_resolver import ClassResolver

from .classification_evaluator import ClassificationEvaluator, ClassificationMetricResults
from .evaluator import Evaluator, MetricResults, evaluate
from .rank_based_evaluator import RankBasedEvaluator, RankBasedMetricResults
from .utils import MetricAnnotation, ValueRange

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
    "MetricAnnotation",
    "ValueRange",
]

evaluator_resolver: ClassResolver[Evaluator] = ClassResolver.from_subclasses(
    base=Evaluator,
    default=RankBasedEvaluator,
)

metric_resolver: ClassResolver[MetricResults] = ClassResolver.from_subclasses(MetricResults)


def get_metric_list() -> List[Tuple[str, MetricAnnotation, Type[MetricResults]]]:
    """Get info about all metrics across all evaluators."""
    return [
        (key, metadata, resolver_cls)
        for resolver_cls in metric_resolver.lookup_dict.values()
        for key, metadata in resolver_cls.metrics.items()
    ]
