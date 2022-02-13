# -*- coding: utf-8 -*-

"""Evaluation."""

from typing import List, Tuple, Type

from class_resolver import ClassResolver

from .classification_evaluator import ClassificationEvaluator, ClassificationMetricResults
from .evaluator import Evaluator, MetricResults, evaluate
from .rank_based_evaluator import RankBasedEvaluator, RankBasedMetricResults
from .utils import MetricAnnotation

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


evaluator_resolver: ClassResolver[Evaluator] = ClassResolver.from_subclasses(
    base=Evaluator,
    default=RankBasedEvaluator,
)

metric_resolver: ClassResolver[MetricResults] = ClassResolver.from_subclasses(MetricResults)


def get_metric_list() -> List[Tuple[str, MetricAnnotation, str, Type[MetricResults]]]:
    """Get info about all metrics across all evaluators."""
    return [
        (key, metadata, resolver_name, resolver_cls)
        for resolver_name, resolver_cls in metric_resolver.lookup_dict.items()
        for key, metadata in resolver_cls.metadata.items()
    ]
