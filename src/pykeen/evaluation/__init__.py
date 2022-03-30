# -*- coding: utf-8 -*-

"""Evaluation."""

from class_resolver import ClassResolver

from .classification_evaluator import ClassificationEvaluator, ClassificationMetricResults
from .evaluator import Evaluator, MetricResults, evaluate
from .rank_based_evaluator import (
    MacroRankBasedEvaluator,
    RankBasedEvaluator,
    RankBasedMetricResults,
    SampledRankBasedEvaluator,
)

__all__ = [
    "evaluate",
    "Evaluator",
    "MetricResults",
    "RankBasedEvaluator",
    "RankBasedMetricResults",
    "MacroRankBasedEvaluator",
    "SampledRankBasedEvaluator",
    "ClassificationEvaluator",
    "ClassificationMetricResults",
    "evaluator_resolver",
    "metric_resolver",
]

evaluator_resolver: ClassResolver[Evaluator] = ClassResolver.from_subclasses(
    base=Evaluator,
    default=RankBasedEvaluator,
)

metric_resolver: ClassResolver[MetricResults] = ClassResolver.from_subclasses(MetricResults)
