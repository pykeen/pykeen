"""Evaluation."""

from class_resolver import ClassResolver

from .classification_evaluator import ClassificationEvaluator, ClassificationMetricResults
from .evaluation_loop import LCWAEvaluationLoop
from .evaluator import Evaluator, MetricResults
from .ogb_evaluator import OGBEvaluator
from .rank_based_evaluator import (
    MacroRankBasedEvaluator,
    RankBasedEvaluator,
    RankBasedMetricResults,
    SampledRankBasedEvaluator,
)

__all__ = [
    "Evaluator",
    "MetricResults",
    "RankBasedEvaluator",
    "RankBasedMetricResults",
    "MacroRankBasedEvaluator",
    "LCWAEvaluationLoop",
    "SampledRankBasedEvaluator",
    "OGBEvaluator",
    "ClassificationEvaluator",
    "ClassificationMetricResults",
    "evaluator_resolver",
    "metric_resolver",
]

#: A resolver for evaluators
evaluator_resolver: ClassResolver[Evaluator] = ClassResolver.from_subclasses(
    base=Evaluator,
    default=RankBasedEvaluator,
)

#: A resolver for metric results
metric_resolver: ClassResolver[MetricResults] = ClassResolver.from_subclasses(MetricResults)
