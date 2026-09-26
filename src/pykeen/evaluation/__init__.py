"""Evaluation."""

from class_resolver import ClassResolver

from .classification_evaluator import ClassificationEvaluator, ClassificationMetricResults
from .evaluation_loop import EvaluationLoop, LCWAEvaluationLoop
from .evaluator import Evaluator, MetricResults
from .ogb_evaluator import OGBEvaluator, evaluate_ogb
from .rank_based_evaluator import (
    MacroRankBasedEvaluator,
    RankBasedEvaluator,
    RankBasedMetricResults,
    SampledRankBasedEvaluator,
    sample_negatives,
)

__all__ = [
    "ClassificationEvaluator",
    "ClassificationMetricResults",
    "EvaluationLoop",
    "Evaluator",
    "LCWAEvaluationLoop",
    "MacroRankBasedEvaluator",
    "MetricResults",
    "OGBEvaluator",
    "RankBasedEvaluator",
    "RankBasedMetricResults",
    "SampledRankBasedEvaluator",
    "evaluate_ogb",
    "evaluator_resolver",
    "metric_resolver",
    "sample_negatives",
]

#: A resolver for evaluators
evaluator_resolver: ClassResolver[Evaluator] = ClassResolver.from_subclasses(
    base=Evaluator,  # type: ignore[type-abstract]
    default=RankBasedEvaluator,
)

#: A resolver for metric results
metric_resolver: ClassResolver[MetricResults] = ClassResolver.from_subclasses(MetricResults)  # type: ignore[type-abstract]
