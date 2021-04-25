# -*- coding: utf-8 -*-

r"""
While the implementation is slightly different than in the negative sampler, from a high-level, the
evaluation can also be thought of as applying corruption operations to triples 
$(h,r,t) \in \mathcal{T_{\text{testing}}}$ in the testing set to generate a set of corrupted triples
$\mathcal{N}(h, r, t)$. Like in training, there is the potential for known positive triples to appear in 
$\mathcal{N}(h, r, t)$, which are false negatives.

If these false negatives are ranked better than the current test triple $(h,r,t)$, other true positive
triples might be ranked worse and therefore worsen the rank-based metrics reported during evaluation.
In order to address this, [bordes2013]_ proposed the \emph{filtered} evaluation setting in which
the corrupted triples are filtered to exclude known true facts from the train, validation and test set.
Thus, the rank does not worsen when ranking a false positive entity better.

.. note ::
    In PyKEEN, we provide training and evaluation pipelines for which the set of positive triples is pre-defined.
    During hyper-parameter optimization (HPO), the set of triples that should be filtered out consists per default of
    the training and validation set. Only during the evaluation of the final model on the test set it consists of
    training, validation, and test triples. We explicitly do not use test triples for filtering during HPO in order
    to avoid any test leakage.

When using an evaluator, i.e., :class:`pykeen.evaluation.RankBasedEvaluator` or
:class:`pykeen.evaluation.SklearnEvaluator` in the filtered setting, per default, the training and the triples to
evaluate are used for filtering the sets of corrupted triples. An additional set of known positive triples can be
provided through the argument `additional_filter_triples` in the constructor of an evaluator.
For instance, if you want to evaluate your final model in the filtered setting on the test set of FB15k-237 following
[bordes2013]_, you need to provide the validation triples as additional positive triples
(`additional_filter_triples`) in the constructor:

.. code-block:: python

    from pykeen.datasets import FB15k237
    from pykeen.evaluation import RankBasedEvaluator
    from pykeen.models import TransE

    # Get FB15k-237 dataset
    fb15k237 = FB15k237()

    # Get triples
    validation_triples = fb15k237.validation.mapped_triples
    testing_triples = fb15k237.testing.mapped_triples

    # Define evaluator, and define validation triples as additional positive triples
    rb_evaluator = RankBasedEvaluator(
        filtered=True,  # Note: this is True by default; we're just being explicit
        additional_filter_triples=fb15k237.validation.mapped_triples
    )

    # Define model
    model = TransE(
        triples_factory=fb15k237.training,
    )

    # Evaluate your model
    results = rb_evaluator.evaluate(
        model=model,
        mapped_triples=fb15k237.testing.mapped_triples,
    )
"""  # noqa:D400,D205

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
