# -*- coding: utf-8 -*-

"""Implementation of wrapper around sklearn metrics."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy
import torch
from dataclasses_json import dataclass_json
from sklearn.metrics import ranking

from .evaluator import Evaluator, MetricResults
from ..typing import MappedTriples, SklearnMetric
from ..utils import normalize_string


@dataclass_json
@dataclass
class SklearnMetricResults(MetricResults):
    """Results from computing metrics."""

    #: The name of the metric
    name: str

    #: The score over all triples
    score: float


SKLEARN_METRICS = {
    normalize_string(f.__name__): f
    for f in (
        ranking.roc_auc_score,
        ranking.average_precision_score,
        ranking.coverage_error,
        ranking.label_ranking_average_precision_score,
        ranking.label_ranking_loss,
    )
}


def _get_sklearn_metric(metric: Union[None, str, SklearnMetric]) -> SklearnMetric:
    """Look up a metric by name if a string is given, pass through a metric, or default to AUC-ROC."""
    if metric is None:
        metric = ranking.roc_auc_score
    elif isinstance(metric, str):
        metric = SKLEARN_METRICS.get(normalize_string(metric))
        if metric is None:
            raise KeyError(f'Unknown metric name: "{metric}". Known are {set(SKLEARN_METRICS.keys())}.')
    return metric


class SklearnEvaluator(Evaluator):
    """An evaluator that uses a Scikit-learn metric."""

    #: The sklearn evaluation metric (e.g. metrics.roc_auc_score)
    metric: SklearnMetric

    def __init__(
        self,
        metric: Union[None, str, SklearnMetric] = None,
    ):
        super().__init__(filtered=False, requires_positive_mask=True)
        self.metric = _get_sklearn_metric(metric=metric)
        self.all_scores = {}
        self.all_positives = {}

    def _process_scores(
        self,
        keys: torch.LongTensor,
        scores: torch.FloatTensor,
        positive_mask: torch.BoolTensor,
        subject_side: bool,
    ) -> None:
        # Transfer to cpu and convert to numpy
        scores = scores.detach().cpu().numpy()
        positive_mask = positive_mask.detach().cpu().numpy()
        keys = keys.detach().cpu().numpy()

        # Ensure that each key gets counted only once
        for i in range(keys.shape[0]):
            # include subject_side flag into key to differentiate between (s, p) and (p, o)
            key = (subject_side,) + tuple(map(int, keys[i]))
            self.all_scores[key] = scores[i]
            self.all_positives[key] = positive_mask[i]

    def process_object_scores_(
        self,
        batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.BoolTensor] = None,
    ) -> None:  # noqa: D102
        if dense_positive_mask is None:
            raise KeyError('Sklearn evaluators need the positive mask!')

        self._process_scores(keys=batch[:, :2], scores=scores, positive_mask=dense_positive_mask, subject_side=False)

    def process_subject_scores_(
        self,
        batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.BoolTensor] = None,
    ) -> None:  # noqa: D102
        if dense_positive_mask is None:
            raise KeyError('Sklearn evaluators need the positive mask!')

        self._process_scores(keys=batch[:, 1:], scores=scores, positive_mask=dense_positive_mask, subject_side=True)

    def finalize(self) -> SklearnMetricResults:  # noqa: D102
        # Important: The order of the values of an dictionary is not guaranteed. Hence, we need to retrieve scores and
        # masks using the exact same key order.
        all_keys = list(self.all_scores.keys())
        y_score = numpy.concatenate([self.all_scores[k] for k in all_keys], axis=0).flatten()
        y_true = numpy.concatenate([self.all_positives[k] for k in all_keys], axis=0).flatten()
        return SklearnMetricResults(
            name=self.metric.__name__,
            score=self.metric(y_true, y_score),
        )
