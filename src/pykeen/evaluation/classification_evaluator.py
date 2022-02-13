# -*- coding: utf-8 -*-

"""Implementation of wrapper around sklearn metrics."""

from typing import Mapping, MutableMapping, Optional, Tuple, cast

import numpy as np
import rexmex.metrics.classification as rmc
import torch

from .evaluator import Evaluator, MetricResults
from .utils import MetricAnnotation, ValueRange
from ..constants import TARGET_TO_INDEX
from ..typing import MappedTriples, Target

__all__ = [
    "ClassificationEvaluator",
    "ClassificationMetricResults",
]

#: Functions with the right signature in the :mod:`rexmex.metrics.classification` that are not themselves metrics
EXCLUDE_CLASSIFIERS = {
    rmc.pr_auc_score,
}

CLASSIFICATION_METRICS: Mapping[str, MetricAnnotation] = {
    key: MetricAnnotation(
        name=func.name,
        description=func.description or "",
        link=func.link,
        value_range=ValueRange(
            upper=func.upper,
            lower=func.lower,
            upper_inclusive=func.upper_inclusive,
            lower_inclusive=func.lower_inclusive,
        ),
        increasing=func.higher_is_better,
        func=func,
        binarize=func.binarize,
    )
    for key, func in rmc.classifications.funcs.items()
    if func not in EXCLUDE_CLASSIFIERS and func.duplicate_of is None
}


class ClassificationMetricResults(MetricResults):
    """Results from computing metrics."""

    metrics = CLASSIFICATION_METRICS

    @classmethod
    def from_scores(cls, y_true, y_score):
        """Return an instance of these metrics from a given set of true and scores."""
        return ClassificationMetricResults(
            {key: metric.score(y_true, y_score) for key, metric in CLASSIFICATION_METRICS.items()}
        )

    def get_metric(self, name: str) -> float:  # noqa: D102
        return self.data[name]


class ClassificationEvaluator(Evaluator):
    """An evaluator that uses a classification metrics."""

    all_scores: MutableMapping[Tuple[Target, int, int], np.ndarray]
    all_positives: MutableMapping[Tuple[Target, int, int], np.ndarray]

    def __init__(self, **kwargs):
        super().__init__(
            filtered=False,
            requires_positive_mask=True,
            **kwargs,
        )
        self.all_scores = {}
        self.all_positives = {}

    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: torch.FloatTensor,
        true_scores: Optional[torch.FloatTensor] = None,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        if dense_positive_mask is None:
            raise KeyError("Sklearn evaluators need the positive mask!")

        # Transfer to cpu and convert to numpy
        scores = scores.detach().cpu().numpy()
        dense_positive_mask = dense_positive_mask.detach().cpu().numpy()
        remaining = [i for i in range(hrt_batch.shape[1]) if i != TARGET_TO_INDEX[target]]
        keys = hrt_batch[:, remaining].detach().cpu().numpy()

        # Ensure that each key gets counted only once
        for i in range(keys.shape[0]):
            # include head_side flag into key to differentiate between (h, r) and (r, t)
            key_suffix = tuple(map(int, keys[i]))
            assert len(key_suffix) == 2
            key_suffix = cast(Tuple[int, int], key_suffix)
            key = (target,) + key_suffix
            self.all_scores[key] = scores[i]
            self.all_positives[key] = dense_positive_mask[i]

    def finalize(self) -> ClassificationMetricResults:  # noqa: D102
        # Because the order of the values of an dictionary is not guaranteed,
        # we need to retrieve scores and masks using the exact same key order.
        all_keys = list(self.all_scores.keys())
        y_score = np.concatenate([self.all_scores[k] for k in all_keys], axis=0).flatten()
        y_true = np.concatenate([self.all_positives[k] for k in all_keys], axis=0).flatten()

        # Clear buffers
        self.all_positives.clear()
        self.all_scores.clear()

        return ClassificationMetricResults.from_scores(y_true, y_score)
