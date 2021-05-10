# -*- coding: utf-8 -*-

"""Implementation of wrapper around sklearn metrics."""

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from dataclasses_json import dataclass_json
from sklearn import metrics

from .evaluator import Evaluator, MetricResults
from ..typing import MappedTriples
from ..utils import fix_dataclass_init_docs

__all__ = [
    'SklearnEvaluator',
    'SklearnMetricResults',
]


@fix_dataclass_init_docs
@dataclass_json
@dataclass
class SklearnMetricResults(MetricResults):
    """Results from computing metrics."""

    #: The area under the ROC curve
    roc_auc_score: float = field(metadata=dict(
        name="AUC-ROC",
        doc='The area under the ROC curve, on [0, 1]. Higher is better.',
        f=metrics.roc_auc_score,
    ))
    #: The area under the precision-recall curve
    average_precision_score: float = field(metadata=dict(
        name="Average Precision",
        doc='The area under the precision-recall curve, on [0, 1]. Higher is better.',
        f=metrics.average_precision_score,
    ))

    #: The coverage error
    # coverage_error: float = field(metadata=dict(
    #     doc='The coverage error',
    #     f=metrics.coverage_error,
    # ))
    #: The label ranking loss (APS)
    # label_ranking_average_precision_score: float = field(metadata=dict(
    #     doc='The label ranking loss (APS)',
    #     f=metrics.label_ranking_average_precision_score,
    # ))
    # #: The label ranking loss
    # label_ranking_loss: float = field(metadata=dict(
    #     doc='The label ranking loss',
    #     f=metrics.label_ranking_loss,
    # ))

    @classmethod
    def from_scores(cls, y_true, y_score):
        """Return an instance of these metrics from a given set of true and scores."""
        return SklearnMetricResults(**{
            f.name: f.metadata['f'](y_true, y_score)
            for f in fields(cls)
        })

    def get_metric(self, name: str) -> float:  # noqa: D102
        return getattr(self, name)


class SklearnEvaluator(Evaluator):
    """An evaluator that uses a Scikit-learn metric."""

    all_scores: Dict[Tuple[Any, ...], np.ndarray]
    all_positives: Dict[Tuple[Any, ...], np.ndarray]

    def __init__(self, **kwargs):
        super().__init__(
            filtered=False,
            requires_positive_mask=True,
            **kwargs,
        )
        self.all_scores = {}
        self.all_positives = {}

    def _process_scores(
        self,
        keys: torch.LongTensor,
        scores: torch.FloatTensor,
        positive_mask: torch.FloatTensor,
        head_side: bool,
    ) -> None:
        # Transfer to cpu and convert to numpy
        scores = scores.detach().cpu().numpy()
        positive_mask = positive_mask.detach().cpu().numpy()
        keys = keys.detach().cpu().numpy()

        # Ensure that each key gets counted only once
        for i in range(keys.shape[0]):
            # include head_side flag into key to differentiate between (h, r) and (r, t)
            key = (head_side,) + tuple(map(int, keys[i]))
            self.all_scores[key] = scores[i]
            self.all_positives[key] = positive_mask[i]

    def process_tail_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        if dense_positive_mask is None:
            raise KeyError('Sklearn evaluators need the positive mask!')

        self._process_scores(keys=hrt_batch[:, :2], scores=scores, positive_mask=dense_positive_mask, head_side=False)

    def process_head_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        if dense_positive_mask is None:
            raise KeyError('Sklearn evaluators need the positive mask!')

        self._process_scores(keys=hrt_batch[:, 1:], scores=scores, positive_mask=dense_positive_mask, head_side=True)

    def finalize(self) -> SklearnMetricResults:  # noqa: D102
        # Important: The order of the values of an dictionary is not guaranteed. Hence, we need to retrieve scores and
        # masks using the exact same key order.
        all_keys = list(self.all_scores.keys())
        # TODO how to define a cutoff on y_scores to make binary?
        # see: https://github.com/xptree/NetMF/blob/77286b826c4af149055237cef65e2a500e15631a/predict.py#L25-L33
        y_score = np.concatenate([self.all_scores[k] for k in all_keys], axis=0).flatten()
        y_true = np.concatenate([self.all_positives[k] for k in all_keys], axis=0).flatten()

        # Clear buffers
        self.all_positives.clear()
        self.all_scores.clear()

        return SklearnMetricResults.from_scores(y_true, y_score)
