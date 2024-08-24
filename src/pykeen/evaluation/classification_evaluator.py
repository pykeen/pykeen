"""Implementation of wrapper around sklearn metrics."""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable, Mapping, MutableMapping
from typing import NamedTuple, cast

import numpy
import numpy as np
import torch

from .evaluator import Evaluator, MetricResults
from ..constants import TARGET_TO_INDEX
from ..metrics.classification import ClassificationMetric, classification_metric_resolver
from ..typing import SIDE_BOTH, ExtendedTarget, MappedTriples, Target, normalize_target

__all__ = [
    "ClassificationEvaluator",
    "ClassificationMetricResults",
]


class ClassificationMetricKey(NamedTuple):
    """A key for classification metrics."""

    side: ExtendedTarget
    metric: str


class ScorePack(NamedTuple):
    """A pack of scores for aggregation."""

    target: ExtendedTarget
    y_true: numpy.ndarray
    y_score: numpy.ndarray


class ClassificationMetricResults(MetricResults[ClassificationMetricKey]):
    """Results from computing metrics."""

    metrics = classification_metric_resolver.lookup_dict

    # docstr-coverage: inherited
    @classmethod
    def key_from_string(cls, s: str | None) -> ClassificationMetricKey:  # noqa: D102
        if s is None:
            s = classification_metric_resolver.make(query=None).key
        # side?.metric
        parts = s.split(".")
        side = normalize_target(None if len(parts) < 2 else parts[0])
        metric = parts[-1]
        return ClassificationMetricKey(side=side, metric=metric)

    @classmethod
    def from_scores(cls, metrics: Iterable[ClassificationMetric], scores_and_true_masks: Iterable[ScorePack]):
        """Return an instance of these metrics from a given set of true and scores."""
        return cls(
            data={
                ClassificationMetricKey(side=pack.target, metric=metric.key): metric(pack.y_true, pack.y_score)
                for metric, pack in itertools.product(metrics, scores_and_true_masks)
            }
        )


def _iter_scores(
    all_scores: Mapping[Target, Mapping[tuple[int, int], numpy.ndarray]],
    all_positives: Mapping[Target, Mapping[tuple[int, int], numpy.ndarray]],
) -> Iterable[ScorePack]:
    sides = sorted(all_scores.keys())
    y_score_for_side = dict()
    y_true_for_side = dict()

    # individual side
    for side in sides:
        # Because the order of the values of a dictionary is not guaranteed,
        # we need to retrieve scores and masks using the exact same key order.
        all_keys = list(all_scores[side].keys())
        y_score = y_score_for_side[side] = np.concatenate([all_scores[side][k] for k in all_keys], axis=0).flatten()
        y_true = y_true_for_side[side] = np.concatenate([all_positives[side][k] for k in all_keys], axis=0).flatten()
        assert y_score.shape == y_true.shape
        if y_true.size == 0:
            raise ValueError(f"Cannot calculate scores from empty array (y_true.shape={y_true.shape}).")
        yield ScorePack(target=side, y_true=y_true, y_score=y_score)

    # combined
    yield ScorePack(
        target=SIDE_BOTH,
        y_true=np.concatenate([y_true_for_side[side] for side in sides]),
        y_score=np.concatenate([y_score_for_side[side] for side in sides]),
    )


class ClassificationEvaluator(Evaluator[ClassificationMetricKey]):
    """An evaluator that uses a classification metrics."""

    metric_result_cls = ClassificationMetricResults
    all_scores: MutableMapping[Target, MutableMapping[tuple[int, int], np.ndarray]]
    all_positives: MutableMapping[Target, MutableMapping[tuple[int, int], np.ndarray]]

    def __init__(self, **kwargs):
        """
        Initialize the evaluator.

        :param kwargs:
            keyword-based parameters passed to :meth:`Evaluator.__init__`.
        """
        super().__init__(
            filtered=False,
            requires_positive_mask=True,
            **kwargs,
        )
        self.all_scores = defaultdict(dict)
        self.all_positives = defaultdict(dict)
        self.metrics = tuple(
            classification_metric_resolver.make(metric_cls)
            for metric_cls in classification_metric_resolver.lookup_dict.values()
        )

    # docstr-coverage: inherited
    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: torch.FloatTensor,
        true_scores: torch.FloatTensor | None = None,
        dense_positive_mask: torch.FloatTensor | None = None,
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
            key = tuple(map(int, keys[i]))
            assert len(key) == 2
            key = cast(tuple[int, int], key)
            self.all_scores[target][key] = scores[i]
            self.all_positives[target][key] = dense_positive_mask[i]

    # docstr-coverage: inherited
    def clear(self) -> None:  # noqa: D102
        self.all_positives.clear()
        self.all_scores.clear()

    # docstr-coverage: inherited
    def finalize(self) -> ClassificationMetricResults:  # noqa: D102
        result = ClassificationMetricResults.from_scores(
            metrics=self.metrics,
            scores_and_true_masks=_iter_scores(all_scores=self.all_scores, all_positives=self.all_positives),
        )
        self.clear()
        return result
