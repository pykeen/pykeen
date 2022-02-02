# -*- coding: utf-8 -*-

"""Implementation of wrapper around sklearn metrics."""

from dataclasses import dataclass, field, fields, make_dataclass
from typing import MutableMapping, Optional, Tuple, cast

import numpy as np
import torch
from dataclasses_json import dataclass_json

from .evaluator import Evaluator, MetricResults
from .rexmex_compat import classifier_annotator
from ..constants import TARGET_TO_INDEX
from ..typing import MappedTriples, Target
from ..utils import fix_dataclass_init_docs

__all__ = [
    "ClassificationEvaluator",
    "ClassificationMetricResults",
]

_fields = [
    (
        metadata.func.__name__,
        float,
        field(
            metadata=dict(
                name=metadata.name,
                doc=metadata.description or "",
                link=metadata.link,
                range=metadata.interval(),
                increasing=metadata.higher_is_better,
                f=metadata.func,
            )
        ),
    )
    for metadata in classifier_annotator.metrics.values()
]

ClassificationMetricResultsBase = make_dataclass(
    "ClassificationMetricResultsBase",
    _fields,
    bases=(MetricResults,),
)


@fix_dataclass_init_docs
@dataclass_json
@dataclass
class ClassificationMetricResults(ClassificationMetricResultsBase):  # type: ignore
    """Results from computing metrics."""

    @classmethod
    def from_scores(cls, y_true, y_score):
        """Return an instance of these metrics from a given set of true and scores."""
        return ClassificationMetricResults(**{f.name: f.metadata["f"](y_true, y_score) for f in fields(cls)})

    def get_metric(self, name: str) -> float:  # noqa: D102
        return getattr(self, name)


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

        return ClassificationMetricResults.from_scores(y_true, y_score)
