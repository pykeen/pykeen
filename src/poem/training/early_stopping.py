# -*- coding: utf-8 -*-

"""Implementation of early stopping."""

import dataclasses
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from ..evaluation import Evaluator, MetricResults


@dataclass
class EarlyStopper:
    """A harness for early stopping.

    If you want to change the validation criteria, inherit from this
    class and override ``EarlyStopper_validate()``.
    """

    #: The evaluator
    evaluator: Evaluator
    #: The triples to use for evaluation
    triples: np.ndarray
    #: The number of epochs after which the model is evaluated on validation set
    frequency: int = 10
    #: The number of iterations (one iteration can correspond to various epochs)
    #: with no improvement after which training will be stopped.
    window: int = 2
    #: The name of the metric to use
    metric: str = 'hits_at_k'
    #: The metric results from all evaluations
    results: List[float] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        """Run after initialization and check the metric is valid."""
        if all(f.name != self.metric for f in dataclasses.fields(MetricResults)):
            raise ValueError(f'Invalid metric name: {self.metric}')

    @property
    def recent_results(self) -> List[float]:  # noqa: D401
        """Previous metrics before the most recent metric."""
        return self.results[-(self.window + 1):-1]

    @property
    def current_result(self) -> float:  # noqa: D401
        """The most recent metric calculated."""
        return self.results[-1]

    @property
    def initialized(self) -> bool:  # noqa: D401
        """Check that enough evaluations have been done that checking is meaningful."""
        return len(self.results) > self.window

    def evaluate(self) -> float:
        """Evaluate on the validation set."""
        metric_results = self.evaluator.evaluate(triples=self.triples)
        result = self._get_result(metric_results)
        self.results.append(result)
        return result

    def _get_result(self, metric_results: MetricResults) -> float:
        result = getattr(metric_results, self.metric)
        if self.metric == 'hits_at_k':
            result = result[10]
        return result

    def validate(self) -> bool:
        """Check whether the performance has recently improved."""
        return self.initialized and self._validate(self.current_result, self.recent_results)

    @staticmethod
    def _validate(current_result: float, recent_results: Iterable[float]) -> bool:
        return all(
            previous_result <= current_result
            for previous_result in recent_results
        )

    def should_stop(self) -> bool:
        """Validate on validation set and check for termination condition."""
        self.evaluate()
        return self.validate()
