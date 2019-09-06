# -*- coding: utf-8 -*-

"""Implementation of early stopping."""

import dataclasses
from dataclasses import dataclass
from typing import Callable, List

import numpy

from ..evaluation import Evaluator, MetricResults
from ..triples import TriplesFactory

__all__ = [
    'smaller_than_any_buffer_element',
    'larger_than_any_buffer_element',
    'EarlyStopper',
]


def smaller_than_any_buffer_element(buffer: numpy.ndarray, result: float, delta: float = 0.) -> bool:
    """Decide if a result is better than at least one buffer element, where smaller is better.

    :param buffer:
        The last results to compare against (excluding the current result).
    :param result:
        The current result.
    :param delta:
        The minimum improvement.

    :return:
        Whether the result is at least delta better than at least one value in the buffer.
    """
    worst_in_window = buffer.max()
    baseline = worst_in_window - delta
    return result < baseline


def larger_than_any_buffer_element(buffer: numpy.ndarray, result: float, delta: float = 0.) -> bool:
    """Decide if a result is better than at least one buffer element, where larger is better.

    :param buffer:
        The last results to compare against (excluding the current result).
    :param result:
        The current result.
    :param delta:
        The minimum improvement.

    :return:
        Whether the result is at least delta better than at least one value in the buffer.
    """
    worst_in_window = buffer.min()
    baseline = worst_in_window + delta
    return result > baseline


@dataclass
class EarlyStopper:
    """A harness for early stopping.

    If you want to change the validation criteria, inherit from this
    class and override ``EarlyStopper._validate()``.
    """

    #: The evaluator
    evaluator: Evaluator
    #: The triples to use for evaluation
    evaluation_triples_factory: TriplesFactory
    #: The number of epochs after which the model is evaluated on validation set
    frequency: int = 10
    #: The number of iterations (one iteration can correspond to various epochs)
    #: with no improvement after which training will be stopped.
    patience: int = 2
    #: The name of the metric to use
    metric: str = 'hits_at_k'
    #: The minimum improvement between two iterations
    delta: float = 0.005
    #: The metric results from all evaluations
    results: List[float] = dataclasses.field(default_factory=list)
    #: A ring buffer to store the recent results
    buffer: numpy.ndarray = dataclasses.field(init=False)
    #: A counter for the ring buffer
    number_evaluations: int = 0
    #: Whether a larger value is better, or a smaller
    larger_is_better: bool = True
    #: The criterion. Set in the constructor based on larger_is_better
    improvement_criterion: Callable[[numpy.ndarray, float, float], bool] = None

    def __post_init__(self):
        """Run after initialization and check the metric is valid."""
        if all(f.name != self.metric for f in dataclasses.fields(MetricResults)):
            raise ValueError(f'Invalid metric name: {self.metric}')
        if self.larger_is_better:
            self.improvement_criterion = larger_than_any_buffer_element
        else:
            self.improvement_criterion = smaller_than_any_buffer_element

        self.buffer = numpy.empty(shape=(self.patience,))

    def _get_result(self, metric_results: MetricResults) -> float:
        result = getattr(metric_results, self.metric)
        if self.metric == 'hits_at_k':
            result = result[10]
        return result

    def should_stop(self) -> bool:
        """Validate on validation set and check for termination condition."""
        # Evaluate
        metric_results = self.evaluator.evaluate(mapped_triples=self.evaluation_triples_factory.mapped_triples)
        result = self._get_result(metric_results)

        # Only check if enough values are already collected
        if self.number_evaluations >= self.patience:
            # Stop if the result did not improve more than delta for patience epochs.
            if not self.improvement_criterion(buffer=self.buffer, result=result, delta=self.delta):
                return True

        # Update ring buffer
        self.buffer[self.number_evaluations % self.patience] = result
        self.number_evaluations += 1

        # Append to history
        self.results.append(result)

        return False
