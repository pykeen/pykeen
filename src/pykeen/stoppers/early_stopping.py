# -*- coding: utf-8 -*-

"""Implementation of early stopping."""

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Union

import numpy

from .stopper import Stopper
from ..evaluation import Evaluator
from ..models.base import Model
from ..trackers import ResultTracker
from ..triples import TriplesFactory
from ..utils import fix_dataclass_init_docs

__all__ = [
    'smaller_than_any_buffer_element',
    'larger_than_any_buffer_element',
    'EarlyStopper',
    'StopperCallback',
]

logger = logging.getLogger(__name__)


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


StopperCallback = Callable[[Stopper, Union[int, float]], None]


@fix_dataclass_init_docs
@dataclass
class EarlyStopper(Stopper):
    """A harness for early stopping."""

    #: The model
    model: Model = dataclasses.field(repr=False)
    #: The evaluator
    evaluator: Evaluator
    #: The triples to use for evaluation
    evaluation_triples_factory: Optional[TriplesFactory]
    #: Size of the evaluation batches
    evaluation_batch_size: Optional[int] = None
    #: Slice size of the evaluation batches
    evaluation_slice_size: Optional[int] = None
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
    results: List[float] = dataclasses.field(default_factory=list, repr=False)
    #: A ring buffer to store the recent results
    buffer: numpy.ndarray = dataclasses.field(init=False)
    #: A counter for the ring buffer
    number_evaluations: int = 0
    #: Whether a larger value is better, or a smaller
    larger_is_better: bool = True
    #: The criterion. Set in the constructor based on larger_is_better
    improvement_criterion: Callable[[numpy.ndarray, float, float], bool] = None
    #: The result tracker
    result_tracker: Optional[ResultTracker] = None
    #: Callbacks when training gets continued
    continue_callbacks: List[StopperCallback] = dataclasses.field(default_factory=list, repr=False)
    #: Callbacks when training is stopped early
    stopped_callbacks: List[StopperCallback] = dataclasses.field(default_factory=list, repr=False)
    #: Did the stopper ever decide to stop?
    stopped: bool = False

    def __post_init__(self):
        """Run after initialization and check the metric is valid."""
        # TODO: Fix this
        # if all(f.name != self.metric for f in dataclasses.fields(self.evaluator.__class__)):
        #     raise ValueError(f'Invalid metric name: {self.metric}')
        if self.evaluation_triples_factory is None:
            raise ValueError('Must specify a validation_triples_factory or a dataset for using early stopping.')

        if self.larger_is_better:
            self.improvement_criterion = larger_than_any_buffer_element
        else:
            self.improvement_criterion = smaller_than_any_buffer_element

        self.buffer = numpy.empty(shape=(self.patience,))

        # Dummy result tracker
        if self.result_tracker is None:
            self.result_tracker = ResultTracker()

    def should_evaluate(self, epoch: int) -> bool:
        """Decide if evaluation should be done based on the current epoch and the internal frequency."""
        return 0 == ((epoch - 1) % self.frequency)

    @property
    def number_results(self) -> int:
        """Count the number of results stored in the early stopper."""
        return len(self.results)

    def should_stop(self) -> bool:
        """Evaluate on a metric and compare to past evaluations to decide if training should stop."""
        # Evaluate
        metric_results = self.evaluator.evaluate(
            model=self.model,
            mapped_triples=self.evaluation_triples_factory.mapped_triples,
            use_tqdm=False,
            batch_size=self.evaluation_batch_size,
            slice_size=self.evaluation_slice_size,
        )
        # After the first evaluation pass the optimal batch and slice size is obtained and saved for re-use
        self.evaluation_batch_size = self.evaluator.batch_size
        self.evaluation_slice_size = self.evaluator.slice_size

        self.result_tracker.log_metrics(
            metrics=metric_results.to_json(),
            step=self.number_evaluations,
            prefix='validation',
        )
        result = metric_results.get_metric(self.metric)

        # Only check if enough values are already collected
        if self.number_evaluations >= self.patience:
            # Stop if the result did not improve more than delta for patience epochs.
            if not self.improvement_criterion(buffer=self.buffer, result=result, delta=self.delta):
                logger.info(f'Stopping early after {self.number_evaluations} evaluations with {self.metric}={result}')
                for stopped_callback in self.stopped_callbacks:
                    stopped_callback(self, result)
                self.stopped = True
                return True

        # Update ring buffer
        self.buffer[self.number_evaluations % self.patience] = result
        self.number_evaluations += 1

        # Append to history
        self.results.append(result)

        for continue_callback in self.continue_callbacks:
            continue_callback(self, result)
        return False

    def get_summary_dict(self) -> Mapping[str, Any]:
        """Get a summary dict."""
        return dict(
            frequency=self.frequency,
            patience=self.patience,
            delta=self.delta,
            metric=self.metric,
            larger_is_better=self.larger_is_better,
            results=self.results,
            stopped=self.stopped,
        )
