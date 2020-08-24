# -*- coding: utf-8 -*-

"""Implementation of early stopping."""

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Union

from .stopper import Stopper
from ..evaluation import Evaluator
from ..models.base import Model
from ..trackers import ResultTracker
from ..triples import TriplesFactory
from ..utils import fix_dataclass_init_docs

__all__ = [
    'is_improvement',
    'EarlyStopper',
    'StopperCallback',
]

logger = logging.getLogger(__name__)

StopperCallback = Callable[[Stopper, Union[int, float]], None]


def is_improvement(
    best_value: float,
    current_value: float,
    larger_is_better: bool,
    absolute_delta: float = 0.0,
) -> bool:
    """
    Decide whether the current value is an improvement over the best value.

    :param best_value:
        The best value so far.
    :param current_value:
        The current value.
    :param larger_is_better:
        Whether a larger value is better.
    :param absolute_delta:
        A minimum absolute improvement until it is considered as an improvement.

    :return:
         Whether the current value is better.
    """
    if larger_is_better:
        best_value = -best_value
        current_value = -current_value

    # now: smaller is better
    return current_value < (best_value - absolute_delta)


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
    #: The best result so far
    best_result: Optional[float] = None
    #: The epoch at which the best result occurred
    best_epoch: Optional[int] = None
    #: The remaining patience
    remaining_patience: int = dataclasses.field(init=False)
    #: The metric results from all evaluations
    results: List[float] = dataclasses.field(default_factory=list, repr=False)
    #: A counter for the ring buffer
    number_evaluations: int = 0
    #: Whether a larger value is better, or a smaller
    larger_is_better: bool = True
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

        self.remaining_patience = self.patience

        # Dummy result tracker
        if self.result_tracker is None:
            self.result_tracker = ResultTracker()

    def should_evaluate(self, epoch: int) -> bool:
        """Decide if evaluation should be done based on the current epoch and the internal frequency."""
        return epoch > 0 and epoch % self.frequency == 0

    @property
    def number_results(self) -> int:
        """Count the number of results stored in the early stopper."""
        return len(self.results)

    def should_stop(self, epoch: int) -> bool:
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
            metrics=metric_results.to_flat_dict(),
            step=epoch,
            prefix='validation',
        )
        result = metric_results.get_metric(self.metric)

        # Append to history
        self.results.append(result)
        self.number_evaluations += 1

        # check for improvement
        if self.best_result is None or is_improvement(
            best_value=self.best_result,
            current_value=result,
            larger_is_better=self.larger_is_better,
            absolute_delta=self.delta,
        ):
            self.best_epoch = epoch
            self.best_result = result
            self.remaining_patience = self.patience
        else:
            self.remaining_patience -= 1

        # Stop if the result did not improve more than delta for patience evaluations
        if self.remaining_patience <= 0:
            logger.info(f'Stopping early after {self.number_evaluations} evaluations with {self.metric}={result}')
            for stopped_callback in self.stopped_callbacks:
                stopped_callback(self, result, epoch)
            self.stopped = True
            return True

        for continue_callback in self.continue_callbacks:
            continue_callback(self, result, epoch)
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
