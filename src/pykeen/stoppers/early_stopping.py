# -*- coding: utf-8 -*-

"""Implementation of early stopping."""

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Union

from .stopper import Stopper
from ..evaluation import Evaluator
from ..models import Model
from ..trackers import ResultTracker
from ..triples import CoreTriplesFactory
from ..typing import InductiveMode
from ..utils import fix_dataclass_init_docs

__all__ = [
    "is_improvement",
    "EarlyStopper",
    "EarlyStoppingLogic",
    "StopperCallback",
]

logger = logging.getLogger(__name__)

StopperCallback = Callable[[Stopper, Union[int, float], int], None]


def is_improvement(
    best_value: float,
    current_value: float,
    larger_is_better: bool,
    relative_delta: float = 0.0,
) -> bool:
    """
    Decide whether the current value is an improvement over the best value.

    :param best_value:
        The best value so far.
    :param current_value:
        The current value.
    :param larger_is_better:
        Whether a larger value is better.
    :param relative_delta:
        A minimum relative improvement until it is considered as an improvement.

    :return:
        Whether the current value is better.
    """
    if larger_is_better:
        return current_value > (1.0 + relative_delta) * best_value

    # now: smaller is better
    return current_value < (1.0 - relative_delta) * best_value


@dataclasses.dataclass
class EarlyStoppingLogic:
    """The early stopping logic."""

    #: the number of reported results with no improvement after which training will be stopped
    patience: int = 2

    # the minimum relative improvement necessary to consider it an improved result
    relative_delta: float = 0.0

    # whether a larger value is better, or a smaller.
    larger_is_better: bool = True

    #: The epoch at which the best result occurred
    best_epoch: Optional[int] = None

    #: The best result so far
    best_metric: float = dataclasses.field(init=False)

    #: The remaining patience
    remaining_patience: int = dataclasses.field(init=False)

    def __post_init__(self):
        """Infer remaining default values."""
        self.remaining_patience = self.patience
        self.best_metric = float("-inf") if self.larger_is_better else float("+inf")

    def is_improvement(self, metric: float) -> bool:
        """Return if the given metric would cause an improvement."""
        return is_improvement(
            best_value=self.best_metric,
            current_value=metric,
            larger_is_better=self.larger_is_better,
            relative_delta=self.relative_delta,
        )

    def report_result(self, metric: float, epoch: int) -> bool:
        """
        Report a result at the given epoch.

        :param metric:
            The result metric.
        :param epoch:
            The epoch.

        :return:
            If the result did not improve more than delta for patience evaluations

        :raises ValueError:
            if more than one metric is reported for a single epoch
        """
        if self.best_epoch is not None and epoch <= self.best_epoch:
            raise ValueError("Cannot report more than one metric for one epoch")

        # check for improvement
        if self.is_improvement(metric):
            self.best_epoch = epoch
            self.best_metric = metric
            self.remaining_patience = self.patience
        else:
            self.remaining_patience -= 1

        # stop if the result did not improve more than delta for patience evaluations
        return self.remaining_patience <= 0


@fix_dataclass_init_docs
@dataclass
class EarlyStopper(Stopper):
    """A harness for early stopping."""

    #: The model
    model: Model = dataclasses.field(repr=False)
    #: The evaluator
    evaluator: Evaluator
    #: The triples to use for training (to be used during filtered evaluation)
    training_triples_factory: CoreTriplesFactory
    #: The triples to use for evaluation
    evaluation_triples_factory: CoreTriplesFactory
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
    metric: str = "hits_at_k"
    #: The minimum relative improvement necessary to consider it an improved result
    relative_delta: float = 0.01
    #: The metric results from all evaluations
    results: List[float] = dataclasses.field(default_factory=list, repr=False)
    #: Whether a larger value is better, or a smaller
    larger_is_better: bool = True
    #: The result tracker
    result_tracker: Optional[ResultTracker] = None
    #: Callbacks when after results are calculated
    result_callbacks: List[StopperCallback] = dataclasses.field(default_factory=list, repr=False)
    #: Callbacks when training gets continued
    continue_callbacks: List[StopperCallback] = dataclasses.field(default_factory=list, repr=False)
    #: Callbacks when training is stopped early
    stopped_callbacks: List[StopperCallback] = dataclasses.field(default_factory=list, repr=False)
    #: Did the stopper ever decide to stop?
    stopped: bool = False

    _stopper: EarlyStoppingLogic = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        """Run after initialization and check the metric is valid."""
        # TODO: Fix this
        # if all(f.name != self.metric for f in dataclasses.fields(self.evaluator.__class__)):
        #     raise ValueError(f'Invalid metric name: {self.metric}')
        self._stopper = EarlyStoppingLogic(
            patience=self.patience,
            relative_delta=self.relative_delta,
            larger_is_better=self.larger_is_better,
        )

    @property
    def remaining_patience(self) -> int:
        """Return the remaining patience."""
        return self._stopper.remaining_patience

    @property
    def best_metric(self) -> float:
        """Return the best result so far."""
        return self._stopper.best_metric

    @property
    def best_epoch(self) -> Optional[int]:
        """Return the epoch at which the best result occurred."""
        return self._stopper.best_epoch

    def should_evaluate(self, epoch: int) -> bool:
        """Decide if evaluation should be done based on the current epoch and the internal frequency."""
        return epoch > 0 and epoch % self.frequency == 0

    @property
    def number_results(self) -> int:
        """Count the number of results stored in the early stopper."""
        return len(self.results)

    def should_stop(self, epoch: int, *, mode: Optional[InductiveMode] = None) -> bool:
        """Evaluate on a metric and compare to past evaluations to decide if training should stop."""
        # Evaluate
        metric_results = self.evaluator.evaluate(
            model=self.model,
            additional_filter_triples=self.training_triples_factory.mapped_triples,
            mapped_triples=self.evaluation_triples_factory.mapped_triples,
            use_tqdm=False,
            batch_size=self.evaluation_batch_size,
            slice_size=self.evaluation_slice_size,
            # Only perform time consuming checks for the first call.
            do_time_consuming_checks=self.evaluation_batch_size is None,
        )
        # After the first evaluation pass the optimal batch and slice size is obtained and saved for re-use
        self.evaluation_batch_size = self.evaluator.batch_size
        self.evaluation_slice_size = self.evaluator.slice_size

        if self.result_tracker is not None:
            self.result_tracker.log_metrics(
                metrics=metric_results.to_flat_dict(),
                step=epoch,
                prefix="validation",
            )
        result = metric_results.get_metric(self.metric)

        # Append to history
        self.results.append(result)

        for result_callback in self.result_callbacks:
            result_callback(self, result, epoch)

        self.stopped = self._stopper.report_result(metric=result, epoch=epoch)
        if self.stopped:
            logger.info(
                f"Stopping early at epoch {epoch}. The best result {self.best_metric} occurred at "
                f"epoch {self.best_epoch}.",
            )
            for stopped_callback in self.stopped_callbacks:
                stopped_callback(self, result, epoch)
            return True

        for continue_callback in self.continue_callbacks:
            continue_callback(self, result, epoch)
        return False

    def get_summary_dict(self) -> Mapping[str, Any]:
        """Get a summary dict."""
        return dict(
            frequency=self.frequency,
            patience=self.patience,
            remaining_patience=self.remaining_patience,
            relative_delta=self.relative_delta,
            metric=self.metric,
            larger_is_better=self.larger_is_better,
            results=self.results,
            stopped=self.stopped,
            best_epoch=self.best_epoch,
            best_metric=self.best_metric,
        )

    def _write_from_summary_dict(
        self,
        *,
        frequency: int,
        patience: int,
        remaining_patience: int,
        relative_delta: float,
        metric: str,
        larger_is_better: bool,
        results: List[float],
        stopped: bool,
        best_epoch: int,
        best_metric: float,
    ) -> None:
        """Write attributes to stopper from a summary dict."""
        self.frequency = frequency
        self.patience = patience
        self.relative_delta = relative_delta
        self.metric = metric
        self.larger_is_better = larger_is_better
        self.results = results
        self.stopped = stopped
        # TODO need a test that this all re-instantiates properly
        self._stopper = EarlyStoppingLogic(
            patience=patience,
            relative_delta=relative_delta,
            larger_is_better=larger_is_better,
        )
        self._stopper.best_epoch = best_epoch
        self._stopper.best_metric = best_metric
        self._stopper.remaining_patience = remaining_patience
