"""Internal utility methods."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from ..trackers.base import ResultTracker

__all__ = [
    "ResultListenerAdapter",
    "MetricSelection",
]


@dataclasses.dataclass
class MetricSelection:
    """The selection of the metric to monitor."""

    # TODO: for some reason, this field is missing in the documentation
    #: the normalized metric name (as seen by the result tracker)
    metric: str

    #: the metric prefix; if None, do not check prefix
    prefix: str | None = None

    #: whether to maximize or minimize the metric
    maximize: bool = True


@dataclasses.dataclass
class ResultListenerAdapter(ResultTracker):
    """An adapter to keep track of the best value and step for a given metric."""

    base: ResultTracker

    metric_selection: MetricSelection

    best: float = dataclasses.field(init=False)
    best_step: None | int = dataclasses.field(default=None, init=False)
    last_step: None | int = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        self.best = float("-inf") if self.metric_selection.maximize else float("+inf")
        self.base_log_metrics = self.base.log_metrics
        self.base.log_metrics = self.log_metrics

    # docstr-coverage: inherited
    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: int | None = None,
        prefix: str | None = None,
    ) -> None:
        self.base_log_metrics(metrics=metrics, step=step, prefix=prefix)
        self.last_step = step

        # prefix filter
        if self.metric_selection.prefix and not prefix == self.metric_selection.prefix:
            return
        # metric filter
        if self.metric_selection.metric not in metrics:
            return
        value = metrics[self.metric_selection.metric]
        if self.metric_selection.maximize and value > self.best:
            self.best_step = step
            self.best = value
        elif not self.metric_selection.maximize and value < self.best:
            self.best_step = step
            self.best = value

    def is_best(self, step: int) -> bool:
        """Check if the given step corresponds to the best."""
        if self.last_step is None:
            raise ValueError(
                "The result tracker did not receive any results so far. Did you forget to use the same result "
                "tracker instance that is running in training?",
            )
        return step == self.best_step
