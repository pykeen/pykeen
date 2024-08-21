"""Scheduling when to make checkpoints."""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping
from typing import Protocol

from class_resolver import ClassResolver

from ..trackers.base import ResultTracker


class StepPredicate(Protocol):
    """A predicate on steps."""

    def __call__(self, step: int) -> bool: ...


@dataclasses.dataclass
class RegularCheckpoints(StepPredicate):
    """Create a checkpoint every $n$ steps."""

    frequency: int = 10

    def __call__(self, step: int) -> bool:
        return not step % self.frequency


@dataclasses.dataclass
class ExplicitCheckpoints(StepPredicate):
    """Create a checkpoint for explicitly chosen steps."""

    steps: Collection[int]

    def __call__(self, step: int) -> bool:
        return step in self.steps


@dataclasses.dataclass
class ResultListenerAdapterResultTracker(ResultTracker):
    """A listener on a result tracker."""

    base: ResultTracker

    metric: str
    prefix: str | None = None

    maximize: bool = True

    best: float = dataclasses.field(init=False)
    best_step: None | int = dataclasses.field(default=None, init=False)
    last_step: None | int = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        self.best = float("-inf") if self.maximize else float("+inf")

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: int | None = None,
        prefix: str | None = None,
    ) -> None:
        self.last_step = step

        # prefix filter
        if self.prefix and not prefix == self.prefix:
            return
        # metric filter
        if self.metric not in metrics:
            return
        value = metrics[self.metric]
        if self.maximize and value > self.best:
            self.best_step = step
            self.best = value
        elif not self.maximize and value < self.best:
            self.best_step = step
            self.best = value


@dataclasses.dataclass
class BestCheckpoints(StepPredicate):
    """Create a checkpoint for explicitly chosen steps."""

    result_tracker: ResultTracker
    metric: str
    prefix: str | None = None
    maximize: bool = True

    adapter: ResultListenerAdapterResultTracker = dataclasses.field(init=False)

    def __post_init__(self):
        self.adapter = ResultListenerAdapterResultTracker(
            self.result_tracker, metric=self.metric, prefix=self.prefix, maximize=self.maximize
        )

    def __call__(self, step: int) -> bool:
        if self.adapter.last_step is None:
            raise ValueError("The result tracker did not receive any results so far.")
        return step == self.adapter.best_step


schedule_resolver = ClassResolver.from_subclasses(base=StepPredicate, default=RegularCheckpoints)
