"""Scheduling when to make checkpoints."""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping, Sequence
from typing import Protocol

from class_resolver import ClassResolver, HintOrType, OneOrManyHintOrType, OneOrManyOptionalKwargs, OptionalKwargs

from ..trackers.base import ResultTracker

__all__ = [
    "CheckpointSchedule",
    "schedule_resolver",
    "inspect_schedule",
]


class CheckpointSchedule(Protocol):
    """Determine whether to create a checkpoint at the given epoch."""

    def __call__(self, step: int) -> bool: ...


@dataclasses.dataclass
class RegularCheckpointSchedule(CheckpointSchedule):
    """
    Create a checkpoint every $n$ steps.

    Example::

        from pykeen.pipeline import

        result = pipeline(
            dataset="nations",
            model="mure",
            training_kwargs=dict(
                num_epochs=10,
                callbacks="checkpoint",
                # create one checkpoint every 3 epochs
                callbacks_kwargs=dict(
                    predicate="regular",
                    predicate_kwargs=dict(
                        frequency=3,
                    ),
                )
            ),
        )
    """

    #: The checkpoint frequency
    frequency: int = 10

    def __call__(self, step: int) -> bool:
        return not step % self.frequency


@dataclasses.dataclass
class ExplicitCheckpointSchedule(CheckpointSchedule):
    """
    Create a checkpoint for explicitly chosen steps.

    Example::

        from pykeen.pipeline import

        result = pipeline(
            dataset="nations",
            model="mure",
            training_kwargs=dict(
                num_epochs=10,
                callbacks="checkpoint",
                # create checkpoints at epoch 1, 7, and 10
                callbacks_kwargs=dict(
                    predicate="explicit",
                    predicate_kwargs=dict(
                        steps=(1, 7, 10)
                    ),
                )
            ),
        )
    """

    steps: Collection[int]

    def __call__(self, step: int) -> bool:
        return step in self.steps


@dataclasses.dataclass
class ResultListenerAdapterResultTracker(ResultTracker):
    """A listener on a result tracker."""

    # note: this is an internal utility class

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
class BestCheckpointSchedule(CheckpointSchedule):
    """Create a checkpoint whenever a metric improves."""

    #: the result tracker which receives updates on metrics
    #: since the same tracker instance needs to receive results from the training loop, we do require a pre-instantiated
    #: one rather than offering to provide hints, too
    result_tracker: ResultTracker

    #: the metric name
    metric: str

    #: the metric prefix; if None, do not check prefix
    prefix: str | None = None

    #: whether to maximize or minimize the metric
    maximize: bool = True

    # note: internal detail
    _adapter: ResultListenerAdapterResultTracker = dataclasses.field(init=False)

    def __post_init__(self):
        self._adapter = ResultListenerAdapterResultTracker(
            self.result_tracker, metric=self.metric, prefix=self.prefix, maximize=self.maximize
        )

    def __call__(self, step: int) -> bool:
        if self._adapter.last_step is None:
            raise ValueError(
                "The result tracker did not receive any results so far. Did you forget to use the same result "
                "tracker instance that is running in training?",
            )
        return step == self._adapter.best_step


@dataclasses.dataclass
class UnionCheckpointSchedule(CheckpointSchedule):
    """Create a checkpoint whenever one of the base schedules requests it."""

    bases: OneOrManyHintOrType
    bases_kwargs: OneOrManyOptionalKwargs = None

    _bases: Sequence[CheckpointSchedule] = dataclasses.field(init=False)

    def __post_init__(self):
        self._bases = schedule_resolver.make_many(self.bases, self.bases_kwargs)

    def __call__(self, step: int) -> bool:
        return any(base(step) for base in self._bases)


#: a resolver for checkpoint schedules
schedule_resolver = ClassResolver.from_subclasses(base=CheckpointSchedule, default=RegularCheckpointSchedule)


def inspect_schedule(
    num_epochs: int = 100, schedule: HintOrType[CheckpointSchedule] = None, schedule_kwargs: OptionalKwargs = None
) -> list[int]:
    """
    Simulate a checkpoint schedule and return the epochs for which a checkpoint would be written.

    >>> inspect_schedule(50)
    [10, 20, 30, 40, 50]
    >>> inspect_schedule(50, schedule="explicit", schedule_kwargs=dict(steps=[30, 35]))
    [30, 35]
    >>> inspect_schedule(
    ...     50,
    ...     schedule="union",
    ...     schedule_kwargs=dict(
    ...         bases=["regular", "explicit"],
    ...         bases_kwargs=[dict(frequency=15), dict(steps=[7,])],
    ...     ),
    ... )
    [7, 15, 30, 45]

    ..warning::
        You cannot easily inspect schedules which depend on training dynamics, e.g., :class:`BestCheckpointSchedule`.

    :param num_epochs:
        the number of epochs
    :param schedule:
        a checkpoint schedule instance or selection
    :param schedule_kwargs:
        additional keyword-based parameters when the schedule needs to instantiated first from a selection

    :return:
        a sorted list of epochs at which a checkpoint would be made
    """
    schedule_instance = schedule_resolver.make(schedule, schedule_kwargs)
    return list(filter(schedule_instance, range(1, num_epochs + 1)))
