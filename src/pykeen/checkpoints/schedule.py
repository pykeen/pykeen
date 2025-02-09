"""Scheduling when to make checkpoints."""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Collection, Sequence

from class_resolver import ClassResolver, OneOrManyHintOrType, OneOrManyOptionalKwargs

from .utils import MetricSelection, ResultListenerAdapter
from ..trackers.base import ResultTracker

__all__ = [
    "CheckpointSchedule",
    "schedule_resolver",
    "EveryCheckpointSchedule",
    "ExplicitCheckpointSchedule",
    "BestCheckpointSchedule",
    "UnionCheckpointSchedule",
]


class CheckpointSchedule(abc.ABC):
    """Interface for checkpoint schedules."""

    @abc.abstractmethod
    def __call__(self, step: int) -> bool:
        """Decide whether to create a checkpoint at the specified epoch."""


@dataclasses.dataclass
class EveryCheckpointSchedule(CheckpointSchedule):
    """Create a checkpoint every $n$ steps."""

    #: The checkpoint frequency
    frequency: int = 10

    def __call__(self, step: int) -> bool:
        return not step % self.frequency


@dataclasses.dataclass
class ExplicitCheckpointSchedule(CheckpointSchedule):
    """Create a checkpoint for explicitly chosen steps."""

    steps: Collection[int]

    def __call__(self, step: int) -> bool:
        return step in self.steps


@dataclasses.dataclass
class BestCheckpointSchedule(CheckpointSchedule):
    """Create a checkpoint whenever a metric improves."""

    #: the result tracker which receives updates on metrics
    #: since the same tracker instance needs to receive results from the training loop, we do require a pre-instantiated
    #: one rather than offering to provide hints, too
    result_tracker: ResultTracker

    #: the metric selection
    metric_selection: MetricSelection

    # note: internal detail
    _adapter: ResultListenerAdapter = dataclasses.field(init=False)

    def __post_init__(self):
        self._adapter = ResultListenerAdapter(self.result_tracker, metric_selection=self.metric_selection)

    def __call__(self, step: int) -> bool:
        return self._adapter.is_best(step)


@dataclasses.dataclass
class UnionCheckpointSchedule(CheckpointSchedule):
    """Create a checkpoint whenever one of the base schedules requires it."""

    bases: OneOrManyHintOrType[CheckpointSchedule]
    bases_kwargs: OneOrManyOptionalKwargs = None

    _bases: Sequence[CheckpointSchedule] = dataclasses.field(init=False)

    def __post_init__(self):
        self._bases = schedule_resolver.make_many(self.bases, self.bases_kwargs)

    def __call__(self, step: int) -> bool:
        return any(base(step) for base in self._bases)


#: a resolver for checkpoint schedules
schedule_resolver = ClassResolver.from_subclasses(base=CheckpointSchedule, default=EveryCheckpointSchedule)
