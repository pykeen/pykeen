"""Scheduling when to make checkpoints."""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Sequence
from typing import Protocol

from class_resolver import ClassResolver, OneOrManyHintOrType, OneOrManyOptionalKwargs

from .utils import MetricSelection, ResultListenerAdapterResultTracker
from ..trackers.base import ResultTracker

__all__ = [
    "CheckpointSchedule",
    "schedule_resolver",
]


class CheckpointSchedule(Protocol):
    """Determine whether to create a checkpoint at the given epoch."""

    def __call__(self, step: int) -> bool: ...


@dataclasses.dataclass
class EveryCheckpointSchedule(CheckpointSchedule):
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
                    schedule="every",
                    schedule_kwargs=dict(
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
                    schedule="explicit",
                    schedule_kwargs=dict(
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
class BestCheckpointSchedule(CheckpointSchedule):
    """
    Create a checkpoint whenever a metric improves.

    Example::

        from pykeen.checkpoints import MetricSelection
        from pykeen.pipeline import pipeline
        from pykeen.trackers import tracker_resolver

        # create a default result tracker (or use a proper one)
        result_tracker = tracker_resolver.make(None)
        result = pipeline(
            dataset="nations",
            model="mure",
            training_kwargs=dict(
                num_epochs=10,
                callbacks="checkpoint",
                callbacks_kwargs=dict(
                    schedule="best",
                    schedule_kwargs=dict(
                        result_tracker=result_tracker,
                        # in this example, we just use the training loss
                        metric_selection=MetricSelection(
                            metric="loss,
                            maximize=False,
                        )
                    ),
                ),
            ),
            # Important: use the same result tracker instance as in the checkpoint callback
            result_tracker=result_tracker
        )
    """

    #: the result tracker which receives updates on metrics
    #: since the same tracker instance needs to receive results from the training loop, we do require a pre-instantiated
    #: one rather than offering to provide hints, too
    result_tracker: ResultTracker

    #: the metric selection
    metric_selection: MetricSelection

    # note: internal detail
    _adapter: ResultListenerAdapterResultTracker = dataclasses.field(init=False)

    def __post_init__(self):
        self._adapter = ResultListenerAdapterResultTracker(self.result_tracker, metric_selection=self.metric_selection)

    def __call__(self, step: int) -> bool:
        return self._adapter.is_best(step)


@dataclasses.dataclass
class UnionCheckpointSchedule(CheckpointSchedule):
    """
    Create a checkpoint whenever one of the base schedules requests it.

    Example::

        from pykeen.pipeline import

        result = pipeline(
            dataset="nations",
            model="mure",
            training_kwargs=dict(
                num_epochs=10,
                callbacks="checkpoint",
                callbacks_kwargs=dict(
                    schedule="union",
                    # create checkpoints every 5 epochs, and at epoch 7
                    schedule_kwargs=dict(
                        bases=["every", "explicit"],
                        bases_kwargs=[dict(frequency=5), dict(steps=[7])]
                    ),
                )
            ),
        )
    """

    bases: OneOrManyHintOrType[CheckpointSchedule]
    bases_kwargs: OneOrManyOptionalKwargs = None

    _bases: Sequence[CheckpointSchedule] = dataclasses.field(init=False)

    def __post_init__(self):
        self._bases = schedule_resolver.make_many(self.bases, self.bases_kwargs)

    def __call__(self, step: int) -> bool:
        return any(base(step) for base in self._bases)


#: a resolver for checkpoint schedules
schedule_resolver = ClassResolver.from_subclasses(base=CheckpointSchedule, default=EveryCheckpointSchedule)
