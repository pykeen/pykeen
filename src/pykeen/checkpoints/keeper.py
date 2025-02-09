"""
Checkpoint cleanup methods.

The cleanup methods determine, for any given set of existing checkpoints, which of them can be pruned.
We provide a set of basic rules that can be easily combined into more complex logic.
"""

import abc
import dataclasses
from collections.abc import Collection, Iterator, Sequence

from class_resolver import ClassResolver, OneOrManyHintOrType, OneOrManyOptionalKwargs

from .utils import MetricSelection, ResultListenerAdapter
from ..trackers.base import ResultTracker

__all__ = [
    "CheckpointKeeper",
    "keeper_resolver",
    "LastCheckpointKeeper",
    "ModuloCheckpointKeeper",
    "ExplicitCheckpointKeeper",
    "BestCheckpointKeeper",
    "UnionCheckpointKeeper",
]


class CheckpointKeeper(abc.ABC):
    """A checkpoint cleanup interface."""

    @abc.abstractmethod
    def __call__(self, steps: Sequence[int]) -> Iterator[int]:
        """Iterate over the steps for which checkpoints should be kept.

        :param steps:
            the sorted list of steps at which checkpoints were written.

        :yields:
            the steps for which checkpoints should be kept
        """


@dataclasses.dataclass
class LastCheckpointKeeper(CheckpointKeeper):
    """Keep the last $n$ checkpoints."""

    #: the number of checkpoints to keep
    keep: int = 1

    def __call__(self, steps: Sequence[int]) -> Iterator[int]:
        yield from steps[-self.keep :]


@dataclasses.dataclass
class ModuloCheckpointKeeper(CheckpointKeeper):
    """Keep checkpoints if the step is divisible by a number."""

    divisor: int = 10

    def __call__(self, steps: Sequence[int]) -> Iterator[int]:
        for step in steps:
            if step % self.divisor == 0:
                yield step


@dataclasses.dataclass
class ExplicitCheckpointKeeper(CheckpointKeeper):
    """Keep checkpoints at explicit steps."""

    keep: Collection[int]

    def __post_init__(self):
        # convert to set for better lookup speed
        self.keep = set(self.keep)

    def __call__(self, steps: Sequence[int]) -> Iterator[int]:
        # the set operation should be a nop of sets
        yield from set(self.keep).intersection(steps)


@dataclasses.dataclass
class BestCheckpointKeeper(CheckpointKeeper):
    """Keep checkpoints for steps that achieved the best value for a metric."""

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

    def __call__(self, steps: Sequence[int]) -> Iterator[int]:
        return filter(self._adapter.is_best, steps)


@dataclasses.dataclass
class UnionCheckpointKeeper(CheckpointKeeper):
    """Keep a checkpoint where one of the criteria is met."""

    bases: OneOrManyHintOrType[CheckpointKeeper]
    bases_kwargs: OneOrManyOptionalKwargs = None

    _bases: Sequence[CheckpointKeeper] = dataclasses.field(init=False)

    def __post_init__(self):
        self._bases = keeper_resolver.make_many(self.bases, self.bases_kwargs)

    def __call__(self, steps: Sequence[int]) -> Iterator[int]:
        result: set[int] = set()
        for base in self._bases:
            result.update(base(steps))
        yield from result


#: a resolver for checkpoint keepers
keeper_resolver: ClassResolver[CheckpointKeeper] = ClassResolver.from_subclasses(
    CheckpointKeeper, default=CheckpointKeeper
)
