"""Method to clean up checkpoints."""

import abc
import dataclasses
from collections.abc import Collection, Iterator, Sequence

from class_resolver import ClassResolver, OneOrManyHintOrType, OneOrManyOptionalKwargs

from .utils import MetricSelection, ResultListenerAdapterResultTracker
from ..trackers.base import ResultTracker

__all__ = [
    "CheckpointKeeper",
    "keeper_resolver",
]


class CheckpointKeeper(abc.ABC):
    """An interface for checkpoint cleanup."""

    @abc.abstractmethod
    def __call__(self, steps: Sequence[int]) -> Iterator[int]:
        """Iterate over the steps for which checkpoints should be kept.

        :param steps:
            the sorted list of steps at which checkpoints were written.
        """
        raise NotImplementedError


@dataclasses.dataclass
class LastCheckpointKeeper(CheckpointKeeper):
    """Keep the last $n$ checkpoints."""

    keep: int = 1

    def __call__(self, steps: Sequence[int]) -> Iterator[int]:  # noqa: D102
        yield from steps[-self.keep :]


@dataclasses.dataclass
class ModuloCheckpointKeeper(CheckpointKeeper):
    """Keep a checkpoint at regularly placed steps."""

    modulo: int = 10

    def __call__(self, steps: Sequence[int]) -> Iterator[int]:  # noqa: D102
        for step in steps:
            if step % self.modulo == 0:
                yield step


@dataclasses.dataclass
class ExplicitCheckpointKeeper(CheckpointKeeper):
    """Keep the checkpoints at explicitly given steps."""

    keep: Collection[int]

    def __post_init__(self):
        # convert to set for better lookup speed
        self.keep = set(self.keep)

    def __call__(self, steps: Sequence[int]) -> Iterator[int]:  # noqa: D102
        keep = self.keep
        if not isinstance(keep, set):
            keep = set(keep)
        yield from keep.intersection(steps)


@dataclasses.dataclass
class BestCheckpointKeeper(CheckpointKeeper):
    """Keep checkpoints for the best value in a given metric."""

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

    def __call__(self, steps: Sequence[int]) -> Iterator[int]:
        return filter(self._adapter.is_best, steps)


@dataclasses.dataclass
class UnionCheckpointKeeper(CheckpointKeeper):
    """Keep checkpoint where any of the criteria is fulfilled."""

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
