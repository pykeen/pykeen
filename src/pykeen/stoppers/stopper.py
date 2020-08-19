# -*- coding: utf-8 -*-

"""Basic stoppers."""

from abc import ABC, abstractmethod

__all__ = [
    'Stopper',
    'NopStopper',
]


class Stopper(ABC):
    """A harness for stopping training."""

    def __init__(self, *args, **kwargs):
        pass

    def should_evaluate(self, epoch: int) -> bool:
        """Check if the stopper should be evaluated on the given epoch."""
        raise NotImplementedError

    @abstractmethod
    def should_stop(self, epoch: int) -> bool:
        """Validate on validation set and check for termination condition."""
        raise NotImplementedError


class NopStopper(Stopper):
    """A stopper that does nothing."""

    def should_evaluate(self, epoch: int) -> bool:
        """Return false; should never evaluate."""
        return False

    def should_stop(self, epoch: int) -> bool:
        """Return false; should never stop."""
        return False
