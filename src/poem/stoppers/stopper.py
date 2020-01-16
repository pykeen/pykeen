# -*- coding: utf-8 -*-

"""Basic stoppers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..evaluation import Evaluator
from ..models.base import BaseModule
from ..triples import TriplesFactory
from ..utils import ResultTracker

__all__ = [
    'Stopper',
    'NopStopper',
]


@dataclass
class Stopper(ABC):
    """A harness for stopping training."""

    #: The model
    model: BaseModule
    #: The evaluator
    evaluator: Evaluator
    #: The triples to use for evaluation
    evaluation_triples_factory: Optional[TriplesFactory]
    #: The result tracker
    result_tracker: Optional[ResultTracker] = None

    def should_evaluate(self, epoch: int) -> bool:
        """Check if the stopper should be evaluated on the given epoch."""
        raise NotImplementedError

    @abstractmethod
    def should_stop(self) -> bool:
        """Validate on validation set and check for termination condition."""
        raise NotImplementedError


@dataclass
class NopStopper(Stopper):
    """A stopper that does nothing."""

    def should_evaluate(self, epoch: int) -> bool:  # noqa: D102
        return False

    def should_stop(self) -> bool:  # noqa: D102
        return False
