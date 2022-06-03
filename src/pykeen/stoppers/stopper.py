# -*- coding: utf-8 -*-

"""Basic stoppers."""

import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional, Union

import torch

from ..typing import InductiveMode

__all__ = [
    "Stopper",
    "NopStopper",
]

logger = logging.getLogger(__name__)


class Stopper(ABC):
    """A harness for stopping training."""

    def __init__(self, *args, **kwargs):
        """
        Initialize the stopper.

        :param args:
            ignored positional parameters
        :param args:
            ignored keyword-based parameters
        """
        # To make MyPy happy
        self.best_epoch = None

    def should_evaluate(self, epoch: int) -> bool:
        """Check if the stopper should be evaluated on the given epoch."""
        raise NotImplementedError

    @abstractmethod
    def should_stop(self, epoch: int, *, mode: Optional[InductiveMode] = None) -> bool:
        """Validate on validation set and check for termination condition."""
        raise NotImplementedError

    @abstractmethod
    def get_summary_dict(self) -> Mapping[str, Any]:
        """Get a summary dict."""
        raise NotImplementedError

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
    ):
        pass

    @staticmethod
    def load_summary_dict_from_training_loop_checkpoint(path: Union[str, pathlib.Path]) -> Mapping[str, Any]:
        """Load the summary dict from a training loop checkpoint.

        :param path:
            Path of the file where to store the state in.

        :return:
            The summary dict of the stopper at the time of saving the checkpoint.
        """
        logger.info(f"=> loading stopper summary dict from training loop checkpoint in '{path}'")
        checkpoint = torch.load(path)
        logger.info(f"=> loaded stopper summary dictionary from checkpoint in '{path}'")
        return checkpoint["stopper_dict"]


class NopStopper(Stopper):
    """A stopper that does nothing."""

    def should_evaluate(self, epoch: int) -> bool:
        """Return false; should never evaluate."""
        return False

    def should_stop(self, epoch: int, *, mode: Optional[InductiveMode] = None) -> bool:
        """Return false; should never stop."""
        return False

    def get_summary_dict(self) -> Mapping[str, Any]:
        """Return empty mapping, doesn't have any attributes."""
        return dict()
