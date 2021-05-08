# -*- coding: utf-8 -*-

"""Training callbacks."""

from __future__ import annotations

from typing import Collection, List, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .training_loop import TrainingLoop

__all__ = [
    'TrainingCallbackHint',
    'TrainingCallback',
    'MultiTrainingCallback',
]

#: A hint for constructing a :class:`MultiTrainingCallback`
TrainingCallbackHint = Union[None, 'TrainingCallback', Collection['TrainingCallback']]


class TrainingCallback:
    """An interface for training callbacks.

    The interaction points are similar to those of
    `Keras <https://keras.io/guides/writing_your_own_callbacks/#an-overview-of-callback-methods>`_.

    The training callback is registered with the training loop, and can access its attributes.
    """

    def __init__(self):
        """Initialize the callback."""
        self._loop = None

    @property
    def loop(self) -> 'TrainingLoop':  # noqa:D401
        """The training loop."""
        if self._loop is None:
            raise ValueError('Callback was never initialized')
        return self._loop

    def register_loop(self, *, loop: 'TrainingLoop') -> None:
        """Register the training loop."""
        self._loop = loop

    def post_batch(self, *, batch) -> None:
        """Call for training batches."""

    def post_epoch(self, *, epoch: int, loss: float) -> None:
        """Call after epoch."""

    def post_train(self, *, losses: List[float]) -> None:
        """Call after training."""


class MultiTrainingCallback(TrainingCallback):
    """A wrapper for calling multiple training callbacks together."""

    #: A collection of callbacks
    callbacks: List[TrainingCallback]

    def __init__(self, callbacks: TrainingCallbackHint = None) -> None:
        """Initialize the callback."""
        super().__init__()
        if callbacks is None:
            self.callbacks = []
        elif isinstance(callbacks, TrainingCallback):
            self.callbacks = [callbacks]
        else:
            self.callbacks = list(callbacks)

    def register_loop(self, *, loop: 'TrainingLoop') -> None:
        """Register the training loop."""
        super().register_loop(loop=loop)
        for callback in self.callbacks:
            callback.register_loop(loop=loop)

    def register_callback(self, callback: TrainingCallback) -> None:
        """Register a callback."""
        self.callbacks.append(callback)

    def post_batch(self, *, batch) -> None:
        """Call for training batches."""
        for callback in self.callbacks:
            callback.post_batch(batch=batch)

    def post_epoch(self, *, epoch: int, loss: float) -> None:
        """Call after epoch."""
        for callback in self.callbacks:
            callback.post_epoch(epoch=epoch, loss=loss)

    def post_train(self, *, losses: List[float]) -> None:
        """Call after training."""
        for callback in self.callbacks:
            callback.post_train(losses=losses)
