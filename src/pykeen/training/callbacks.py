# -*- coding: utf-8 -*-

"""Training callbacks.

Training callbacks allow for arbitrary extension of the functionality of the :class:`pykeen.training.TrainingLoop`
without subclassing it. Each callback instance has a ``loop`` attribute that allows access to the parent training
loop and all of its attributes, including the model. The interaction points are similar to those of
`Keras <https://keras.io/guides/writing_your_own_callbacks/#an-overview-of-callback-methods>`_.

Examples
--------
It was suggested in `Issue #333 <https://github.com/pykeen/pykeen/issues/333>`_ that it might
be useful to log all batch losses. This could be accomplished with the following:

.. code-block:: python

    from pykeen.training import TrainingCallback

    class BatchLossReportCallback(TrainingCallback):
        def on_batch(self, epoch: int, batch_loss: float):
            print(epoch, batch_loss)
"""

from typing import Collection, List, Union

from ..trackers import ResultTracker

__all__ = [
    'TrainingCallbackHint',
    'TrainingCallback',
    'TrackerCallback',
    'MultiTrainingCallback',
]


class TrainingCallback:
    """An interface for training callbacks."""

    def __init__(self):
        """Initialize the callback."""
        self._loop = None

    @property
    def loop(self):  # noqa:D401
        """The training loop."""
        if self._loop is None:
            raise ValueError('Callback was never initialized')
        return self._loop

    def register_loop(self, loop) -> None:
        """Register the training loop."""
        self._loop = loop

    def on_batch(self, epoch: int, batch_loss: float) -> None:
        """Call for training batches."""

    def post_batches(self, epoch: int, batch) -> None:
        """Call for training batches."""

    def post_epoch(self, epoch: int, epoch_loss: float) -> None:
        """Call after epoch."""

    def post_train(self, losses: List[float]) -> None:
        """Call after training."""


class TrackerCallback(TrainingCallback):
    """An adapter for the :class:`pykeen.trackers.ResultTracker`."""

    def __init__(self, result_tracker: ResultTracker):
        super().__init__()
        self.result_tracker = result_tracker

    def post_epoch(self, epoch: int, epoch_loss: float) -> None:
        """Log the epoch and loss."""
        self.result_tracker.log_metrics({'loss': epoch_loss}, step=epoch)


#: A hint for constructing a :class:`MultiTrainingCallback`
TrainingCallbackHint = Union[None, TrainingCallback, Collection[TrainingCallback]]


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

    def register_loop(self, loop) -> None:
        """Register the training loop."""
        super().register_loop(loop=loop)
        for callback in self.callbacks:
            callback.register_loop(loop=loop)

    def register_callback(self, callback: TrainingCallback) -> None:
        """Register a callback."""
        self.callbacks.append(callback)
        if self._loop is not None:
            callback.register_loop(self._loop)

    def on_batch(self, epoch: int, batch_loss: float) -> None:
        """Call for training batches."""
        for callback in self.callbacks:
            callback.on_batch(epoch=epoch, batch_loss=batch_loss)

    def post_batches(self, epoch: int, batch) -> None:
        """Call for training batches."""
        for callback in self.callbacks:
            callback.post_batches(epoch=epoch, batch=batch)

    def post_epoch(self, epoch: int, epoch_loss: float) -> None:
        """Call after epoch."""
        for callback in self.callbacks:
            callback.post_epoch(epoch=epoch, epoch_loss=epoch_loss)

    def post_train(self, losses: List[float]) -> None:
        """Call after training."""
        for callback in self.callbacks:
            callback.post_train(losses=losses)
