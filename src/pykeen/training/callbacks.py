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
        def on_batch(self, epoch: int, batch, batch_loss: float):
            print(epoch, batch_loss)
"""

from typing import Any, Collection, List, Union

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
        self._training_loop = None

    @property
    def training_loop(self):  # noqa:D401
        """The training loop."""
        if self._training_loop is None:
            raise ValueError('Callback was never initialized')
        return self._training_loop

    @property
    def model(self):  # noqa:D401
        """The model, accessed via the training loop."""
        return self.training_loop.model

    @property
    def loss(self):  # noqa: D401
        """The loss, accessed via the training loop."""
        return self.training_loop.loss

    @property
    def optimizer(self):  # noqa:D401
        """The optimizer, accessed via the training loop."""
        return self.training_loop.optimizer

    def register_training_loop(self, training_loop) -> None:
        """Register the training loop."""
        self._training_loop = training_loop

    def on_batch(self, epoch: int, batch, batch_loss: float, **kwargs: Any) -> None:
        """Call for training batches."""

    def post_batch(self, epoch: int, batch, **kwargs: Any) -> None:
        """Call for training batches."""

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:
        """Call after epoch."""

    def post_train(self, losses: List[float], **kwargs: Any) -> None:
        """Call after training."""


class TrackerCallback(TrainingCallback):
    """An adapter for the :class:`pykeen.trackers.ResultTracker`."""

    def __init__(self, result_tracker: ResultTracker):
        super().__init__()
        self.result_tracker = result_tracker

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:
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

    def register_training_loop(self, loop) -> None:
        """Register the training loop."""
        super().register_training_loop(training_loop=loop)
        for callback in self.callbacks:
            callback.register_training_loop(training_loop=loop)

    def register_callback(self, callback: TrainingCallback) -> None:
        """Register a callback."""
        self.callbacks.append(callback)
        if self._training_loop is not None:
            callback.register_training_loop(self._training_loop)

    def on_batch(self, epoch: int, batch, batch_loss: float, **kwargs: Any) -> None:
        """Call for each batch."""
        for callback in self.callbacks:
            callback.on_batch(epoch=epoch, batch=batch, batch_loss=batch_loss)

    def post_batch(self, epoch: int, batch, **kwargs: Any) -> None:
        """Call after each batch."""
        for callback in self.callbacks:
            callback.post_batch(epoch=epoch, batch=batch)

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:
        """Call after epoch."""
        for callback in self.callbacks:
            callback.post_epoch(epoch=epoch, epoch_loss=epoch_loss)

    def post_train(self, losses: List[float], **kwargs: Any) -> None:
        """Call after training."""
        for callback in self.callbacks:
            callback.post_train(losses=losses)
