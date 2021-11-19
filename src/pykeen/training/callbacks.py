# -*- coding: utf-8 -*-

"""Training callbacks.

Training callbacks allow for arbitrary extension of the functionality of the :class:`pykeen.training.TrainingLoop`
without subclassing it. Each callback instance has a ``loop`` attribute that allows access to the parent training
loop and all of its attributes, including the model. The interaction points are similar to those of
`Keras <https://keras.io/guides/writing_your_own_callbacks/#an-overview-of-callback-methods>`_.

Examples
--------
The following are vignettes showing how PyKEEN's training loop can be arbitrarily extended
using callbacks. If you find that none of the hooks in the :class:`TrainingCallback`
help do what you want, feel free to open an issue.

Reporting Batch Loss
~~~~~~~~~~~~~~~~~~~~
It was suggested in `Issue #333 <https://github.com/pykeen/pykeen/issues/333>`_ that it might
be useful to log all batch losses. This could be accomplished with the following:

.. code-block:: python

    from pykeen.training import TrainingCallback

    class BatchLossReportCallback(TrainingCallback):
        def on_batch(self, epoch: int, batch, batch_loss: float):
            print(epoch, batch_loss)

Implementing Gradient Clipping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`Gradient
clipping <https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem>`_
is one technique used to avoid the exploding gradient problem. Despite it being a very simple, it has several
`theoretical implications <https://openreview.net/forum?id=BJgnXpVYwS>`_.

In order to reproduce the reference experiments on R-GCN performed by [schlichtkrull2018]_,
gradient clipping must be used before each step of the optimizer. The following example shows how
to implement a gradient clipping callback:

.. code-block:: python

    from pykeen.training import TrainingCallback
    from pykeen.nn.utils import clip_grad_value_

    class GradientClippingCallback(TrainingCallback):
        def __init__(self, clip_value: float = 1.0):
            super().__init__()
            self.clip_value = clip_value

        def pre_step(self, **kwargs: Any):
            clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)
"""

from typing import Any, Collection, List, Optional, Union

from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from ..trackers import ResultTracker

__all__ = [
    "TrainingCallbackHint",
    "TrainingCallback",
    "TrackerCallback",
    "MultiTrainingCallback",
    "GradientNormClippingCallback",
    "GradientAbsClippingCallback",
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
            raise ValueError("Callback was never initialized")
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

    def pre_step(self, **kwargs: Any) -> None:
        """Call before the optimizer's step."""

    def post_batch(self, epoch: int, batch, **kwargs: Any) -> None:
        """Call for training batches."""

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:
        """Call after epoch."""

    def post_train(self, losses: List[float], **kwargs: Any) -> None:
        """Call after training."""


class TrackerCallback(TrainingCallback):
    """
    An adapter for the :class:`pykeen.trackers.ResultTracker`.

    It logs the loss after each epoch to the given result tracker,
    """

    def __init__(self, result_tracker: ResultTracker):
        """
        Initialize the callback.

        :param result_tracker:
            The result tracker to which the loss is logged.
        """
        super().__init__()
        self.result_tracker = result_tracker

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:  # noqa: D102
        self.result_tracker.log_metrics({"loss": epoch_loss}, step=epoch)


class GradientNormClippingCallback(TrainingCallback):
    """A callback for gradient clipping before stepping the optimizer with :func:`torch.nn.utils.clip_grad_norm_`."""

    def __init__(self, max_norm: float, norm_type: Optional[float] = None):
        """
        Initialize the callback.

        :param max_norm:
            The maximum gradient norm for use with gradient clipping.
        :param norm_type:
            The gradient norm type to use for maximum gradient norm, cf. :func:`torch.nn.utils.clip_grad_norm_`
        """
        super().__init__()
        self.max_norm = max_norm
        self.norm_type = norm_type or 2.0

    def pre_step(self, **kwargs: Any) -> None:  # noqa: D102
        clip_grad_norm_(
            parameters=self.model.get_grad_params(),
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            error_if_nonfinite=True,  # this will become default in future releases of pytorch
        )


class GradientAbsClippingCallback(TrainingCallback):
    """A callback for gradient clipping before stepping the optimizer with :func:`torch.nn.utils.clip_grad_value_`."""

    def __init__(self, clip_value: float):
        """
        Initialize the callback.

        :param clip_value:
            The maximum absolute value in gradients, cf. :func:`torch.nn.utils.clip_grad_value_`. If None, no
            gradient clipping will be used.
        """
        super().__init__()
        self.clip_value = clip_value

    def pre_step(self, **kwargs: Any) -> None:  # noqa: D102
        clip_grad_value_(self.model.get_grad_params(), clip_value=self.clip_value)


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

    def register_training_loop(self, loop) -> None:  # noqa: D102
        super().register_training_loop(training_loop=loop)
        for callback in self.callbacks:
            callback.register_training_loop(training_loop=loop)

    def register_callback(self, callback: TrainingCallback) -> None:
        """Register a callback."""
        self.callbacks.append(callback)
        if self._training_loop is not None:
            callback.register_training_loop(self._training_loop)

    def on_batch(self, epoch: int, batch, batch_loss: float, **kwargs: Any) -> None:  # noqa: D102
        for callback in self.callbacks:
            callback.on_batch(epoch=epoch, batch=batch, batch_loss=batch_loss, **kwargs)

    def post_batch(self, epoch: int, batch, **kwargs: Any) -> None:  # noqa: D102
        for callback in self.callbacks:
            callback.post_batch(epoch=epoch, batch=batch, **kwargs)

    def pre_step(self, **kwargs: Any) -> None:  # noqa: D102
        for callback in self.callbacks:
            callback.pre_step(**kwargs)

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:  # noqa: D102
        for callback in self.callbacks:
            callback.post_epoch(epoch=epoch, epoch_loss=epoch_loss, **kwargs)

    def post_train(self, losses: List[float], **kwargs: Any) -> None:  # noqa: D102
        for callback in self.callbacks:
            callback.post_train(losses=losses, **kwargs)
