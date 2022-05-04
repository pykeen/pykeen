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

import pathlib
from typing import Any, List, Optional

from class_resolver import ClassResolver, HintOrType, OptionalKwargs
from torch import optim
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from ..evaluation import Evaluator, evaluator_resolver
from ..losses import Loss
from ..models import Model
from ..stoppers import Stopper
from ..trackers import ResultTracker
from ..triples import CoreTriplesFactory
from ..typing import MappedTriples, OneOrSequence

__all__ = [
    "TrainingCallbackHint",
    "TrainingCallback",
    "StopperTrainingCallback",
    "TrackerTrainingCallback",
    "EvaluationTrainingCallback",
    "MultiTrainingCallback",
    "GradientNormClippingTrainingCallback",
    "GradientAbsClippingTrainingCallback",
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
    def model(self) -> Model:  # noqa:D401
        """The model, accessed via the training loop."""
        return self.training_loop.model

    @property
    def loss(self) -> Loss:  # noqa: D401
        """The loss, accessed via the training loop."""
        return self.training_loop.loss

    @property
    def optimizer(self) -> optim.Optimizer:  # noqa:D401
        """The optimizer, accessed via the training loop."""
        return self.training_loop.optimizer

    @property
    def result_tracker(self) -> ResultTracker:  # noqa: D401
        """The result tracker, accessed via the training loop."""
        assert self.training_loop.result_tracker is not None
        return self.training_loop.result_tracker

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


class TrackerTrainingCallback(TrainingCallback):
    """
    An adapter for the :class:`pykeen.trackers.ResultTracker`.

    It logs the loss after each epoch to the given result tracker,
    """

    # docstr-coverage: inherited
    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:  # noqa: D102
        self.result_tracker.log_metrics({"loss": epoch_loss}, step=epoch)


class GradientNormClippingTrainingCallback(TrainingCallback):
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

    # docstr-coverage: inherited
    def pre_step(self, **kwargs: Any) -> None:  # noqa: D102
        clip_grad_norm_(
            parameters=self.model.get_grad_params(),
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            error_if_nonfinite=True,  # this will become default in future releases of pytorch
        )


class GradientAbsClippingTrainingCallback(TrainingCallback):
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

    # docstr-coverage: inherited
    def pre_step(self, **kwargs: Any) -> None:  # noqa: D102
        clip_grad_value_(self.model.get_grad_params(), clip_value=self.clip_value)


class EvaluationTrainingCallback(TrainingCallback):
    """
    A callback for regular evaluation.

    Example: evaluate training performance

    .. code-block:: python

        from pykeen.datasets import get_dataset
        from pykeen.pipeline import pipeline

        dataset = get_dataset(dataset="nations")
        result = pipeline(
            dataset=dataset,
            model="mure",
            training_loop_kwargs=dict(
                result_tracker="console",
            ),
            training_kwargs=dict(
                num_epochs=100,
                callbacks="evaluation",
                callback_kwargs=dict(
                    evaluation_triples=dataset.training.mapped_triples,
                    prefix="training",
                ),
            ),
        )
    """

    def __init__(
        self,
        *,
        evaluation_triples: MappedTriples,
        frequency: int = 1,
        evaluator: HintOrType[Evaluator] = None,
        evaluator_kwargs: OptionalKwargs = None,
        prefix: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the callback.

        :param evaluation_triples:
            the triples on which to evaluate
        :param frequency:
            the evaluation frequency in epochs
        :param evaluator:
            the evaluator to use for evaluation, cf. `evaluator_resolver`
        :param evaluator_kwargs:
            additional keyword-based parameters for the evaluator
        :param prefix:
            the prefix to use for logging the metrics
        :param kwargs:
            additional keyword-based parameters passed to `evaluate`
        """
        super().__init__()
        self.frequency = frequency
        self.evaluation_triples = evaluation_triples
        self.evaluator = evaluator_resolver.make(evaluator, evaluator_kwargs)
        self.prefix = prefix
        self.kwargs = kwargs
        self.batch_size = self.kwargs.pop("batch_size", None)

    # docstr-coverage: inherited
    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:  # noqa: D102
        if epoch % self.frequency:
            return
        result = self.evaluator.evaluate(
            model=self.model,
            mapped_triples=self.evaluation_triples,
            device=self.training_loop.device,
            batch_size=self.evaluator.batch_size or self.batch_size,
            **self.kwargs,
        )
        self.result_tracker.log_metrics(metrics=result.to_flat_dict(), step=epoch, prefix=self.prefix)


class StopperTrainingCallback(TrainingCallback):
    """An adapter for the :class:`pykeen.stopper.Stopper`."""

    def __init__(
        self,
        stopper: Stopper,
        *,
        triples_factory: CoreTriplesFactory,
        last_best_epoch: Optional[int] = None,
        best_epoch_model_file_path: Optional[pathlib.Path],
    ):
        """
        Initialize the callback.

        :param stopper:
            the stopper
        :param triples_factory:
            the triples factory used for saving the state
        :param last_best_epoch:
            the last best epoch
        :param best_epoch_model_file_path:
            the path under which to store the best model checkpoint
        """
        super().__init__()
        self.stopper = stopper
        self.triples_factory = triples_factory
        self.last_best_epoch = last_best_epoch
        self.best_epoch_model_file_path = best_epoch_model_file_path

    # docstr-coverage: inherited
    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:  # noqa: D102
        if self.stopper.should_evaluate(epoch):
            # TODO how to pass inductive mode
            if self.stopper.should_stop(epoch):
                self.training_loop._should_stop = True
            # Since the model is also used within the stopper, its graph and cache have to be cleared
            self.model._free_graph_and_cache()
            # When the stopper obtained a new best epoch, this model has to be saved for reconstruction
        if self.stopper.best_epoch != self.last_best_epoch and self.best_epoch_model_file_path is not None:
            self.training_loop._save_state(path=self.best_epoch_model_file_path, triples_factory=self.triples_factory)
            self.last_best_epoch = epoch


callback_resolver: ClassResolver[TrainingCallback] = ClassResolver.from_subclasses(
    base=TrainingCallback,
)

#: A hint for constructing a :class:`MultiTrainingCallback`
TrainingCallbackHint = OneOrSequence[HintOrType[TrainingCallback]]
TrainingCallbackKwargsHint = OneOrSequence[OptionalKwargs]


class MultiTrainingCallback(TrainingCallback):
    """A wrapper for calling multiple training callbacks together."""

    #: A collection of callbacks
    callbacks: List[TrainingCallback]

    def __init__(
        self,
        callbacks: TrainingCallbackHint = None,
        callback_kwargs: TrainingCallbackKwargsHint = None,
    ) -> None:
        """
        Initialize the callback.

        .. note ::
            the constructor allows "broadcasting" of callbacks, i.e., proving a single callback,
            but a list of callback kwargs. In this case, for each element of this list the given
            callback is instantiated.

        :param callbacks:
            the callbacks
        :param callback_kwargs:
            additional keyword-based parameters for instantiating the callbacks
        """
        super().__init__()
        self.callbacks = callback_resolver.make_many(callbacks, callback_kwargs) if callbacks else []

    # docstr-coverage: inherited
    def register_training_loop(self, loop) -> None:  # noqa: D102
        super().register_training_loop(training_loop=loop)
        for callback in self.callbacks:
            callback.register_training_loop(training_loop=loop)

    def register_callback(self, callback: TrainingCallback) -> None:
        """Register a callback."""
        self.callbacks.append(callback)
        if self._training_loop is not None:
            callback.register_training_loop(self._training_loop)

    # docstr-coverage: inherited
    def on_batch(self, epoch: int, batch, batch_loss: float, **kwargs: Any) -> None:  # noqa: D102
        for callback in self.callbacks:
            callback.on_batch(epoch=epoch, batch=batch, batch_loss=batch_loss, **kwargs)

    # docstr-coverage: inherited
    def post_batch(self, epoch: int, batch, **kwargs: Any) -> None:  # noqa: D102
        for callback in self.callbacks:
            callback.post_batch(epoch=epoch, batch=batch, **kwargs)

    # docstr-coverage: inherited
    def pre_step(self, **kwargs: Any) -> None:  # noqa: D102
        for callback in self.callbacks:
            callback.pre_step(**kwargs)

    # docstr-coverage: inherited
    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:  # noqa: D102
        for callback in self.callbacks:
            callback.post_epoch(epoch=epoch, epoch_loss=epoch_loss, **kwargs)

    # docstr-coverage: inherited
    def post_train(self, losses: List[float], **kwargs: Any) -> None:  # noqa: D102
        for callback in self.callbacks:
            callback.post_train(losses=losses, **kwargs)
