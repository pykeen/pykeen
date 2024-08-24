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

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from class_resolver import ClassResolver, HintOrType, OptionalKwargs
from torch import optim
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch_max_mem import maximize_memory_utilization

from .. import training  # required for type annotations
from ..evaluation import Evaluator, evaluator_resolver
from ..evaluation.evaluation_loop import AdditionalFilterTriplesHint, LCWAEvaluationLoop
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
    "EvaluationLoopTrainingCallback",
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
    def training_loop(self) -> training.TrainingLoop:  # noqa:D401
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

    def register_training_loop(self, training_loop: training.TrainingLoop) -> None:
        """Register the training loop."""
        self._training_loop = training_loop

    def pre_batch(self, **kwargs: Any) -> None:
        """Call before training batch."""

    def on_batch(self, epoch: int, batch, batch_loss: float, **kwargs: Any) -> None:
        """Call for training batches."""

    def pre_step(self, **kwargs: Any) -> None:
        """Call before the optimizer's step."""

    def post_batch(self, epoch: int, batch, **kwargs: Any) -> None:
        """Call for training batches."""

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:
        """Call after epoch."""

    def post_train(self, losses: list[float], **kwargs: Any) -> None:
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

    def __init__(self, max_norm: float, norm_type: float | None = None):
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
        prefix: str | None = None,
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


class EvaluationLoopTrainingCallback(TrainingCallback):
    """A callback for regular evaluation using new-style evaluation loops."""

    def __init__(
        self,
        factory: CoreTriplesFactory,
        frequency: int = 1,
        prefix: str | None = None,
        evaluator: HintOrType[Evaluator] = None,
        evaluator_kwargs: OptionalKwargs = None,
        additional_filter_triples: AdditionalFilterTriplesHint = None,
        **kwargs,
    ):
        """
        Initialize the callback.

        :param factory:
            the triples factory comprising the evaluation triples
        :param frequency:
            the evaluation frequency
        :param prefix:
            a prefix to use for logging (e.g., to distinguish between different splits)
        :param evaluator:
            the evaluator, or a hint thereof
        :param evaluator_kwargs:
            additional keyword-based parameters used for the evaluation instantiation
        :param additional_filter_triples:
            additional filter triples to use for creating the filter
        :param kwargs:
            additional keyword-based parameters passed to :meth:`EvaluationLoop.evaluate`
        """
        super().__init__()
        self.frequency = frequency
        self.prefix = prefix

        self.factory = factory
        self.evaluator = evaluator_resolver.make(evaluator, evaluator_kwargs)
        # lazy init
        self._evaluation_loop = None
        self.kwargs = kwargs
        self.additional_filter_triples = additional_filter_triples

    @property
    def evaluation_loop(self):
        """Return the evaluation loop instance (lazy-initialization)."""
        if self._evaluation_loop is None:
            self._evaluation_loop = LCWAEvaluationLoop(
                triples_factory=self.factory,
                evaluator=self.evaluator,
                model=self.model,
                additional_filter_triples=self.additional_filter_triples,
            )
        return self._evaluation_loop

    # docstr-coverage: inherited
    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:  # noqa: D102
        if epoch % self.frequency:
            return
        result = self.evaluation_loop.evaluate(**self.kwargs)
        self.result_tracker.log_metrics(metrics=result.to_flat_dict(), step=epoch, prefix=self.prefix)


class StopperTrainingCallback(TrainingCallback):
    """An adapter for the :class:`pykeen.stopper.Stopper`."""

    def __init__(
        self,
        stopper: Stopper,
        *,
        triples_factory: CoreTriplesFactory,
        last_best_epoch: int | None = None,
        best_epoch_model_file_path: pathlib.Path | None,
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


class OptimizerTrainingCallback(TrainingCallback):
    """Use optimizer to update parameters."""

    # TODO: we may want to separate TrainingCallback from pre-step callbacks in the future
    def __init__(self, only_size_probing: bool = False, pre_step_callbacks: Sequence[TrainingCallback] | None = None):
        """Initialize the callback.

        :param only_size_probing:
            whether this is during size probing, where we do not want to apply weight changes
        :param pre_step_callbacks:
            callbacks to apply before making the step, e.g., for gradient clipping.
        """
        super().__init__()
        self.only_size_probing = only_size_probing
        self.pre_step_callbacks = tuple(pre_step_callbacks or [])

    # docstr-coverage: inherited
    def pre_batch(self, **kwargs: Any) -> None:  # noqa: D102
        # Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old instance

        # note: we want to run this step during size probing to cleanup any remaining grads
        self.optimizer.zero_grad(set_to_none=True)

    # docstr-coverage: inherited
    def post_batch(self, epoch: int, batch, **kwargs: Any) -> None:  # noqa: D102
        # pre-step callbacks
        for cb in self.pre_step_callbacks:
            cb.pre_step(epoch=epoch, **kwargs)

        # when called by batch_size_search(), the parameter update should not be applied.
        if not self.only_size_probing:
            # update parameters according to optimizer
            self.optimizer.step()

        # After changing applying the gradients to the embeddings, the model is notified that the forward
        # constraints are no longer applied
        # note: we want to apply this during size probing to properly account for the memory necessary for e.g.,
        # regularization
        self.model.post_parameter_update()


class LearningRateSchedulerTrainingCallback(TrainingCallback):
    """Update learning rate scheduler."""

    # docstr-coverage: inherited
    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:  # noqa: D102
        if self.training_loop.lr_scheduler is None:
            raise ValueError(f"{self} can only be called when a learning rate schedule is used.")
        self.training_loop.lr_scheduler.step(epoch=epoch)


def _hasher(kwargs: Mapping[str, Any]) -> int:
    # do not share optimal parameters across different training loops
    return id(kwargs["training_loop"])


@maximize_memory_utilization(parameter_name=("batch_size", "slice_size"), hasher=_hasher)
@torch.inference_mode()
def _validation_loss_amo_wrapper(
    training_loop: training.TrainingLoop,
    triples_factory: CoreTriplesFactory,
    batch_size: int,
    slice_size: int,
    label_smoothing: float,
    epoch: int,
    callback: MultiTrainingCallback,
    **kwargs,
) -> float:
    """Calculate validation loss with automatic batch size optimization."""
    return training_loop._train_epoch(
        # todo: create dataset only once
        batches=training_loop._create_training_data_loader(
            triples_factory=triples_factory, batch_size=batch_size, drop_last=False, **kwargs
        ),
        label_smoothing=label_smoothing,
        callbacks=callback,
        epoch=epoch,
        # no sub-batching (for evaluation, we can just reduce batch size without any effect)
        sub_batch_size=None,
        slice_size=slice_size if training_loop.supports_slicing else None,
        # this is handled by the AMO wrapper
        only_size_probing=False,
        # no backward passes
        backward=False,
    )


class EvaluationLossTrainingCallback(TrainingCallback):
    """
    Calculate loss on an evaluation set.

    .. code-block ::

        from pykeen.datasets import get_dataset
        from pykeen.pipeline import pipeline

        dataset = get_dataset(dataset="nations")
        pipeline(
            dataset=dataset,
            model="mure",
            training_kwargs=dict(
                callbacks="evaluation-loss",
                callback_kwargs=dict(triples_factory=dataset.validation),
                prefix="validation",
            ),
            result_tracker="console",
        )
    """

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        callbacks: TrainingCallbackHint = None,
        callbacks_kwargs: TrainingCallbackKwargsHint = None,
        maximum_batch_size: int | None = None,
        label_smoothing: float = 0.0,
        data_loader_kwargs: Mapping[str, Any] | None = None,
        prefix: str = "validation",
    ):
        """
        Initialize the callback.

        :param triples_factory:
            the evaluation triples factory
        :param callbacks:
            callbacks for the validation loss loop
        :param callbacks_kwargs:
            keyword-based parameters for the callbacks of the validation loss loop
        :param maximum_batch_size:
            the maximum batch size
        :param label_smoothing:
            the label smoothing to use; usually this should be matched with the training settings
        :param data_loader_kwargs:
            the keyword based parameters for the data loader
        :param prefix:
            the prefix to use for logging
        """
        super().__init__()
        self.triples_factory = triples_factory
        self.prefix = prefix
        self.label_smoothing = label_smoothing
        if data_loader_kwargs is None:
            data_loader_kwargs = dict(sampler=None)
        self.data_loader_kwargs = data_loader_kwargs
        self.maximum_batch_size = maximum_batch_size
        self.callback = MultiTrainingCallback(callbacks=callbacks, callbacks_kwargs=callbacks_kwargs)

    # docstr-coverage: inherited
    def register_training_loop(self, training_loop: training.TrainingLoop) -> None:  # noqa: D102
        super().register_training_loop(training_loop)
        self.callback.register_training_loop(training_loop=training_loop)

    # docstr-coverage: inherited
    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:  # noqa: D102
        from .lcwa import LCWATrainingLoop

        # set to evaluation mode
        self.model.eval()

        # determine maximum batch size
        maximum_batch_size = self.maximum_batch_size or self.triples_factory.num_triples
        if self.model.device.type != "cuda":
            # try to avoid OOM kills on cpu for large datasets
            maximum_batch_size = min(maximum_batch_size, 2**16)

        loss = _validation_loss_amo_wrapper(
            training_loop=self.training_loop,
            triples_factory=self.triples_factory,
            # TODO: this should be num_instances rather than num_triples; also for cpu, we may want to reduce this
            batch_size=maximum_batch_size,
            # note: slicing is only effective for LCWA training
            slice_size=self.training_loop.num_targets if isinstance(self.training_loop, LCWATrainingLoop) else 1,
            label_smoothing=self.label_smoothing,
            callback=self.callback,
            epoch=epoch,
            **self.data_loader_kwargs,
        )
        self.result_tracker.log_metrics(metrics=dict(loss=loss), step=epoch, prefix=self.prefix)


#: A hint for constructing a :class:`MultiTrainingCallback`
TrainingCallbackHint = OneOrSequence[HintOrType[TrainingCallback]]
TrainingCallbackKwargsHint = OneOrSequence[OptionalKwargs]


class MultiTrainingCallback(TrainingCallback):
    """A wrapper for calling multiple training callbacks together."""

    #: A collection of callbacks
    callbacks: list[TrainingCallback]

    def __init__(
        self,
        callbacks: TrainingCallbackHint = None,
        callbacks_kwargs: TrainingCallbackKwargsHint = None,
    ) -> None:
        """
        Initialize the callback.

        .. note ::
            the constructor allows "broadcasting" of callbacks, i.e., proving a single callback,
            but a list of callback kwargs. In this case, for each element of this list the given
            callback is instantiated.

        :param callbacks:
            the callbacks
        :param callbacks_kwargs:
            additional keyword-based parameters for instantiating the callbacks
        """
        super().__init__()
        self.callbacks = callback_resolver.make_many(callbacks, callbacks_kwargs) if callbacks else []

    # docstr-coverage: inherited
    def register_training_loop(self, training_loop: training.TrainingLoop) -> None:  # noqa: D102
        super().register_training_loop(training_loop=training_loop)
        for callback in self.callbacks:
            callback.register_training_loop(training_loop=training_loop)

    def register_callback(self, callback: TrainingCallback) -> None:
        """Register a callback."""
        self.callbacks.append(callback)
        if self._training_loop is not None:
            callback.register_training_loop(self._training_loop)

    # docstr-coverage: inherited
    def pre_batch(self, **kwargs: Any) -> None:  # noqa: D102
        for callback in self.callbacks:
            callback.pre_batch(**kwargs)

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
    def post_train(self, losses: list[float], **kwargs: Any) -> None:  # noqa: D102
        for callback in self.callbacks:
            callback.post_train(losses=losses, **kwargs)


callback_resolver: ClassResolver[TrainingCallback] = ClassResolver.from_subclasses(
    base=TrainingCallback,
    skip={MultiTrainingCallback},
)
