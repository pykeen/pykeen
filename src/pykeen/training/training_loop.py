# -*- coding: utf-8 -*-

"""Training loops for KGE models using multi-modal information."""

import gc
import logging
import os
import pathlib
import pickle
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from hashlib import md5
from tempfile import NamedTemporaryFile
from typing import Any, ClassVar, Generic, IO, List, Mapping, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange

from .callbacks import MultiTrainingCallback, TrackerCallback, TrainingCallbackHint
from ..constants import PYKEEN_CHECKPOINTS, PYKEEN_DEFAULT_CHECKPOINT
from ..losses import Loss
from ..models import Model, RGCN
from ..stoppers import Stopper
from ..trackers import ResultTracker
from ..training.schlichtkrull_sampler import GraphSampler
from ..triples import CoreTriplesFactory, Instances
from ..utils import (
    format_relative_comparison, get_batchnorm_modules, is_cuda_oom_error, is_cudnn_error,
    normalize_string,
)

__all__ = [
    'TrainingLoop',
    'NonFiniteLossError',
    'TrainingApproachLossMismatchError',
    'SubBatchingNotSupportedError',
]

logger = logging.getLogger(__name__)

SampleType = TypeVar("SampleType")
BatchType = TypeVar("BatchType")


class NonFiniteLossError(RuntimeError):
    """An exception raised for non-finite loss values."""


class TrainingApproachLossMismatchError(TypeError):
    """An exception when an illegal loss function is used with a given training approach."""


class CheckpointMismatchError(RuntimeError):
    """An exception when a provided checkpoint file does not match the current training loop setup."""


class SubBatchingNotSupportedError(NotImplementedError):
    """An exception raised when sub batching is not implemented."""

    def __init__(self, model: Model):
        super().__init__(model)
        self.model = model

    def __str__(self):  # noqa: D105
        return (
            f'No sub-batching support for {self.model.__class__.__name__} due to modules '
            f'{get_batchnorm_modules(self.model)}.'
        )


def _get_optimizer_kwargs(optimizer: Optimizer) -> Mapping[str, Any]:
    optimizer_kwargs = optimizer.state_dict()
    optimizer_kwargs = {
        key: value
        for key, value in optimizer_kwargs['param_groups'][0].items()
        if key != 'params'
    }
    return optimizer_kwargs


class TrainingLoop(Generic[SampleType, BatchType], ABC):
    """A training loop."""

    model: Model
    optimizer: Optimizer

    losses_per_epochs: List[float]
    loss_blacklist: ClassVar[Optional[List[Type[Loss]]]] = None

    hpo_default = dict(
        num_epochs=dict(type=int, low=100, high=1000, q=100),
        batch_size=dict(type=int, low=32, high=4000, q=100),
    )

    def __init__(
        self,
        model: Model,
        triples_factory: CoreTriplesFactory,
        optimizer: Optional[Optimizer] = None,
        automatic_memory_optimization: bool = True,
    ) -> None:
        """Initialize the training loop.

        :param model: The model to train
        :param triples_factory: The training triples factory
        :param optimizer: The optimizer to use while training the model
        :param automatic_memory_optimization: bool
            Whether to automatically optimize the sub-batch size during
            training and batch size during evaluation with regards to the hardware at hand.
        """
        self.model = model
        self.optimizer = optimizer
        self.losses_per_epochs = []
        self.automatic_memory_optimization = automatic_memory_optimization

        logger.debug("we don't really need the triples factory: %s", triples_factory)

        if self.loss_blacklist and isinstance(self.model.loss, tuple(self.loss_blacklist)):
            raise TrainingApproachLossMismatchError(
                f'Can not use loss {self.model.loss.__class__.__name__}'
                f' with training approach {self.__class__.__name__}',
            )

        # The internal epoch state tracks the last finished epoch of the training loop to allow for
        # seamless loading and saving of training checkpoints
        self._epoch = 0

    @classmethod
    def get_normalized_name(cls) -> str:
        """Get the normalized name of the training loop."""
        return normalize_string(cls.__name__, suffix=TrainingLoop.__name__)

    @property
    def device(self):  # noqa: D401
        """The device used by the model."""
        return self.model.device

    @property
    def loss(self):  # noqa: D401
        """The loss used by the model."""
        return self.model.loss

    @property
    def checksum(self) -> str:  # noqa: D401
        """The checksum of the model and optimizer the training loop was configured with."""
        h = md5()  # noqa: S303
        h.update(str(self.model).encode('utf-8'))
        h.update(str(self.optimizer).encode('utf-8'))
        return h.hexdigest()

    def train(
        self,
        triples_factory: CoreTriplesFactory,
        num_epochs: int = 1,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        label_smoothing: float = 0.0,
        sampler: Optional[str] = None,
        continue_training: bool = False,
        only_size_probing: bool = False,
        use_tqdm: bool = True,
        use_tqdm_batch: bool = True,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
        stopper: Optional[Stopper] = None,
        result_tracker: Optional[ResultTracker] = None,
        sub_batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        clear_optimizer: bool = False,
        checkpoint_directory: Union[None, str, pathlib.Path] = None,
        checkpoint_name: Optional[str] = None,
        checkpoint_frequency: Optional[int] = None,
        checkpoint_on_failure: bool = False,
        drop_last: Optional[bool] = None,
        callbacks: TrainingCallbackHint = None,
    ) -> Optional[List[float]]:
        """Train the KGE model.

        :param triples_factory:
            The training triples.
        :param num_epochs:
            The number of epochs to train the model.
        :param batch_size:
            If set the batch size to use for mini-batch training. Otherwise find the largest possible batch_size
            automatically.
        :param slice_size: >0
            The divisor for the scoring function when using slicing. This is only possible for LCWA training loops in
            general and only for models that have the slicing capability implemented.
        :param label_smoothing: (0 <= label_smoothing < 1)
            If larger than zero, use label smoothing.
        :param sampler: (None or 'schlichtkrull')
            The type of sampler to use. At the moment sLCWA in R-GCN is the only user of schlichtkrull sampling.
        :param continue_training:
            If set to False, (re-)initialize the model's weights. Otherwise continue training.
        :param only_size_probing:
            The evaluation is only performed for two batches to test the memory footprint, especially on GPUs.
        :param use_tqdm: Should a progress bar be shown for epochs?
        :param use_tqdm_batch: Should a progress bar be shown for batching (inside the epoch progress bar)?
        :param tqdm_kwargs:
            Keyword arguments passed to :mod:`tqdm` managing the progress bar.
        :param stopper:
            An instance of :class:`pykeen.stopper.EarlyStopper` with settings for checking
            if training should stop early
        :param result_tracker:
            The result tracker.
        :param sub_batch_size:
            If provided split each batch into sub-batches to avoid memory issues for large models / small GPUs.
        :param num_workers:
            The number of child CPU workers used for loading data. If None, data are loaded in the main process.
        :param clear_optimizer:
            Whether to delete the optimizer instance after training (as the optimizer might have additional memory
            consumption due to e.g. moments in Adam).
        :param checkpoint_directory:
            An optional directory to store the checkpoint files. If None, a subdirectory named ``checkpoints`` in the
            directory defined by :data:`pykeen.constants.PYKEEN_HOME` is used. Unless the environment variable
            ``PYKEEN_HOME`` is overridden, this will be ``~/.pykeen/checkpoints``.
        :param checkpoint_name:
            The filename for saving checkpoints. If the given filename exists already, that file will be loaded and used
            to continue training.
        :param checkpoint_frequency:
            The frequency of saving checkpoints in minutes. Setting it to 0 will save a checkpoint after every epoch.
        :param checkpoint_on_failure:
            Whether to save a checkpoint in cases of a RuntimeError or MemoryError. This option differs from ordinary
            checkpoints, since ordinary checkpoints are only saved after a successful epoch. When saving checkpoints
            due to failure of the training loop there is no guarantee that all random states can be recovered correctly,
            which might cause problems with regards to the reproducibility of that specific training loop. Therefore,
            these checkpoints are saved with a distinct checkpoint name, which will be
            ``PyKEEN_just_saved_my_day_{datetime}.pt`` in the given checkpoint_root.
        :param drop_last:
            Whether to drop the last batch in each epoch to prevent smaller batches. Defaults to False, except if the
            model contains batch normalization layers. Can be provided explicitly to override.
        :param callbacks:
            An optional :class:`pykeen.training.TrainingCallback` or collection of callback instances that define
            one of several functionalities. Their interface was inspired by Keras.

        :return:
            The losses per epoch.
        """
        # Create training instances. Use the _create_instances function to allow subclasses
        # to modify this behavior
        training_instances = self._create_instances(triples_factory)

        # In some cases, e.g. using Optuna for HPO, the cuda cache from a previous run is not cleared
        torch.cuda.empty_cache()

        # A checkpoint root is always created to ensure a fallback checkpoint can be saved
        if checkpoint_directory is None:
            checkpoint_directory = PYKEEN_CHECKPOINTS
        checkpoint_directory = pathlib.Path(checkpoint_directory)
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        logger.debug('using checkpoint_root at %s', checkpoint_directory)

        # If a checkpoint file is given, it must be loaded if it exists already
        save_checkpoints = False
        checkpoint_path = None
        best_epoch_model_file_path = None
        last_best_epoch = None
        if checkpoint_name:
            checkpoint_path = checkpoint_directory.joinpath(checkpoint_name)
            if checkpoint_path.is_file():
                best_epoch_model_file_path, last_best_epoch = self._load_state(path=checkpoint_path)
                if stopper is not None:
                    stopper_dict = stopper.load_summary_dict_from_training_loop_checkpoint(path=checkpoint_path)
                    # If the stopper dict has any keys, those are written back to the stopper
                    if stopper_dict:
                        stopper._write_from_summary_dict(**stopper_dict)
                    else:
                        logger.warning(
                            'the training loop was configured with a stopper but no stopper configuration was '
                            'saved in the checkpoint',
                        )
                continue_training = True
            else:
                logger.info(f"=> no checkpoint found at '{checkpoint_path}'. Creating a new file.")
            # The checkpoint frequency needs to be set to save checkpoints
            if checkpoint_frequency is None:
                checkpoint_frequency = 30
            save_checkpoints = True
        elif checkpoint_frequency is not None:
            logger.warning(
                "A checkpoint frequency was set, but no checkpoint file was given. No checkpoints will be created",
            )

        checkpoint_on_failure_file_path = None
        if checkpoint_on_failure:
            # In case a checkpoint frequency was set, we warn that no checkpoints will be saved
            date_string = datetime.now().strftime('%Y%m%d_%H_%M_%S')
            # If no checkpoints were requested, a fallback checkpoint is set in case the training loop crashes
            checkpoint_on_failure_file_path = checkpoint_directory.joinpath(
                PYKEEN_DEFAULT_CHECKPOINT.replace('.', f"_{date_string}."),
            )

        # If the stopper loaded from the training loop checkpoint stopped the training, we return those results
        if getattr(stopper, 'stopped', False):
            result: Optional[List[float]] = self.losses_per_epochs
        else:
            result = self._train(
                num_epochs=num_epochs,
                batch_size=batch_size,
                slice_size=slice_size,
                label_smoothing=label_smoothing,
                sampler=sampler,
                continue_training=continue_training,
                only_size_probing=only_size_probing,
                use_tqdm=use_tqdm,
                use_tqdm_batch=use_tqdm_batch,
                tqdm_kwargs=tqdm_kwargs,
                stopper=stopper,
                result_tracker=result_tracker,
                sub_batch_size=sub_batch_size,
                num_workers=num_workers,
                save_checkpoints=save_checkpoints,
                checkpoint_path=checkpoint_path,
                checkpoint_frequency=checkpoint_frequency,
                checkpoint_on_failure_file_path=checkpoint_on_failure_file_path,
                best_epoch_model_file_path=best_epoch_model_file_path,
                last_best_epoch=last_best_epoch,
                drop_last=drop_last,
                callbacks=callbacks,
                triples_factory=triples_factory,
                training_instances=training_instances,
            )

        # Ensure the release of memory
        torch.cuda.empty_cache()

        # Clear optimizer
        if clear_optimizer:
            self.optimizer = None

        return result

    def _train(  # noqa: C901
        self,
        triples_factory: CoreTriplesFactory,
        training_instances: Instances,
        num_epochs: int = 1,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        label_smoothing: float = 0.0,
        sampler: Optional[str] = None,
        continue_training: bool = False,
        only_size_probing: bool = False,
        use_tqdm: bool = True,
        use_tqdm_batch: bool = True,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
        stopper: Optional[Stopper] = None,
        result_tracker: Optional[ResultTracker] = None,
        sub_batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        save_checkpoints: bool = False,
        checkpoint_path: Union[None, str, pathlib.Path] = None,
        checkpoint_frequency: Optional[int] = None,
        checkpoint_on_failure_file_path: Union[None, str, pathlib.Path] = None,
        best_epoch_model_file_path: Optional[pathlib.Path] = None,
        last_best_epoch: Optional[int] = None,
        drop_last: Optional[bool] = None,
        callbacks: TrainingCallbackHint = None,
    ) -> Optional[List[float]]:
        """Train the KGE model.

        :param triples_factory:
            The training triples factory
        :param num_epochs:
            The number of epochs to train the model.
        :param batch_size:
            If set the batch size to use for mini-batch training. Otherwise find the largest possible batch_size
            automatically.
        :param slice_size: >0
            The divisor for the scoring function when using slicing. This is only possible for LCWA training loops in
            general and only for models that have the slicing capability implemented.
        :param label_smoothing: (0 <= label_smoothing < 1)
            If larger than zero, use label smoothing.
        :param sampler: (None or 'schlichtkrull')
            The type of sampler to use. At the moment sLCWA in R-GCN is the only user of schlichtkrull sampling.
        :param continue_training:
            If set to False, (re-)initialize the model's weights. Otherwise continue training.
        :param only_size_probing:
            The evaluation is only performed for two batches to test the memory footprint, especially on GPUs.
        :param use_tqdm:
            Turn on the progress bar for epochs
        :param use_tqdm_batch:
            Turn on the progress bar for batches (sub-progress bar for epochs)
        :param tqdm_kwargs:
            Keyword arguments passed to :mod:`tqdm` managing the progress bar.
        :param stopper:
            An instance of :class:`pykeen.stopper.Stopper` with settings for checking
            if training should stop early
        :param result_tracker:
            The result tracker.
        :param sub_batch_size:
            If provided split each batch into sub-batches to avoid memory issues for large models / small GPUs.
        :param num_workers:
            The number of child CPU workers used for loading data. If None, data are loaded in the main process.
        :param save_checkpoints:
            Activate saving checkpoints.
        :param checkpoint_path:
            The full filepath for saving checkpoints.
        :param checkpoint_frequency:
            The frequency of saving checkpoints in minutes. Setting it to 0 will save a checkpoint after every epoch.
        :param checkpoint_on_failure_file_path:
            The full filepath for saving checkpoints on failure.
        :param best_epoch_model_file_path:
            The file path for the best epoch model when using early stoppers and resuming training.
        :param last_best_epoch:
            The last best epoch that the early stopper saved when resuming training.
        :param drop_last:
            Whether to drop the last batch in each epoch to prevent smaller batches. Defaults to False, except if the
            model contains batch normalization layers. Can be provided explicitly to override.

        :return:
            The losses per epoch.
        """
        if self.optimizer is None:
            raise ValueError('optimizer must be set before running _train()')
        # When using early stopping models have to be saved separately at the best epoch, since the training loop will
        # due to the patience continue to train after the best epoch and thus alter the model
        if (
            stopper is not None
            and not only_size_probing
            and last_best_epoch is None
            and best_epoch_model_file_path is None
        ):
            # Create a path
            best_epoch_model_file_path = pathlib.Path(NamedTemporaryFile().name)
        best_epoch_model_checkpoint_file_path: Optional[pathlib.Path] = None

        if isinstance(self.model, RGCN) and sampler != 'schlichtkrull':
            logger.warning(
                'Using RGCN without graph-based sampling! Please select sampler="schlichtkrull" instead of %s.',
                sampler,
            )

        # Prepare all of the callbacks
        callback = MultiTrainingCallback(callbacks)
        # Register a callback for the result tracker, if given
        if result_tracker is not None:
            callback.register_callback(TrackerCallback(result_tracker))

        callback.register_training_loop(self)

        # Take the biggest possible training batch_size, if batch_size not set
        batch_size_sufficient = False
        if batch_size is None:
            if self.automatic_memory_optimization:
                # Using automatic memory optimization on CPU may result in undocumented crashes due to OS' OOM killer.
                if self.model.device.type == 'cpu':
                    batch_size = 256
                    batch_size_sufficient = True
                    logger.info(
                        "Currently automatic memory optimization only supports GPUs, but you're using a CPU. "
                        "Therefore, the batch_size will be set to the default value '{batch_size}'",
                    )
                else:
                    batch_size, batch_size_sufficient = self.batch_size_search(
                        triples_factory=triples_factory,
                        training_instances=training_instances,
                    )
            else:
                batch_size = 256
                logger.info(f"No batch_size provided. Setting batch_size to '{batch_size}'.")

        # This will find necessary parameters to optimize the use of the hardware at hand
        if (
            not only_size_probing
            and self.automatic_memory_optimization
            and not batch_size_sufficient
            and not continue_training
        ):
            # return the relevant parameters slice_size and batch_size
            sub_batch_size, slice_size = self.sub_batch_and_slice(
                batch_size=batch_size,
                sampler=sampler,
                triples_factory=triples_factory,
                training_instances=training_instances,
            )

        if sub_batch_size is None or sub_batch_size == batch_size:  # by default do not split batches in sub-batches
            sub_batch_size = batch_size
        elif get_batchnorm_modules(self.model):  # if there are any, this is truthy
            raise SubBatchingNotSupportedError(self.model)

        model_contains_batch_norm = bool(get_batchnorm_modules(self.model))
        if batch_size == 1 and model_contains_batch_norm:
            raise ValueError("Cannot train a model with batch_size=1 containing BatchNorm layers.")
        if drop_last is None:
            drop_last = model_contains_batch_norm
            if drop_last and not only_size_probing:
                logger.info(
                    "Dropping last (incomplete) batch each epoch (%s batches).",
                    format_relative_comparison(part=1, total=len(training_instances)),
                )

        # Force weight initialization if training continuation is not explicitly requested.
        if not continue_training:
            # Reset the weights
            self.model.reset_parameters_()

            # Create new optimizer
            optimizer_kwargs = _get_optimizer_kwargs(self.optimizer)
            self.optimizer = self.optimizer.__class__(
                params=self.model.get_grad_params(),
                **optimizer_kwargs,
            )
        elif not self.optimizer.state:
            raise ValueError('Cannot continue_training without being trained once.')

        # Ensure the model is on the correct device
        self.model = self.model.to(self.device)

        # Create Sampler
        if sampler == 'schlichtkrull':
            if triples_factory is None:
                raise ValueError('need to pass triples_factory when using graph sampling')
            sampler = GraphSampler(triples_factory, num_samples=sub_batch_size)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        if num_workers is None:
            num_workers = 0

        # Bind
        num_training_instances = len(training_instances)

        _use_outer_tqdm = not only_size_probing and use_tqdm
        _use_inner_tqdm = _use_outer_tqdm and use_tqdm_batch

        # When size probing, we don't want progress bars
        if _use_outer_tqdm:
            # Create progress bar
            _tqdm_kwargs = dict(desc=f'Training epochs on {self.device}', unit='epoch')
            if tqdm_kwargs is not None:
                _tqdm_kwargs.update(tqdm_kwargs)
            epochs = trange(self._epoch + 1, 1 + num_epochs, **_tqdm_kwargs, initial=self._epoch, total=num_epochs)
        elif only_size_probing:
            epochs = range(1, 1 + num_epochs)
        else:
            epochs = range(self._epoch + 1, 1 + num_epochs)

        logger.debug(f'using stopper: {stopper}')

        train_data_loader = DataLoader(
            sampler=sampler,
            dataset=training_instances,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )

        # Save the time to track when the saved point was available
        last_checkpoint = time.time()

        # Training Loop
        for epoch in epochs:
            # When training with an early stopper the memory pressure changes, which may allow for errors each epoch
            try:
                # Enforce training mode
                self.model.train()

                # Accumulate loss over epoch
                current_epoch_loss = 0.

                # Batching
                # Only create a progress bar when not in size probing mode
                if _use_inner_tqdm:
                    batches = tqdm(
                        train_data_loader,
                        desc=f'Training batches on {self.device}',
                        leave=False,
                        unit='batch',
                    )
                else:
                    batches = train_data_loader

                # Flag to check when to quit the size probing
                evaluated_once = False

                for batch in batches:
                    # Recall that torch *accumulates* gradients. Before passing in a
                    # new instance, you need to zero out the gradients from the old instance
                    self.optimizer.zero_grad()

                    # Get batch size of current batch (last batch may be incomplete)
                    current_batch_size = self._get_batch_size(batch)

                    # accumulate gradients for whole batch
                    for start in range(0, current_batch_size, sub_batch_size):
                        stop = min(start + sub_batch_size, current_batch_size)

                        # forward pass call
                        batch_loss = self._forward_pass(
                            batch,
                            start,
                            stop,
                            current_batch_size,
                            label_smoothing,
                            slice_size,
                        )
                        current_epoch_loss += batch_loss
                        callback.on_batch(epoch=epoch, batch=batch, batch_loss=batch_loss)

                    # when called by batch_size_search(), the parameter update should not be applied.
                    if not only_size_probing:
                        # update parameters according to optimizer
                        self.optimizer.step()

                    # After changing applying the gradients to the embeddings, the model is notified that the forward
                    # constraints are no longer applied
                    self.model.post_parameter_update()

                    # For testing purposes we're only interested in processing one batch
                    if only_size_probing and evaluated_once:
                        break

                    callback.post_batch(epoch=epoch, batch=batch)

                    evaluated_once = True

                del batch
                del batches
                gc.collect()
                self.optimizer.zero_grad()
                self._free_graph_and_cache()

                # When size probing we don't need the losses
                if only_size_probing:
                    return None

                # Track epoch loss
                epoch_loss = current_epoch_loss / num_training_instances
                self.losses_per_epochs.append(epoch_loss)

                # Print loss information to console
                if _use_outer_tqdm:
                    epochs.set_postfix({
                        'loss': self.losses_per_epochs[-1],
                        'prev_loss': self.losses_per_epochs[-2] if epoch > 2 else float('nan'),
                    })

                # Save the last successful finished epoch
                self._epoch = epoch

                should_stop = False
                if stopper is not None and stopper.should_evaluate(epoch):
                    if stopper.should_stop(epoch):
                        should_stop = True
                    # Since the model is also used within the stopper, its graph and cache have to be cleared
                    self._free_graph_and_cache()
                # When the stopper obtained a new best epoch, this model has to be saved for reconstruction
                if (
                    stopper is not None
                    and stopper.best_epoch != last_best_epoch
                    and best_epoch_model_file_path is not None
                ):
                    self._save_state(path=best_epoch_model_file_path)
                    last_best_epoch = epoch
            # When the training loop failed, a fallback checkpoint is created to resume training.
            except (MemoryError, RuntimeError) as e:
                # During automatic memory optimization only the error message is of interest
                if only_size_probing:
                    raise e

                logger.warning(f'The training loop just failed during epoch {epoch} due to error {str(e)}.')
                if checkpoint_on_failure_file_path:
                    # When there wasn't a best epoch the checkpoint path should be None
                    if last_best_epoch is not None and best_epoch_model_file_path is not None:
                        best_epoch_model_checkpoint_file_path = best_epoch_model_file_path
                    self._save_state(
                        path=checkpoint_on_failure_file_path,
                        stopper=stopper,
                        best_epoch_model_checkpoint_file_path=best_epoch_model_checkpoint_file_path,
                    )
                    logger.warning(
                        "However, don't worry we got you covered. PyKEEN just saved a checkpoint when this "
                        f"happened at '{checkpoint_on_failure_file_path}'. To resume training from the checkpoint "
                        f"file just restart your code and pass this file path to the training loop or pipeline you "
                        f"used as 'checkpoint_file' argument.",
                    )
                # Delete temporary best epoch model
                if best_epoch_model_file_path is not None and best_epoch_model_file_path.is_file():
                    os.remove(best_epoch_model_file_path)
                raise e

            # Includes a call to result_tracker.log_metrics
            callback.post_epoch(epoch=epoch, epoch_loss=epoch_loss)

            # If a checkpoint file is given, we check whether it is time to save a checkpoint
            if save_checkpoints and checkpoint_path is not None:
                minutes_since_last_checkpoint = (time.time() - last_checkpoint) // 60
                # MyPy overrides are because you should
                if (
                    minutes_since_last_checkpoint >= checkpoint_frequency  # type: ignore
                    or should_stop
                    or epoch == num_epochs
                ):
                    # When there wasn't a best epoch the checkpoint path should be None
                    if last_best_epoch is not None and best_epoch_model_file_path is not None:
                        best_epoch_model_checkpoint_file_path = best_epoch_model_file_path
                    self._save_state(
                        path=checkpoint_path,
                        stopper=stopper,
                        best_epoch_model_checkpoint_file_path=best_epoch_model_checkpoint_file_path,
                    )  # type: ignore
                    last_checkpoint = time.time()

            if should_stop and last_best_epoch is not None and best_epoch_model_file_path is not None:
                self._load_state(path=best_epoch_model_file_path)
                # Delete temporary best epoch model
                if pathlib.Path.is_file(best_epoch_model_file_path):
                    os.remove(best_epoch_model_file_path)
                return self.losses_per_epochs

        callback.post_train(losses=self.losses_per_epochs)

        # If the stopper didn't stop the training loop but derived a best epoch, the model has to be reconstructed
        # at that state
        if stopper is not None and last_best_epoch is not None and best_epoch_model_file_path is not None:
            self._load_state(path=best_epoch_model_file_path)
            # Delete temporary best epoch model
            if pathlib.Path.is_file(best_epoch_model_file_path):
                os.remove(best_epoch_model_file_path)

        return self.losses_per_epochs

    def _forward_pass(
        self,
        batch: BatchType,
        start: int,
        stop: int,
        current_batch_size: int,
        label_smoothing: float,
        slice_size: Optional[int],
    ) -> float:
        # forward pass
        loss = self._process_batch(
            batch=batch,
            start=start,
            stop=stop,
            label_smoothing=label_smoothing,
            slice_size=slice_size,
        )

        # raise error when non-finite loss occurs (NaN, +/-inf)
        if not torch.isfinite(loss):
            raise NonFiniteLossError('Loss is non-finite.')

        # correction for loss reduction
        if self.model.loss.reduction == 'mean':
            this_sub_batch_size = stop - start
            loss *= (this_sub_batch_size / current_batch_size)

        # backward pass
        loss.backward()
        current_epoch_loss = loss.item()

        self.model.post_forward_pass()
        # TODO why not call torch.cuda.empty_cache()? or call self._free_graph_and_cache()?

        return current_epoch_loss

    @staticmethod
    @abstractmethod
    def _get_batch_size(batch: BatchType) -> int:
        """Get the batch size from a (sub-) batch."""
        raise NotImplementedError

    @abstractmethod
    def _create_instances(self, triples_factory: CoreTriplesFactory) -> Instances:
        """Create the training instances at the beginning of the training loop."""
        raise NotImplementedError

    @abstractmethod
    def _process_batch(
        self,
        batch: BatchType,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """Process a single batch and returns the loss."""
        raise NotImplementedError

    def batch_size_search(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        training_instances: Instances,
        batch_size: Optional[int] = None,
    ) -> Tuple[int, bool]:
        """Find the maximum batch size for training with the current setting.

        This method checks how big the batch size can be for the current model with the given training data and the
        hardware at hand. If possible, the method will output the determined batch size and a boolean value indicating
        that this batch size was successfully evaluated. Otherwise, the output will be batch size 1 and the boolean
        value will be False.

        :param triples_factory:
            The triples factory over which search is run
        :param training_instances:
            The training instances generated from the triples factory
        :param batch_size:
            The batch size to start the search with. If None, set batch_size=num_triples (i.e. full batch training).

        :return:
            Tuple containing the maximum possible batch size as well as an indicator if the evaluation with that size
            was successful.
        """
        if batch_size is None:
            batch_size = 8192

        # Set upper bound
        batch_size = min(batch_size, triples_factory.num_triples)

        reached_max = False
        evaluated_once = False
        logger.info('Starting batch_size search for training now...')
        while True:
            logger.debug(f'Trying batch_size={batch_size}.')
            try:
                self._free_graph_and_cache()
                self._train(
                    num_epochs=1,
                    batch_size=batch_size,
                    sub_batch_size=None,
                    only_size_probing=True,
                    triples_factory=triples_factory,
                    training_instances=training_instances,
                )
            except RuntimeError as runtime_error:
                self._free_graph_and_cache()
                if not is_cudnn_error(runtime_error) and not is_cuda_oom_error(runtime_error):
                    raise runtime_error
                if batch_size == 1:
                    logger.debug(f"batch_size={batch_size} does not fit into your memory with these parameters.")
                    break

                reached_max = True
                batch_size //= 2

                if evaluated_once:
                    logger.info(f'Concluded batch_size search with batch_size={batch_size}.')
                    break

                logger.debug(f'batch_size={batch_size} was too big, trying less now.')
            else:
                self._free_graph_and_cache()
                if not reached_max and batch_size <= triples_factory.num_triples:
                    batch_size *= 2
                else:
                    logger.info(f'Concluded batch_size search with batch_size={batch_size}.')
                    evaluated_once = True
                    break

        return batch_size, evaluated_once

    def sub_batch_and_slice(
        self,
        *,
        batch_size: int,
        sampler: Optional[str],
        triples_factory: CoreTriplesFactory,
        training_instances: Instances,
    ) -> Tuple[int, Optional[int]]:
        """Check if sub-batching and/or slicing is necessary to train the model on the hardware at hand."""
        sub_batch_size, finished_search, supports_sub_batching = self._sub_batch_size_search(
            batch_size=batch_size,
            sampler=sampler,
            triples_factory=triples_factory,
            training_instances=training_instances,
        )
        # If the sub_batch_size did not finish search with a possibility that fits the hardware, we have to try slicing
        if finished_search:
            return sub_batch_size, None

        slice_size = self._slice_size_search(
            triples_factory=triples_factory,
            training_instances=training_instances,
            batch_size=batch_size,
            sub_batch_size=sub_batch_size,
            supports_sub_batching=supports_sub_batching,
        )
        return sub_batch_size, slice_size

    @abstractmethod
    def _slice_size_search(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        training_instances: Instances,
        batch_size: int,
        sub_batch_size: int,
        supports_sub_batching: bool,
    ) -> int:
        """Find the maximum slice size for training with the current setting.

        This method finds the biggest slice size to train the model with the given training data and the desired batch
        and sub_batch size on the hardware at hand. If even the slice size 1 is too high, it will raise an error.
        Otherwise it will return the determined slice size.

        :param batch_size:
            The batch size to use.
        :param sub_batch_size:
            The sub-batch size to use.
        :param supports_sub_batching:
            Indicator if the model supports sub-batching. This is used to create appropriate error messages, if needed.

        :return:
            The slice_size that allows training the model with the given parameters on this hardware.

        :raises MemoryError:
            If it is not possible to train the model on the hardware at hand with the given parameters.
        """
        raise NotImplementedError

    def _sub_batch_size_search(
        self,
        *,
        batch_size: int,
        sampler: Optional[str],
        triples_factory: CoreTriplesFactory,
        training_instances: Instances,
    ) -> Tuple[int, bool, bool]:
        """Find the allowable sub batch size for training with the current setting.

        This method checks if it is possible to train the model with the given training data and the desired batch size
        on the hardware at hand. If possible, the sub-batch size equals the batch size. Otherwise, the maximum
        permissible sub-batch size is determined.

        :param batch_size:
            The initial batch size to start with.

        :return:
            Tuple containing the sub-batch size to use and indicating if the search was finished, i.e. successfully
            without hardware errors, as well as if sub-batching is possible
        """
        sub_batch_size = batch_size
        finished_search = False
        supports_sub_batching = True

        try:
            # The cache of the previous run has to be freed to allow accurate memory availability estimates
            self._free_graph_and_cache()
            logger.debug(f'Trying batch_size {batch_size} for training now.')
            self._train(
                triples_factory=triples_factory,
                training_instances=training_instances,
                num_epochs=1,
                batch_size=batch_size,
                sub_batch_size=sub_batch_size,
                sampler=sampler,
                only_size_probing=True,
            )
        except RuntimeError as runtime_error:
            self._free_graph_and_cache()
            if not is_cudnn_error(runtime_error) and not is_cuda_oom_error(runtime_error):
                raise runtime_error
            logger.debug(f'The batch_size {batch_size} was too big, sub_batching is required.')
            sub_batch_size //= 2
        else:
            finished_search = True
            logger.debug('No sub-batching required.')

        if not finished_search:
            logger.info('Starting sub_batch_size search for training now...')
            if get_batchnorm_modules(self.model):  # if there are any, this is truthy
                logger.info('This model does not support sub-batching.')
                supports_sub_batching = False
                sub_batch_size = batch_size
            else:
                while True:
                    logger.debug(f'Trying sub_batch_size {sub_batch_size} now.')
                    try:
                        self._free_graph_and_cache()
                        self._train(
                            num_epochs=1,
                            batch_size=batch_size,
                            sub_batch_size=sub_batch_size,
                            sampler=sampler,
                            only_size_probing=True,
                            triples_factory=triples_factory,
                            training_instances=training_instances,
                        )
                    except RuntimeError as runtime_error:
                        self._free_graph_and_cache()
                        if not is_cudnn_error(runtime_error) and not is_cuda_oom_error(runtime_error):
                            raise runtime_error
                        if sub_batch_size == 1:
                            logger.info(
                                f"Even sub_batch_size={sub_batch_size} does not fit in memory with these parameters",
                            )
                            break
                        logger.debug(f'The sub_batch_size {sub_batch_size} was too big, trying less now.')
                        sub_batch_size //= 2
                    else:
                        finished_search = True
                        logger.info(f'Concluded search with sub_batch_size {sub_batch_size}.')
                        break

        self._free_graph_and_cache()

        return sub_batch_size, finished_search, supports_sub_batching

    def _free_graph_and_cache(self):
        self.model._free_graph_and_cache()
        # The cache of the previous run has to be freed to allow accurate memory availability estimates
        torch.cuda.empty_cache()

    def _save_state(
        self,
        path: Union[IO[bytes], str, pathlib.Path],
        stopper: Optional[Stopper] = None,
        best_epoch_model_checkpoint_file_path: Optional[pathlib.Path] = None,
    ) -> None:
        """Save the state of the training loop.

        :param path:
            Path of the file where to store the state in.
        :param stopper:
            An instance of :class:`pykeen.stopper.EarlyStopper` with settings for checking
            if training should stop early
        :param best_epoch_model_checkpoint_file_path:
            The file path for the checkpoint of the best epoch model when using early stopping.
        """
        if self.optimizer is None:
            raise ValueError

        logger.debug("=> Saving checkpoint.")

        if stopper is None:
            stopper_dict: Mapping[str, Any] = dict()
        else:
            stopper_dict = stopper.get_summary_dict()

        # Only if a cuda device is available, the random state is accessed
        if torch.cuda.is_available():
            torch_cuda_random_state = torch.cuda.get_rng_state()
        else:
            torch_cuda_random_state = None

        if best_epoch_model_checkpoint_file_path is not None:
            best_epoch_model_checkpoint = torch.load(best_epoch_model_checkpoint_file_path)
        else:
            best_epoch_model_checkpoint = None

        torch.save(
            {
                'epoch': self._epoch,
                'loss': self.losses_per_epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'checksum': self.checksum,
                'random_seed': self.model._random_seed,
                'stopper_dict': stopper_dict,
                'random_state': random.getstate(),
                'np_random_state': np.random.get_state(),
                'torch_random_state': torch.random.get_rng_state(),
                'torch_cuda_random_state': torch_cuda_random_state,
                # This is an entire checkpoint for the optional best model when using early stopping
                'best_epoch_model_checkpoint': best_epoch_model_checkpoint,
            },
            path,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
        )
        logger.info(f"=> Saved checkpoint after having finished epoch {self._epoch}.")

    def _load_state(
        self,
        path: Union[str, pathlib.Path],
    ) -> Tuple[Optional[pathlib.Path], Optional[int]]:
        """Load the state of the training loop from a checkpoint.

        :param path:
            Path of the file where to load the state from.

        :return:
            Temporary file path of the best epoch model and the best epoch when using early stoppers, None otherwise.

        :raises CheckpointMismatchError:
            If the given checkpoint file has a non-matching checksum, i.e. it was saved with a different configuration.
        """
        if self.optimizer is None:
            raise ValueError

        logger.info(f"=> loading checkpoint '{path}'")
        checkpoint = torch.load(path)
        if checkpoint['checksum'] != self.checksum:
            raise CheckpointMismatchError(
                f"The checkpoint file '{path}' that was provided already exists, but seems to be "
                f"from a different training loop setup.",
            )
        # Cuda requires its own random state, which can only be set when a cuda device is available
        torch_cuda_random_state = checkpoint['torch_cuda_random_state']
        if torch_cuda_random_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_random_state)
        elif torch_cuda_random_state is not None and not torch.cuda.is_available():
            logger.warning(
                "You're currently trying to resume the training loop on a CPU from a checkpoint that was saved "
                "with a GPU. Therefore, the random state for the CUDA devices can't be set and results may not "
                "be deterministic.",
            )
        elif torch_cuda_random_state is None and torch.cuda.is_available():
            logger.warning(
                "You're currently trying to resume the training loop on a GPU from a checkpoint that was saved "
                "without a GPU. Therefore, the random state for the CUDA devices won't be set and results may not "
                "be deterministic.",
            )

        # If the checkpoint was saved with a best epoch model from the early stopper, this model has to be retrieved
        best_epoch_model_file_path = None
        best_epoch = None
        if checkpoint.get('best_epoch_model_checkpoint'):
            best_epoch_model_file_path = pathlib.Path(NamedTemporaryFile().name)
            best_epoch = checkpoint['best_epoch_model_checkpoint']['epoch']
            torch.save(
                checkpoint['best_epoch_model_checkpoint'],
                best_epoch_model_file_path,
                pickle_protocol=pickle.HIGHEST_PROTOCOL,
            )

        self._epoch = checkpoint['epoch']
        self.losses_per_epochs = checkpoint['loss']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['np_random_state'])
        torch.random.set_rng_state(checkpoint['torch_random_state'])
        logger.info(f"=> loaded checkpoint '{path}' stopped after having finished epoch {checkpoint['epoch']}")

        return best_epoch_model_file_path, best_epoch
