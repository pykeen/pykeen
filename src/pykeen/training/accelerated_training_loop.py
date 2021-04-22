# -*- coding: utf-8 -*-

"""Accelerated Training loop for KGE models using HF Accelerate"""

import gc
import logging
import pathlib
import time
from typing import Any, List, Mapping, Optional, Type, Union

import torch
from accelerate import Accelerator
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange

from .training_loop import NonFiniteLossError, SubBatchingNotSupportedError, TrainingLoop, _get_optimizer_kwargs
from ..losses import Loss, has_mr_loss
from ..models import Model, RGCN
from ..stoppers import Stopper
from ..trackers import ResultTracker
from ..training.schlichtkrull_sampler import GraphSampler
from ..triples import Instances
from ..utils import format_relative_comparison, get_batchnorm_modules

__all__ = [
    'AcceleratedTrainingLoop',
]

logger = logging.getLogger(__name__)


class AcceleratedTrainingLoop(TrainingLoop):
    """A training loop with HF accelerate."""

    training_instances: Optional[Instances]
    losses_per_epochs: List[float]
    loss_blacklist: Optional[List[Type[Loss]]] = None

    hpo_default = dict(
        num_epochs=dict(type=int, low=100, high=1000, q=100),
        batch_size=dict(type=int, low=32, high=4000, q=100),
    )

    def __init__(
        self,
        model: Model,
        optimizer: Optional[Optimizer] = None,
        automatic_memory_optimization: bool = True
    ) -> None:
        """Initialize the training loop.

        :param model: The model to train
        :param optimizer: The optimizer to use while training the model
        :param automatic_memory_optimization: bool
            Whether to automatically optimize the sub-batch size during
            training and batch size during evaluation with regards to the hardware at hand.
        """
        super().__init__(model=model, optimizer=optimizer, automatic_memory_optimization=automatic_memory_optimization)
        self.accelerator = Accelerator()

    @property
    def device(self):  # noqa: D401
        """The device used by the model."""
        return self.accelerator.device

    def _train(  # noqa: C901
        self,
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
        drop_last: Optional[bool] = None,
    ) -> Optional[List[float]]:
        """Train the KGE model.

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
        :param drop_last:
            Whether to drop the last batch in each epoch to prevent smaller batches. Defaults to False, except if the
            model contains batch normalization layers. Can be provided explicitly to override.

        :return:
            The losses per epoch.
        """
        if self.training_instances is None:
            raise ValueError('must set training instances before running _train()')
        if self.optimizer is None:
            raise ValueError('optimizer must be set before running _train()')

        if isinstance(self.model, RGCN) and sampler != 'schlichtkrull':
            logger.warning(
                'Using RGCN without graph-based sampling! Please select sampler="schlichtkrull" instead of %s.',
                sampler,
            )

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
                    batch_size, batch_size_sufficient = self.batch_size_search()
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
            sub_batch_size, slice_size = self.sub_batch_and_slice(batch_size=batch_size, sampler=sampler)

        # Create dummy result tracker
        if result_tracker is None:
            result_tracker = ResultTracker()

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
                    format_relative_comparison(part=1, total=len(self.training_instances)),
                )

        # Sanity check
        if has_mr_loss(self.model) and label_smoothing > 0.:
            raise RuntimeError('Label smoothing can not be used with margin ranking loss.')

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
            sampler = GraphSampler(self.triples_factory, num_samples=sub_batch_size)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        if num_workers is None:
            num_workers = 0

        # Bind
        num_training_instances = len(self.training_instances)

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
            dataset=self.training_instances,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )

        # Save the time to track when the saved point was available
        last_checkpoint = time.time()

        # Accelerate-specific initialization of the model, optimizer, and data loader
        self.model, self.optimizer, train_data_loader = self.accelerator.prepare(
            self.model,
            self.optimizer,
            train_data_loader
        )

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
                        disable=not self.accelerator.is_local_main_process
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
                        current_epoch_loss += self._forward_pass(
                            batch,
                            start,
                            stop,
                            current_batch_size,
                            label_smoothing,
                            slice_size,
                        )

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
                result_tracker.log_metrics({'loss': epoch_loss}, step=epoch)

                # Print loss information to console
                if _use_outer_tqdm:
                    epochs.set_postfix({
                        'loss': self.losses_per_epochs[-1],
                        'prev_loss': self.losses_per_epochs[-2] if epoch > 2 else float('nan'),
                    })

                # Save the last successful finished epoch
                self._epoch = epoch

                should_stop = False
                if stopper is not None and stopper.should_evaluate(epoch) and stopper.should_stop(epoch):
                    should_stop = True
            # When the training loop failed, a fallback checkpoint is created to resume training.
            except (MemoryError, RuntimeError) as e:
                logger.warning(f'The training loop just failed during epoch {epoch} due to error {str(e)}.')
                if checkpoint_on_failure_file_path:
                    self._save_state(path=checkpoint_on_failure_file_path, stopper=stopper)
                    logger.warning(
                        "However, don't worry we got you covered. PyKEEN just saved a checkpoint when this happened "
                        f"at '{checkpoint_on_failure_file_path}'. To resume training from the checkpoint file just "
                        f"restart your code and pass this file path to the training loop or pipeline you used "
                        f"as 'checkpoint_file' argument.",
                    )
                raise e

            # If a checkpoint file is given, we check whether it is time to save a checkpoint
            if save_checkpoints:
                minutes_since_last_checkpoint = (time.time() - last_checkpoint) // 60
                # MyPy overrides are because you should
                if (
                    minutes_since_last_checkpoint >= checkpoint_frequency  # type: ignore
                    or should_stop
                    or epoch == num_epochs
                ):
                    self._save_state(path=checkpoint_path, stopper=stopper)  # type: ignore
                    last_checkpoint = time.time()

            if should_stop:
                return self.losses_per_epochs

        return self.losses_per_epochs

    def _forward_pass(self, batch, start, stop, current_batch_size, label_smoothing, slice_size):
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

        # backward pass with accelerate
        self.accelerator.backward(loss)
        current_epoch_loss = loss.item()

        self.model.post_forward_pass()
        # TODO why not call torch.cuda.empty_cache()? or call self._free_graph_and_cache()?

        return current_epoch_loss

    def _free_graph_and_cache(self):
        self.model._free_graph_and_cache()
        # The cache of the previous run has to be freed to allow accurate memory availability estimates
        # Turned off for accelerate
        # torch.cuda.empty_cache()
