# -*- coding: utf-8 -*-

"""Training loops for KGE models using multi-modal information."""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional, Tuple, Type, Union

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from ..losses import Loss
from ..models.base import BaseModule
from ..stoppers import Stopper
from ..training.schlichtkrull_sampler import GraphSampler
from ..triples import Instances, TriplesFactory
from ..typing import MappedTriples
from ..utils import ResultTracker, normalize_string

__all__ = [
    'TrainingLoop',
    'NonFiniteLossError',
    'AssumptionLossMismatchError',
    'SubBatchingNotSupportedError',
]

logger = logging.getLogger(__name__)


class NonFiniteLossError(RuntimeError):
    """An exception raised for non-finite loss values."""


class AssumptionLossMismatchError(TypeError):
    """An exception when an illegal loss function is used with a given training assumption."""


class SubBatchingNotSupportedError(NotImplementedError):
    """An exception raised when sub batching is not implemented."""

    def __init__(self, model: BaseModule):
        super().__init__(model)
        self.model = model

    def __str__(self):  # noqa: D105
        return f'No sub-batching support for {self.model.__class__.__name__} due to modules ' \
               f'{self.model.modules_not_supporting_sub_batching}.'


def _get_optimizer_kwargs(optimizer: Optimizer) -> Mapping[str, Any]:
    optimizer_kwargs = optimizer.state_dict()
    optimizer_kwargs = {
        key: value
        for key, value in optimizer_kwargs['param_groups'][0].items()
        if key != 'params'
    }
    return optimizer_kwargs


class TrainingLoop(ABC):
    """A training loop."""

    training_instances: Optional[Instances]
    losses_per_epochs: List[float]
    loss_blacklist: Optional[List[Type[Loss]]] = None

    hpo_default = dict(
        num_epochs=dict(type=int, low=100, high=1000, q=100),
        batch_size=dict(type=int, low=32, high=4000, q=100),
    )

    def __init__(
        self,
        model: BaseModule,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the training loop.

        :param model: The model to train
        :param optimizer: The optimizer to use while training the model
        """
        self.model = model
        self.optimizer = optimizer
        self.training_instances = None
        self.losses_per_epochs = []

        if self.loss_blacklist and isinstance(self.model.loss, tuple(self.loss_blacklist)):
            raise AssumptionLossMismatchError(
                f'Can not use loss {self.model.loss.__class__.__name__}'
                f' with training assumption {self.__class__.__name__}',
            )

        if self.model.is_mr_loss:
            self._loss_helper = self._mr_loss_helper
        elif self.model.is_self_adversiarial_neg_sampling_loss:
            self._loss_helper = self._self_adversarial_negative_sampling_loss_helper
        else:
            self._loss_helper = self._label_loss_helper

    @classmethod
    def get_normalized_name(cls) -> str:
        """Get the normalized name of the training loop."""
        return normalize_string(cls.__name__, suffix=TrainingLoop.__name__)

    @property
    def triples_factory(self) -> TriplesFactory:  # noqa: D401
        """The triples factory in the model."""
        return self.model.triples_factory

    @property
    def device(self):  # noqa: D401
        """The device used by the model."""
        return self.model.device

    def train(
        self,
        num_epochs: int = 1,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        label_smoothing: float = 0.0,
        sampler: Optional[str] = None,
        continue_training: bool = False,
        only_size_probing: bool = False,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
        stopper: Optional[Stopper] = None,
        result_tracker: Optional[ResultTracker] = None,
        sub_batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> List[float]:
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
            The type of sampler to use. At the moment OWA in R-GCN is the only user of schlichtkrull sampling.
        :param continue_training:
            If set to False, (re-)initialize the model's weights. Otherwise continue training.
        :param only_size_probing:
            The evaluation is only performed for two batches to test the memory footprint, especially on GPUs.
        :param tqdm_kwargs:
            Keyword arguments passed to :mod:`tqdm` managing the progress bar.
        :param stopper:
            An instance of :class:`poem.stopper.EarlyStopper` with settings for checking
            if training should stop early
        :param result_tracker:
            The result tracker.
        :param sub_batch_size:
            If provided split each batch into sub-batches to avoid memory issues for large models / small GPUs.
        :param num_workers:
            The number of child CPU workers used for loading data. If None, data are loaded in the main process.

        :return:
            A pair of the KGE model and the losses per epoch.
        """
        # Create training instances
        # During size probing the training instances should not show the tqdm progress bar
        self.training_instances = self._create_instances(use_tqdm=not only_size_probing)

        return self._train(
            num_epochs=num_epochs,
            batch_size=batch_size,
            slice_size=slice_size,
            label_smoothing=label_smoothing,
            sampler=sampler,
            continue_training=continue_training,
            only_size_probing=only_size_probing,
            tqdm_kwargs=tqdm_kwargs,
            stopper=stopper,
            result_tracker=result_tracker,
            sub_batch_size=sub_batch_size,
            num_workers=num_workers,
        )

    def _train(  # noqa: C901
        self,
        num_epochs: int = 1,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        label_smoothing: float = 0.0,
        sampler: Optional[str] = None,
        continue_training: bool = False,
        only_size_probing: bool = False,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
        stopper: Optional[Stopper] = None,
        result_tracker: Optional[ResultTracker] = None,
        sub_batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> List[float]:
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
            The type of sampler to use. At the moment OWA in R-GCN is the only user of schlichtkrull sampling.
        :param continue_training:
            If set to False, (re-)initialize the model's weights. Otherwise continue training.
        :param only_size_probing:
            The evaluation is only performed for two batches to test the memory footprint, especially on GPUs.
        :param tqdm_kwargs:
            Keyword arguments passed to :mod:`tqdm` managing the progress bar.
        :param stopper:
            An instance of :class:`poem.stopper.EarlyStopper` with settings for checking
            if training should stop early
        :param result_tracker:
            The result tracker.
        :param sub_batch_size:
            If provided split each batch into sub-batches to avoid memory issues for large models / small GPUs.
        :param num_workers:
            The number of child CPU workers used for loading data. If None, data are loaded in the main process.

        :return:
            A pair of the KGE model and the losses per epoch.
        """
        # Take the biggest possible training batch_size, if batch_size not set
        if batch_size is None:
            if self.model.automatic_memory_optimization:
                batch_size, _ = self.batch_size_search()
            else:
                batch_size = 256

        # This will find necessary parameters to optimize the use of the hardware at hand
        if not only_size_probing and self.model.automatic_memory_optimization:
            # return the relevant parameters slice_size and batch_size
            sub_batch_size, slice_size = self.sub_batch_and_slice(batch_size)

        # Create dummy result tracker
        if result_tracker is None:
            result_tracker = ResultTracker()

        if sub_batch_size is None or sub_batch_size == batch_size:  # by default do not split batches in sub-batches
            sub_batch_size = batch_size
        elif not self.model.supports_subbatching:
            raise SubBatchingNotSupportedError(self.model)

        # Sanity check
        if self.model.is_mr_loss and label_smoothing > 0.:
            raise RuntimeError('Label smoothing can not be used with margin ranking loss.')

        # Ensure the model is on the correct device
        self.model: BaseModule = self.model.to(self.device)

        # Force weight initialization if training continuation is not explicitly requested.
        if not continue_training:
            # Reset the weights
            self.model.reset_weights_()

            # Create new optimizer
            optimizer_kwargs = _get_optimizer_kwargs(self.optimizer)
            self.optimizer = self.optimizer.__class__(
                params=self.model.get_grad_params(),
                **optimizer_kwargs
            )
        elif not self.optimizer.state:
            raise ValueError('Cannot continue_training without being trained once.')

        # Create Sampler
        if sampler == 'schlichtkrull':
            sampler = GraphSampler(self.triples_factory, num_samples=sub_batch_size)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        if num_workers is None:
            num_workers = 0

        train_data_loader = DataLoader(
            sampler=sampler,
            dataset=self.training_instances,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        # Bind
        num_training_instances = self.training_instances.num_instances

        # When size probing, we don't want progress bars
        if not only_size_probing:
            # Create progress bar
            _tqdm_kwargs = dict(desc=f'Training on {self.device}', unit='epoch')
            if tqdm_kwargs is not None:
                _tqdm_kwargs.update(tqdm_kwargs)
            epochs = trange(1, 1 + num_epochs, **_tqdm_kwargs)
            logger.info(f'using stopper: {stopper}')
        else:
            epochs = range(1, 1 + num_epochs)
            logger.debug(f'using stopper: {stopper}')

        # Training Loop
        for epoch in epochs:
            # Enforce training mode
            self.model.train()

            # Accumulate loss over epoch
            current_epoch_loss = 0.

            # Batching
            # Only create a progress bar when not in size probing mode
            if not only_size_probing:
                batches = tqdm(train_data_loader, desc=f'Training batches', leave=False, unit='batch')
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

                    # forward pass
                    loss = self._process_batch(
                        batch=batch,
                        start=start,
                        stop=stop,
                        label_smoothing=label_smoothing,
                        slice_size=slice_size
                    )

                    # raise error when non-finite loss occurs (NaN, +/-inf)
                    if not torch.isfinite(loss):
                        raise NonFiniteLossError

                    # correction for loss reduction
                    if self.model.loss.reduction == 'mean':
                        this_sub_batch_size = stop - start
                        loss *= (this_sub_batch_size / current_batch_size)

                    # backward pass
                    loss.backward()
                    current_epoch_loss += loss.item()

                    # reset the regularizer to free the computational graph
                    self.model.regularizer.reset()

                # update parameters according to optimizer
                self.optimizer.step()

                # After changing applying the gradients to the embeddings, the model is notified that the forward
                # constraints are no longer applied
                self.model.post_parameter_update()

                # For testing purposes we're only interested in processing one batch
                if only_size_probing and evaluated_once:
                    break

                evaluated_once = True

            # Track epoch loss
            epoch_loss = current_epoch_loss / num_training_instances
            self.losses_per_epochs.append(epoch_loss)
            result_tracker.log_metrics({'loss': epoch_loss}, step=epoch)

            if stopper is not None and stopper.should_evaluate(epoch) and stopper.should_stop():
                return self.losses_per_epochs

            if not only_size_probing:
                # Print loss information to console
                epochs.set_postfix({
                    'loss': self.losses_per_epochs[-1],
                    'prev_loss': self.losses_per_epochs[-2] if epoch > 2 else float('nan'),
                })

        return self.losses_per_epochs

    @staticmethod
    @abstractmethod
    def _get_batch_size(batch: Union[MappedTriples, Tuple[MappedTriples, torch.FloatTensor]]) -> int:
        """Get the batch size from a (sub-) batch."""
        raise NotImplementedError

    @abstractmethod
    def _create_instances(self, use_tqdm: Optional[bool] = None) -> Instances:
        """Create the training instances at the beginning of the training loop."""
        raise NotImplementedError

    @abstractmethod
    def _process_batch(
        self,
        batch: Any,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """Process a single batch and returns the loss."""
        raise NotImplementedError

    def batch_size_search(
        self,
        batch_size: int = 8192,
    ) -> Tuple[int, bool]:
        """Find the maximum batch size for training with the current setting.

        This method checks how big the batch size can be for the current model with the given training data and the
        hardware at hand. If possible, the method will output the determined batch size and a boolean value indicating
        that this batch size was successfully evaluated. Otherwise, the output will be batch size 1 and the boolean
        value will be False.

        :param batch_size:
            The batch size to start the search with.

        :return:
            Tuple containing the maximum possible batch size as well as an indicator if the evaluation with that size
            was successful.
        """
        reached_max = False
        evaluated_once = False
        logger.info(f'Starting batch_size search for training now...')
        while True:
            logger.debug(f'Trying batch_size={batch_size}')
            try:
                self._train(num_epochs=1, batch_size=batch_size, sub_batch_size=None, only_size_probing=True)
            except RuntimeError as e:
                if 'CUDA out of memory.' not in e.args[0]:
                    raise e
                if batch_size == 1:
                    logger.debug(
                        f"Batch_size {batch_size} does not fit into your memory with these parameters."
                    )
                    break

                if evaluated_once:
                    # Weird error requiring to half the batch size to avoid OOM
                    batch_size //= 2
                    logger.info(f'Concluded batch_size search with batch_size={batch_size}.')
                    break

                logger.debug(f'The batch_size {batch_size} was too big, trying less now')
                reached_max = True
                batch_size //= 2
            else:
                if not reached_max:
                    batch_size *= 2
                else:
                    # Weird error requiring to half the batch size to avoid OOM
                    batch_size //= 2
                    logger.info(f'Concluded batch_size search with batch_size={batch_size}.')
                    break

                evaluated_once = True

        return batch_size, evaluated_once

    def sub_batch_and_slice(self, batch_size: int) -> Tuple[int, int]:
        """Check if sub-batching and/or slicing is necessary to train the model on the hardware at hand."""
        sub_batch_size, finished_search, supports_sub_batching = self._sub_batch_size_search(batch_size=batch_size)
        # If the sub_batch_size did not finish search with a possibility that fits the hardware, we have to try slicing
        if not finished_search:
            slice_size = self._slice_size_search(
                batch_size=batch_size,
                sub_batch_size=sub_batch_size,
                supports_sub_batching=supports_sub_batching,
            )
        else:
            slice_size = None

        return sub_batch_size, slice_size

    @abstractmethod
    def _slice_size_search(self, batch_size: int, sub_batch_size: int, supports_sub_batching: bool) -> int:
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

    def _sub_batch_size_search(self, batch_size: int) -> Tuple[int, bool, bool]:
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
            logger.debug(f'Trying batch_size {batch_size} for training now.')
            self._train(num_epochs=1, batch_size=batch_size, sub_batch_size=sub_batch_size, only_size_probing=True)
        except RuntimeError as e:
            if 'CUDA out of memory.' not in e.args[0]:
                raise e
            logger.debug(f'The batch_size {batch_size} was too big, sub_batching is required.')
            sub_batch_size //= 2
        else:
            finished_search = True
            logger.debug(f'No sub-batching required.')

        if not finished_search:
            logger.info(f'Starting sub_batch_size search for training now...')
            if not self.model.supports_subbatching:
                logger.info(f'This model does not support sub-batching.')
                supports_sub_batching = False
                sub_batch_size = batch_size
            else:
                while True:
                    logger.debug(f'Trying sub_batch_size {sub_batch_size} now.')
                    try:
                        self.train(
                            num_epochs=1,
                            batch_size=batch_size,
                            sub_batch_size=sub_batch_size,
                            only_size_probing=True
                        )
                    except RuntimeError as e:
                        if 'CUDA out of memory.' not in e.args[0]:
                            raise e
                        if sub_batch_size == 1:
                            logger.info(
                                f"Even sub_batch_size={sub_batch_size} does not fit in memory with these parameters")
                            break
                        logger.debug(f'The sub_batch_size {sub_batch_size} was too big, trying less now.')
                        sub_batch_size //= 2
                    else:
                        finished_search = True
                        logger.info(f'Concluded search with sub_batch_size {sub_batch_size}.')
                        break

        return sub_batch_size, finished_search, supports_sub_batching

    def _mr_loss_helper(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        _label_smoothing=None,
    ) -> torch.FloatTensor:
        raise NotImplementedError

    def _label_loss_helper(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        label_smoothing: float,
    ) -> torch.FloatTensor:
        raise NotImplementedError

    def _self_adversarial_negative_sampling_loss_helper(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        _label_smoothing=None,
    ) -> torch.FloatTensor:
        raise NotImplementedError

    def to_embeddingdb(self, session=None, use_tqdm: bool = False):
        """Upload to the embedding database.

        :param session: Optional SQLAlchemy session
        :param use_tqdm: Use :mod:`tqdm` progress bar?
        :rtype: embeddingdb.sql.models.Collection
        """
        return self.model.to_embeddingdb(session=session, use_tqdm=use_tqdm)
