# -*- coding: utf-8 -*-

"""Training loops for KGE models using multi-modal information."""

from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .early_stopping import EarlyStopper
from ..models.base import BaseModule
from ..training.schlichtkrull_sampler import GraphSampler
from ..triples import Instances, TriplesFactory
from ..utils import ResultTracker

__all__ = [
    'TrainingLoop',
]


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

        if self.model.is_mr_loss:
            self._loss_helper = self._mr_loss_helper
        elif self.model.is_self_adversiarial_neg_sampling_loss:
            self._loss_helper = self._self_adversarial_negative_sampling_loss_helper
        else:
            self._loss_helper = self._label_loss_helper

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
        batch_size: int = 128,
        label_smoothing: float = 0.0,
        sampler: Optional[str] = None,
        continue_training: bool = False,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
        early_stopper: Optional[EarlyStopper] = None,
        result_tracker: Optional[ResultTracker] = None,
        sub_batch_size: Optional[int] = None,
    ) -> List[float]:
        """Train the KGE model.

        :param num_epochs:
            The number of epochs to train the model.
        :param batch_size:
            The batch size to use for mini-batch training.
        :param label_smoothing: (0 <= label_smoothing < 1)
            If larger than zero, use label smoothing.
        :param sampler: (None or 'schlichtkrull')
            The type of sampler to use. At the moment OWA in R-GCN is the only user of schlichtkrull sampling.
        :param continue_training:
            If set to False, (re-)initialize the model's weights. Otherwise continue training.
        :param tqdm_kwargs:
            Keyword arguments passed to :mod:`tqdm` managing the progress bar.
        :param early_stopper:
            An instance of :class:`poem.training.EarlyStopper` with settings for checking
            if training should stop early
        :param result_tracker:
            The result tracker.
        :param sub_batch_size:
            If provided split each batch into sub-batches to avoid memory issues for large models / small GPUs.
        :return:
            A pair of the KGE model and the losses per epoch.
        """
        # Create dummy result tracker
        if result_tracker is None:
            result_tracker = ResultTracker()

        # by default do not split batches in sub-batches
        if sub_batch_size is None:
            sub_batch_size = batch_size

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

        # Create training instances
        self.training_instances = self._create_instances()

        # Create Sampler
        if sampler == 'schlichtkrull':
            sampler = GraphSampler(self.triples_factory)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_data_loader = DataLoader(
            sampler=sampler,
            dataset=self.training_instances,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        # Bind
        num_training_instances = self.training_instances.num_instances

        # Create progress bar
        _tqdm_kwargs = dict(desc=f'Training on {self.device}', unit='epoch')
        if tqdm_kwargs is not None:
            _tqdm_kwargs.update(tqdm_kwargs)
        epochs = trange(1, 1 + num_epochs, **_tqdm_kwargs)

        # Training Loop
        for epoch in epochs:
            # Enforce training mode
            self.model.train()

            # Accumulate loss over epoch
            current_epoch_loss = 0.

            # Batching
            batches = tqdm(train_data_loader, desc=f'Training batches', leave=False, unit='batch')
            for batch in batches:
                # Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old instance
                self.optimizer.zero_grad()

                # accumulate gradients for whole batch
                for start in range(0, batch_size, sub_batch_size):
                    stop = max(start + sub_batch_size, batch_size)

                    # forward pass
                    loss = self._process_batch(batch=batch, start=start, stop=stop, label_smoothing=label_smoothing)

                    # correction for loss reduction
                    if self.model.criterion.reduction == 'mean':
                        this_sub_batch_size = stop - start
                        loss *= (this_sub_batch_size / batch_size)

                    # backward pass
                    loss.backward()
                    current_epoch_loss += loss.item()

                # update parameters according to optimizer
                self.optimizer.step()

                # After changing applying the gradients to the embeddings, the model is notified that the forward
                # constraints are no longer applied
                self.model.post_parameter_update()

            # Track epoch loss
            epoch_loss = current_epoch_loss / num_training_instances
            self.losses_per_epochs.append(epoch_loss)
            result_tracker.log_metrics({'loss': epoch_loss}, step=epoch)

            if (
                early_stopper is not None
                and 0 == ((epoch - 1) % early_stopper.frequency)  # only check with given frequency
                and early_stopper.should_stop()
            ):
                return self.losses_per_epochs

            # Print loss information to console
            epochs.set_postfix({
                'loss': self.losses_per_epochs[-1],
                'prev_loss': self.losses_per_epochs[-2] if epoch > 2 else float('nan'),
            })

        return self.losses_per_epochs

    @abstractmethod
    def _create_instances(self) -> Instances:
        """Create the training instances at the beginning of the training loop."""
        raise NotImplementedError

    @abstractmethod
    def _process_batch(self, batch: Any, start: int, stop: int, label_smoothing: float = 0.0) -> torch.FloatTensor:
        """Process a single batch and returns the loss."""
        raise NotImplementedError

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
