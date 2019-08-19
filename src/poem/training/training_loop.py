# -*- coding: utf-8 -*-

"""Training loops for KGE models using multi-modal information."""

from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from tqdm import tqdm, trange

from .early_stopping import EarlyStopper
from .utils import lazy_compile_random_batches
from ..instance_creation_factories import Instances, TriplesFactory
from ..models.base import BaseModule
from ..version import get_version

__all__ = [
    'TrainingLoop',
]


class TrainingLoop(ABC):
    """A training loop."""

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

    @property
    def triples_factory(self) -> TriplesFactory:  # noqa: D401
        """The triples factory in the model."""
        return self.model.triples_factory

    @property
    def all_entities(self) -> np.ndarray:  # noqa: D401
        """All entities in the triples factory."""
        return self.model.triples_factory.all_entities

    @property
    def device(self):  # noqa: D401
        """The device used by the model."""
        return self.model.device

    def train(
        self,
        num_epochs: int = 1,
        batch_size: int = 128,
        label_smoothing: float = 0.0,
        continue_training: bool = False,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
        early_stopper: Optional[EarlyStopper] = None,
    ) -> List[float]:
        """Train the KGE model.

        :param num_epochs:
            The number of epochs to train the model.
        :param batch_size:
            The batch size to use for mini-batch training.
        :param label_smoothing: (0 <= label_smoothing < 1)
            If larger than zero, use label smoothing.
        :param continue_training:
            If set to False, (re-)initialize the model's weights. Otherwise continue training.
        :param tqdm_kwargs:
            Keyword arguments passed to :mod:`tqdm` managing the progress bar.
        :param early_stopper:
            An instance of :class:`poem.training.EarlyStopper` with settings for checking
            if training should stop early

        :return:
            A pair of the KGE model and the losses per epoch.
        """
        # Sanity check
        if self.model.compute_mr_loss and label_smoothing > 0.:
            raise ValueError('Margin Ranking Loss cannot be used together with label smoothing.')

        # Ensure the model is on the correct device
        self.model: BaseModule = self.model.to(self.device)

        # Force weight initialization if training continuation is not explicitly requested.
        if not continue_training:
            self.model.reset_weights_()

        # Create training instances
        self.training_instances = self._create_instances()

        # Bind
        num_training_instances = self.training_instances.num_instances

        # Create progressbar
        _tqdm_kwargs = dict(desc=f'⚽ Training epoch on {self.device}', unit='epoch', unit_scale=True)
        if tqdm_kwargs is not None:
            _tqdm_kwargs.update(tqdm_kwargs)
        epochs = trange(num_epochs, **_tqdm_kwargs)

        # Training Loop
        indices = np.arange(num_training_instances)
        for epoch in epochs:
            # Enforce training mode
            self.model.train()

            # Accumulate loss over epoch
            current_epoch_loss = 0.

            # Batching
            for batch in lazy_compile_random_batches(
                indices=indices,
                batch_size=batch_size,
                batch_compiler=self._compile_batch
            ):
                loss = self._process_batch(batch=batch, label_smoothing=label_smoothing)

                # Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old instance
                self.optimizer.zero_grad()
                loss.backward()
                current_epoch_loss += loss.item()
                self.optimizer.step()
                # After changing applying the gradients to the embeddings, the model is notified that the forward
                # constraints are no longer applied
                self.model.forward_constraint_applied = False

            # Track epoch loss
            self.losses_per_epochs.append(current_epoch_loss / num_training_instances)

            if (
                early_stopper is not None
                and 0 == (epoch % early_stopper.frequency)  # only check with given frequency
                and early_stopper.should_stop()
            ):
                return self.losses_per_epochs

            # Print loss information to console
            epochs.set_postfix({
                'loss': self.losses_per_epochs[-1],
                'prev_loss': self.losses_per_epochs[-2] if epoch > 1 else float('nan')
            })

        return self.losses_per_epochs

    @abstractmethod
    def _create_instances(self) -> Instances:
        """Create the training instances at the beginning of the training loop."""
        raise NotImplementedError

    @abstractmethod
    def _compile_batch(self, batch_indices: np.ndarray) -> Any:
        """Compile a single training batch."""
        raise NotImplementedError

    @abstractmethod
    def _process_batch(self, batch: Any, label_smoothing: float = 0.0) -> torch.FloatTensor:
        """Process a single batch and returns the loss."""
        raise NotImplementedError

    def _get_entity_to_vector_dict(self) -> Mapping:
        raise NotImplementedError

    def to_embeddingdb(self, session=None, use_tqdm: bool = False):
        """Upload to the embedding database.

        :param session: Optional SQLAlchemy session
        :param use_tqdm: Use :mod:`tqdm` progress bar?
        :rtype: embeddingdb.sql.models.Collection
        """
        from embeddingdb.sql.models import Embedding, Collection

        if session is None:
            from embeddingdb.sql.models import get_session
            session = get_session()

        collection = Collection(
            package_name='poem',
            package_version=get_version(),
            dimensions=...,
            extras=...,
        )

        it = self._get_entity_to_vector_dict().items()
        if use_tqdm:
            it = tqdm(it, desc='Building SQLAlchemy models')
        for curie, vector in it:
            embedding = Embedding(
                collection=collection,
                curie=curie,
                vector=vector,
            )
            session.add(embedding)
        session.add(collection)
        session.commit()
        return collection
