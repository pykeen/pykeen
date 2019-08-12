# -*- coding: utf-8 -*-

"""Training KGE models based on the OWA."""

from typing import Any, List, Mapping, Optional, Type

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from tqdm import trange

from .early_stopping import EarlyStopper
from .training_loop import TrainingLoop
from .utils import split_list_in_batches
from ..models import BaseModule
from ..negative_sampling import BasicNegativeSampler, NegativeSampler

__all__ = [
    'OWATrainingLoop',
]


class OWATrainingLoop(TrainingLoop):
    """A training loop using the OWA."""

    negative_sampler: NegativeSampler

    def __init__(
            self,
            model: Optional[BaseModule] = None,
            optimizer: Optional[Optimizer] = None,
            negative_sampler_cls: Type[NegativeSampler] = None,
    ):
        """Initialize the training loop."""
        super().__init__(
            model=model,
            optimizer=optimizer,
        )

        if negative_sampler_cls is None:
            negative_sampler_cls = BasicNegativeSampler

        self.negative_sampler = negative_sampler_cls(all_entities=self.all_entities)

    def _create_negative_samples(self, pos_batch, num_negs_per_pos=1):
        return [
            self.negative_sampler.sample(positive_batch=pos_batch)
            for _ in range(num_negs_per_pos)
        ]

    def train(
            self,
            num_epochs: int,
            batch_size: int,
            num_negs_per_pos: int = 1,
            label_smoothing: bool = False,
            label_smoothing_epsilon: float = 0.1,
            tqdm_kwargs: Optional[Mapping[str, Any]] = None,
            early_stopper: Optional[EarlyStopper] = None,
    ) -> List[float]:
        """Train the KGE model."""
        if self.model.compute_mr_loss and label_smoothing:
            raise ValueError('Margin Ranking Loss cannot be used together with label smoothing')

        training_instances = self.model.triples_factory.create_owa_instances()
        pos_triples = training_instances.instances
        num_pos_triples = training_instances.num_instances
        num_entities = training_instances.num_entities

        _tqdm_kwargs = dict(desc=f'Training epoch on {self.device}')
        if tqdm_kwargs is not None:
            _tqdm_kwargs.update(tqdm_kwargs)
        epochs = trange(num_epochs, **_tqdm_kwargs)
        for epoch in epochs:
            indices = np.arange(num_pos_triples)
            np.random.shuffle(indices)
            pos_triples = pos_triples[indices]
            pos_batches = split_list_in_batches(input_list=pos_triples, batch_size=batch_size)
            current_epoch_loss = 0.

            for i, pos_batch in enumerate(pos_batches):
                current_batch_size = len(pos_batch)

                neg_samples = self._create_negative_samples(pos_batch, num_negs_per_pos=num_negs_per_pos)
                pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=self.device)
                neg_batch = torch.tensor(neg_samples, dtype=torch.long, device=self.device).view(-1, 3)

                positive_scores = self.model.forward_owa(pos_batch)
                positive_scores = positive_scores.repeat(num_negs_per_pos, 1)
                negative_scores = self.model.forward_owa(neg_batch)

                """
                TODO: Define two functions, one for compute_mr_loss() and the other for model.compute_label_loss()
                Check for self.model.compute_mr_loss when entering train(), and assign corresponding fct.
                Avoids repetitive checks.
                """
                if self.model.is_mr_loss:
                    loss = self.model.compute_mr_loss(
                        positive_scores=positive_scores,
                        negative_scores=negative_scores,
                    )
                else:
                    predictions = torch.cat([positive_scores, negative_scores], 0)
                    ones = torch.ones_like(positive_scores, device=self.device)
                    zeros = torch.zeros_like(negative_scores, device=self.device)
                    labels = torch.cat([ones, zeros], 0)

                    if label_smoothing:
                        # TODO give variables better names
                        _a = labels * (1.0 - label_smoothing_epsilon)
                        _b = label_smoothing_epsilon / (num_entities - 1)
                        labels = _a + _b

                    loss = self.model.compute_label_loss(predictions=predictions, labels=labels)

                # Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old instance
                self.optimizer.zero_grad()
                loss.backward()
                current_epoch_loss += (loss.item() * current_batch_size * num_negs_per_pos)
                self.optimizer.step()
                # After changing applying the gradients to the embeddings, the model is notified that the forward
                # constraints are no longer applied
                self.model.forward_constraint_applied = False

            # Track epoch loss
            self.losses_per_epochs.append(current_epoch_loss / (len(pos_triples) * num_negs_per_pos))

            if (
                    early_stopper is not None
                    and 0 == (epoch % early_stopper.frequency)  # only check with given frequency
                    and early_stopper.should_stop()
            ):
                return self.losses_per_epochs

            epochs.write(f'Losses: {self.losses_per_epochs}')

        return self.losses_per_epochs
