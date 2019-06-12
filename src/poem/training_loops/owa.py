# -*- coding: utf-8 -*-

"""Training KGE models based on the OWA."""

import logging
import timeit

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from .base import TrainingLoop
from .utils import split_list_in_batches
from ..negative_sampling.basic_negative_sampler import BasicNegativeSampler

__all__ = [
    'OWATrainingLoop',
]

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class OWATrainingLoop(TrainingLoop):
    def __init__(self, kge_model: nn.Module, optimizer, all_entities, negative_sampler=None):
        super().__init__(kge_model=kge_model, optimizer=optimizer, all_entities=all_entities)
        # Later, different negative sampling algorithms can be set
        self.negative_sampler = negative_sampler or BasicNegativeSampler(all_entities=self.all_entities)

    def _create_negative_samples(self, pos_batch, num_negs_per_pos=1):
        """."""
        list_neg_batches = []

        for _ in range(num_negs_per_pos):
            neg_batch = self.negative_sampler.sample(positive_batch=pos_batch)
            list_neg_batches.append(neg_batch)

        return list_neg_batches

    def train(self, training_instances, num_epochs, batch_size, num_negs_per_pos=1):
        pos_triples = training_instances.instances
        num_pos_triples = pos_triples.shape[0]

        start_training = timeit.default_timer()

        _tqdm_kwargs = dict(desc='Training epoch')

        log.info(f'****running model on {self.kge_model.device}****')

        for _ in trange(num_epochs, **_tqdm_kwargs):
            indices = np.arange(num_pos_triples)
            np.random.shuffle(indices)
            pos_triples = pos_triples[indices]
            pos_batches = split_list_in_batches(input_list=pos_triples, batch_size=batch_size)
            current_epoch_loss = 0.

            for i, pos_batch in enumerate(pos_batches):
                current_batch_size = len(pos_batch)

                self.optimizer.zero_grad()
                pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=self.kge_model.device)

                neg_samples = self._create_negative_samples(pos_batch, num_negs_per_pos=num_negs_per_pos)
                neg_batch = torch.tensor(neg_samples, dtype=torch.long, device=self.kge_model.device).view(-1, 3)

                positive_scores = self.kge_model(pos_batch)
                positive_scores = positive_scores.repeat(num_negs_per_pos)
                negative_scores = self.kge_model(neg_batch)

                loss = self.kge_model.compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)

                # Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old instance
                # self.optimizer.zero_grad()
                # loss = compute_loss_fct(loss_fct)
                # loss = self.kge_model(pos_batch, neg_batch)
                # current_epoch_loss += (loss.item() * current_batch_size)

                current_epoch_loss += (loss.item() * current_batch_size * num_negs_per_pos)

                loss.backward()
                self.optimizer.step()

            # Track epoch loss
            self.losses_per_epochs.append(current_epoch_loss / len(pos_triples))

        stop_training = timeit.default_timer()
        log.debug("training took %.2fs seconds", stop_training - start_training)

        return self.kge_model, self.losses_per_epochs
