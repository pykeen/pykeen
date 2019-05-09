# -*- coding: utf-8 -*-

"""Training KGE models based on the CWA."""

import logging
import timeit

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from poem.instance_creation_factories.instances import Instances
from poem.negative_sampling.basic_negative_sampler import BasicNegativeSampler
from poem.training_loops.basic_training_loop import TrainingLoop
from poem.training_loops.utils import split_list_in_batches

log = logging.getLogger(__name__)


class CWATrainingLoop(TrainingLoop):
    """."""

    def __init__(self, kge_model: nn.Module, optimizer, all_entities):
        super().__init__(kge_model=kge_model, optimizer=optimizer, all_entities=all_entities)

    def train(self, training_instances: Instances, num_epochs: int, batch_size: int):
        """."""
        self.kge_model = self.kge_model.to(self.kge_model.device)
        subject_relation_pairs = training_instances.instances
        labels = training_instances.labels
        num_triples = subject_relation_pairs.shape[0]

        start_training = timeit.default_timer()

        _tqdm_kwargs = dict(desc='Training epoch')

        log.info(f'****running model on {self.kge_model.device}****')

        for _ in trange(num_epochs, **_tqdm_kwargs):
            indices = np.arange(num_triples)
            np.random.shuffle(indices)
            subject_relation_pairs = subject_relation_pairs[indices]
            batches = split_list_in_batches(input_list=subject_relation_pairs, batch_size=batch_size)
            labels_batches = split_list_in_batches(input_list=labels, batch_size=batch_size)
            current_epoch_loss = 0.

            for i, batch_pairs in enumerate(batches):
                current_batch_size = len(batch_pairs)
                batch_pairs = torch.tensor(batch_pairs, dtype=torch.long, device=self.kge_model.device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.float, device=self.kge_model.device)

                # Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old instance
                self.optimizer.zero_grad()
                loss = self.kge_model(batch_pairs, batch_pairs)
                current_epoch_loss += (loss.item() * current_batch_size)

                loss.backward()
                self.optimizer.step()

            # Track epoch loss
            self.losses_per_epochs.append(current_epoch_loss / len(subject_relation_pairs))

        stop_training = timeit.default_timer()
        log.debug("training took %.2fs seconds", stop_training - start_training)

        return self.kge_model, self.losses_per_epochs
