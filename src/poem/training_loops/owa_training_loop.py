# -*- coding: utf-8 -*-

"""Training KGE models based on the OWA."""

import logging
import timeit

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from poem.constants import BATCH_SIZE, NUM_EPOCHS
from poem.instance_creation_factories.triples_factory import Instances
from poem.kge_models.utils import get_optimizer
from poem.model_config import ModelConfig
from poem.training_loops.basic_training_loop import TrainingLoop
from poem.training_loops.utils import split_list_in_batches

log = logging.getLogger(__name__)


class OWATrainingLoop(TrainingLoop):
    """."""

    def __init__(self, model_config: ModelConfig, kge_model: nn.Module, instances: Instances):
        super().__init__(model_config=model_config, kge_model=kge_model, instances=instances)

    def train(self):
        """."""
        self.kge_model = self.kge_model.to(self.kge_model.device)

        optimizer = get_optimizer(config=self.config, kge_model=self.kge_model)
        num_pos_triples = self.instances.training_instances.shape[0]
        num_entities = self.all_entities.shape[0]

        start_training = timeit.default_timer()

        _tqdm_kwargs = dict(desc='Training epoch')

        log.info(f'****running model on {self.kge_model.device}****')

        for _ in trange(self.config[NUM_EPOCHS], **_tqdm_kwargs):
            indices = np.arange(num_pos_triples)
            np.random.shuffle(indices)
            pos_triples = pos_triples[indices]
            pos_batches = split_list_in_batches(input_list=pos_triples, batch_size=self.config[BATCH_SIZE])
            current_epoch_loss = 0.

            for i, pos_batch in enumerate(pos_batches):
                current_batch_size = len(pos_batch)
                batch_subjs = pos_batch[:, 0:1]
                batch_relations = pos_batch[:, 1:2]
                batch_objs = pos_batch[:, 2:3]

                num_subj_corrupt = len(pos_batch) // 2
                num_obj_corrupt = len(pos_batch) - num_subj_corrupt
                pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=self.kge_model.device)

                corrupted_subj_indices = np.random.choice(np.arange(0, num_entities), size=num_subj_corrupt)
                corrupted_subjects = np.reshape(self.all_entities[corrupted_subj_indices], newshape=(-1, 1))
                subject_based_corrupted_triples = np.concatenate(
                    [corrupted_subjects, batch_relations[:num_subj_corrupt], batch_objs[:num_subj_corrupt]], axis=1)

                corrupted_obj_indices = np.random.choice(np.arange(0, num_entities), size=num_obj_corrupt)
                corrupted_objects = np.reshape(self.all_entities[corrupted_obj_indices], newshape=(-1, 1))

                object_based_corrupted_triples = np.concatenate(
                    [batch_subjs[num_subj_corrupt:], batch_relations[num_subj_corrupt:], corrupted_objects], axis=1)

                neg_batch = np.concatenate([subject_based_corrupted_triples, object_based_corrupted_triples], axis=0)

                neg_batch = torch.tensor(neg_batch, dtype=torch.long, device=self.kge_model.device)

                # Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old instance
                optimizer.zero_grad()
                loss = self.kge_model(pos_batch, neg_batch)
                current_epoch_loss += (loss.item() * current_batch_size)

                loss.backward()
                optimizer.step()

            # Track epoch loss
            self.losses_per_epochs.append(current_epoch_loss / len(pos_triples))

        stop_training = timeit.default_timer()
        log.debug("training took %.2fs seconds", stop_training - start_training)

        return self.kge_model, self.losses_per_epochs
