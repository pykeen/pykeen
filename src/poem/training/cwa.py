# -*- coding: utf-8 -*-

"""Training KGE models based on the CWA."""

from typing import List

import numpy as np
import torch
from tqdm import trange

from .training_loop import TrainingLoop
from .utils import apply_label_smoothing, lazy_compile_random_batches

__all__ = [
    'CWATrainingLoop',
]


class CWATrainingLoop(TrainingLoop):
    """A training loop that uses the closed world assumption."""

    def train(
            self,
            num_epochs: int,
            batch_size: int,
            label_smoothing: bool = True,
            label_smoothing_epsilon: float = 0.1,
    ) -> List[float]:
        """Train the model using the closed world assumption."""
        # Ensure the model is on the correct device
        self.model = self.model.to(self.device)

        training_instances = self.triples_factory.create_cwa_instances()
        subject_relation_pairs = training_instances.instances
        labels = training_instances.labels
        num_entities = self.model.num_entities

        num_triples = subject_relation_pairs.shape[0]
        indices = np.arange(num_triples)

        _tqdm_kwargs = dict(desc=f'Training epoch on {self.device}')
        for _ in trange(num_epochs, **_tqdm_kwargs):
            # Compile training batches using random shuffling
            train_batches = lazy_compile_random_batches(
                indices=indices,
                input_array=subject_relation_pairs,
                target_array=labels,
                batch_size=batch_size,
            )

            # Enforce training mode
            self.model.train()

            current_epoch_loss = 0.
            for batch_pairs, batch_labels in train_batches:
                current_batch_size = len(batch_pairs)
                batch_pairs = torch.tensor(batch_pairs, dtype=torch.long, device=self.device)

                # Construct dense target
                batch_labels_full = torch.zeros((current_batch_size, num_entities), device=self.device)
                for i in range(current_batch_size):
                    batch_labels_full[i, batch_labels[i]] = 1

                if label_smoothing:
                    batch_labels_full = apply_label_smoothing(
                        batch_labels_full,
                        label_smoothing_epsilon,
                        num_entities,
                    )

                # Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old instance
                self.optimizer.zero_grad()
                predictions = self.model.forward_cwa(batch_pairs)
                loss = self.model.compute_label_loss(predictions, batch_labels_full)
                current_epoch_loss += (loss.item() * current_batch_size)

                loss.backward()
                self.optimizer.step()

            # Track epoch loss
            self.losses_per_epochs.append(current_epoch_loss / num_triples)

        return self.losses_per_epochs
