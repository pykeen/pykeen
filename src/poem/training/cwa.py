# -*- coding: utf-8 -*-

"""Training KGE models based on the CWA."""

from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.optim.optimizer import Optimizer

from .training_loop import TrainingLoop
from .utils import apply_label_smoothing
from ..instance_creation_factories import Instances
from ..models import BaseModule

__all__ = [
    'CWATrainingLoop',
    'CWANotImplementedError',
]


class CWANotImplementedError(NotImplementedError):
    """Raised when trying to train with CWA on the wrong type of criterion."""


class CWATrainingLoop(TrainingLoop):
    """A training loop that uses the closed world assumption."""

    def __init__(
        self,
        model: BaseModule,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        super().__init__(model=model, optimizer=optimizer)
        if self.model.is_mr_loss:
            raise CWANotImplementedError('CWA has not been implemented for mean ranking loss yet')

    def _create_instances(self) -> Instances:  # noqa: D102
        return self.triples_factory.create_cwa_instances()

    def _compile_batch(self, batch_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # noqa: D102
        input_batch = self.training_instances.mapped_triples[batch_indices]
        target_batch = self.training_instances.labels[batch_indices]
        return input_batch, target_batch

    def _process_batch(self, batch: Any, label_smoothing: float = 0.0) -> torch.FloatTensor:  # noqa: D102
        # Split batch components
        batch_pairs, batch_labels = batch

        # Send batch to device
        batch_pairs = torch.tensor(batch_pairs, dtype=torch.long, device=self.device)

        # Construct dense target
        current_batch_size = len(batch_pairs)
        num_entities = self.model.num_entities
        batch_labels_full = torch.zeros((current_batch_size, num_entities), device=self.device)
        for i in range(current_batch_size):
            batch_labels_full[i, batch_labels[i]] = 1

        # Apply label smoothing
        if label_smoothing > 0.:
            batch_labels_full = apply_label_smoothing(
                labels=batch_labels_full,
                epsilon=label_smoothing,
                num_classes=num_entities,
            )

        predictions = self.model.forward_cwa(batch=batch_pairs)
        loss = self.model.compute_label_loss(predictions=predictions, labels=batch_labels_full) / num_entities

        return loss
