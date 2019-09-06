# -*- coding: utf-8 -*-

"""Training KGE models based on the CWA."""

from typing import Tuple

import torch

from .training_loop import TrainingLoop
from .utils import apply_label_smoothing
from ..triples import Instances

__all__ = [
    'CWATrainingLoop',
]


class CWATrainingLoop(TrainingLoop):
    """A training loop that uses the closed world assumption."""

    def _create_instances(self) -> Instances:  # noqa: D102
        return self.triples_factory.create_cwa_instances()

    def _process_batch(
        self,
        batch: Tuple[torch.LongTensor, torch.FloatTensor],
        label_smoothing: float = 0.0,
    ) -> torch.FloatTensor:  # noqa: D102
        # Split batch components
        batch_pairs, batch_labels_full = batch

        # Send batch to device
        batch_pairs = batch_pairs.to(device=self.device)
        batch_labels_full = batch_labels_full.to(device=self.device)

        predictions = self.model.forward_cwa(batch=batch_pairs)

        loss = self._loss_helper(
            predictions,
            batch_labels_full,
            label_smoothing,
        )
        return loss

    def _label_loss_helper(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
        label_smoothing: float,
    ) -> torch.FloatTensor:
        # Apply label smoothing
        if label_smoothing > 0.:
            labels = apply_label_smoothing(
                labels=labels,
                epsilon=label_smoothing,
                num_classes=self.model.num_entities,
            )

        loss = self.model.compute_label_loss(
            predictions=predictions,
            labels=labels,
        )

        return loss

    def _mr_loss_helper(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
        _label_smoothing=None,
    ) -> torch.FloatTensor:
        # This shows how often one row has to be repeated
        repeat_rows = (labels == 1).nonzero()[:, 0]
        # Create boolean indices for negative labels in the repeated rows
        labels_negative = labels[repeat_rows] == 0
        # Repeat the predictions and filter for negative labels
        negative_scores = predictions[repeat_rows][labels_negative]

        # This tells us how often each true label should be repeated
        repeat_true_labels = (labels[repeat_rows] == 0).nonzero()[:, 0]
        # First filter the predictions for true labels and then repeat them based on the repeat vector
        positive_scores = predictions[labels == 1][repeat_true_labels]

        return self.model.compute_mr_loss(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
        )
