# -*- coding: utf-8 -*-

"""Training KGE models based on the CWA."""

from typing import Any, Mapping, Optional, Tuple, Type

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
        optimizer_cls: Optional[Type[Optimizer]] = None,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
        )
        if self.model.is_mr_loss:
            raise CWANotImplementedError('CWA has not been implemented for mean ranking loss yet')

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

        # Bind number of entities
        num_entities = self.model.num_entities

        # Apply label smoothing
        if label_smoothing > 0.:
            batch_labels_full = apply_label_smoothing(
                labels=batch_labels_full,
                epsilon=label_smoothing,
                num_classes=num_entities,
            )

        predictions = self.model.forward_cwa(batch=batch_pairs)
        # Normalize the loss to have the average loss per positive triple
        # This allows comparability of OWA and CWA losses
        loss = self.model.compute_label_loss(predictions=predictions, labels=batch_labels_full) / num_entities

        return loss
