# -*- coding: utf-8 -*-

"""Training KGE models based on the OWA."""

from typing import Any, Mapping, Optional, Type

import torch
from torch.optim.optimizer import Optimizer

from .training_loop import TrainingLoop
from .utils import apply_label_smoothing
from ..losses import CrossEntropyLoss
from ..models.base import BaseModule
from ..sampling import BasicNegativeSampler, NegativeSampler
from ..triples import OWAInstances
from ..typing import MappedTriples

__all__ = [
    'OWATrainingLoop',
]


class OWATrainingLoop(TrainingLoop):
    """A training loop that uses the open world assumption."""

    negative_sampler: NegativeSampler
    loss_blacklist = [CrossEntropyLoss]

    def __init__(
        self,
        model: BaseModule,
        optimizer: Optional[Optimizer] = None,
        negative_sampler_cls: Optional[Type[NegativeSampler]] = None,
        negative_sampler_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """Initialize the training loop.

        :param model: The model to train
        :param optimizer: The optimizer to use while training the model
        :param negative_sampler_cls: The class of the negative sampler
        :param negative_sampler_kwargs: Keyword arguments to pass to the negative sampler class on instantiation
         for every positive one
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
        )

        if negative_sampler_cls is None:
            negative_sampler_cls = BasicNegativeSampler

        self.negative_sampler = negative_sampler_cls(
            triples_factory=self.triples_factory,
            **(negative_sampler_kwargs or {}),
        )

    @property
    def num_negs_per_pos(self) -> int:
        """Return number of negatives per positive from the sampler.

        Property for API compatibility
        """
        return self.negative_sampler.num_negs_per_pos

    def _create_instances(self) -> OWAInstances:  # noqa: D102
        return self.triples_factory.create_owa_instances()

    @staticmethod
    def _get_batch_size(batch: MappedTriples) -> int:  # noqa: D102
        return batch.shape[0]

    def _process_batch(
        self,
        batch: MappedTriples,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
    ) -> torch.FloatTensor:  # noqa: D102
        # Send positive batch to device
        positive_batch = batch[start:stop].to(device=self.device)

        # Create negative samples
        neg_samples = self.negative_sampler.sample(positive_batch=positive_batch)

        # Ensure they reside on the device (should hold already for most simple negative samplers, e.g.
        # BasicNegativeSampler, BernoulliNegativeSampler
        negative_batch = neg_samples.to(self.device)

        # Make it negative batch broadcastable (required for num_negs_per_pos > 1).
        negative_batch = negative_batch.view(-1, 3)

        # Compute negative and positive scores
        positive_scores = self.model.score_hrt(positive_batch)
        negative_scores = self.model.score_hrt(negative_batch)

        loss = self._loss_helper(
            positive_scores,
            negative_scores,
            label_smoothing,
        )
        return loss

    def _mr_loss_helper(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        _label_smoothing=None,
    ) -> torch.FloatTensor:
        # Repeat positives scores (necessary for more than one negative per positive)
        if self.num_negs_per_pos > 1:
            positive_scores = positive_scores.repeat(self.num_negs_per_pos, 1)

        return self.model.compute_mr_loss(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
        )

    def _self_adversarial_negative_sampling_loss_helper(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        _label_smoothing=None,
    ) -> torch.FloatTensor:
        """Compute self adversarial negative sampling loss."""
        return self.model.compute_self_adversarial_negative_sampling_loss(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
        )

    def _label_loss_helper(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        label_smoothing: float,
    ) -> torch.FloatTensor:
        # Stack predictions
        predictions = torch.cat([positive_scores, negative_scores], dim=0)
        # Create target
        ones = torch.ones_like(positive_scores, device=self.device)
        zeros = torch.zeros_like(negative_scores, device=self.device)
        labels = torch.cat([ones, zeros], dim=0)

        if label_smoothing > 0.:
            labels = apply_label_smoothing(
                labels=labels,
                epsilon=label_smoothing,
                num_classes=self.model.num_entities,
            )

        # Normalize the loss to have the average loss per positive triple
        # This allows comparability of OWA and LCWA losses
        return self.model.compute_label_loss(
            predictions=predictions,
            labels=labels,
        )
