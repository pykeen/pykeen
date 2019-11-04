# -*- coding: utf-8 -*-

"""Training KGE models based on the OWA."""

from typing import Optional, Type

import torch
from torch.optim.optimizer import Optimizer

from .training_loop import TrainingLoop
from .utils import apply_label_smoothing
from ..models.base import BaseModule
from ..sampling import BasicNegativeSampler, NegativeSampler
from ..typing import MappedTriples

__all__ = [
    'OWATrainingLoop',
]


class OWATrainingLoop(TrainingLoop):
    """A training loop that uses the open world assumption."""

    negative_sampler: NegativeSampler

    def __init__(
        self,
        model: BaseModule,
        optimizer: Optional[Optimizer] = None,
        negative_sampler_cls: Optional[Type[NegativeSampler]] = None,
        num_negs_per_pos: int = 1,
    ):
        """Initialize the training loop.

        :param model: The model to train
        :param optimizer: The optimizer to use while training the model
        :param negative_sampler_cls: The class of the negative sampler
        :param num_negs_per_pos: The number of negative triples to generate
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
        )

        # TODO: Make this part of the negative sampler?
        self.num_negs_per_pos = num_negs_per_pos

    def _create_negative_samples(
        self,
        positive_batch: torch.LongTensor,
        num_negs_per_pos: int = 1,
    ) -> torch.LongTensor:
        # TODO: Pass num_negs_per_pos to sampler to allow further optimization
        return torch.cat(
            [
                self.negative_sampler.sample(positive_batch=positive_batch)
                for _ in range(num_negs_per_pos)
            ],
            dim=0,
        )

    def _create_instances(self):  # noqa: D102
        return self.triples_factory.create_owa_instances()

    def _process_batch(
        self,
        batch: MappedTriples,
        label_smoothing: float = 0.0,
    ) -> torch.FloatTensor:  # noqa: D102
        # Send positive batch to device
        positive_batch = batch.to(device=self.device)

        # Create negative samples
        neg_samples = self._create_negative_samples(
            positive_batch=positive_batch,
            num_negs_per_pos=self.num_negs_per_pos,
        )

        # Ensure they reside on the device (should hold already for most simple negative samplers, e.g.
        # BasicNegativeSampler, BernoulliNegativeSampler
        negative_batch = neg_samples.to(self.device)

        # Make it negative batch broadcastable (required for self.num_negs_per_pos > 1).
        negative_batch = negative_batch.view(-1, 3)

        # Compute negative and positive scores
        positive_scores = self.model.forward_owa(positive_batch)
        negative_scores = self.model.forward_owa(negative_batch)

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
        # This allows comparability of OWA and CWA losses
        return self.model.compute_label_loss(
            predictions=predictions,
            labels=labels,
        )
