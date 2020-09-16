# -*- coding: utf-8 -*-

"""Training KGE models based on the sLCWA."""

import logging
from typing import Any, Mapping, Optional, Type

import torch
from torch.optim.optimizer import Optimizer

from .training_loop import TrainingLoop
from .utils import apply_label_smoothing
from ..losses import CrossEntropyLoss
from ..models.base import Model
from ..sampling import BasicNegativeSampler, NegativeSampler
from ..triples import SLCWAInstances
from ..typing import MappedTriples

__all__ = [
    'SLCWATrainingLoop',
]

logger = logging.getLogger(__name__)


class SLCWATrainingLoop(TrainingLoop):
    """A training loop that uses the stochastic local closed world assumption training approach."""

    negative_sampler: NegativeSampler
    loss_blacklist = [CrossEntropyLoss]

    def __init__(
        self,
        model: Model,
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

    def _create_instances(self, use_tqdm: Optional[bool] = None) -> SLCWAInstances:  # noqa: D102
        return self.triples_factory.create_slcwa_instances()

    @staticmethod
    def _get_batch_size(batch: MappedTriples) -> int:  # noqa: D102
        return batch.shape[0]

    def _process_batch(
        self,
        batch: MappedTriples,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # Slicing is not possible in sLCWA training loops
        if slice_size is not None:
            raise AttributeError('Slicing is not possible for sLCWA training loops.')

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
        # This allows comparability of sLCWA and LCWA losses
        return self.model.compute_label_loss(
            predictions=predictions,
            labels=labels,
        )

    def _slice_size_search(
        self,
        batch_size: int,
        sub_batch_size: int,
        supports_sub_batching: bool,
    ) -> None:  # noqa: D102
        # Slicing is not possible for sLCWA
        if supports_sub_batching:
            report = "This model supports sub-batching, but it also requires slicing, which is not possible for sLCWA"
        else:
            report = "This model doesn't support sub-batching and slicing is not possible for sLCWA"
        logger.warning(report)
        raise MemoryError("The current model can't be trained on this hardware with these parameters.")
