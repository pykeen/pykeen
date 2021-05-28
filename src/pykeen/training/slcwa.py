# -*- coding: utf-8 -*-

"""Training KGE models based on the sLCWA."""

import logging
from typing import Any, Mapping, Optional

import torch
from class_resolver import HintOrType
from torch.optim.optimizer import Optimizer

from .training_loop import TrainingLoop
from ..losses import CrossEntropyLoss
from ..models import Model
from ..sampling import NegativeSampler, negative_sampler_resolver
from ..triples import CoreTriplesFactory, Instances
from ..triples.instances import SLCWABatchType, SLCWASampleType
from ..typing import MappedTriples

__all__ = [
    'SLCWATrainingLoop',
]

logger = logging.getLogger(__name__)


class SLCWATrainingLoop(TrainingLoop[SLCWASampleType, SLCWABatchType]):
    """A training loop that uses the stochastic local closed world assumption training approach."""

    negative_sampler: NegativeSampler
    loss_blacklist = [CrossEntropyLoss]

    def __init__(
        self,
        model: Model,
        triples_factory: CoreTriplesFactory,
        optimizer: Optional[Optimizer] = None,
        negative_sampler: HintOrType[NegativeSampler] = None,
        negative_sampler_kwargs: Optional[Mapping[str, Any]] = None,
        automatic_memory_optimization: bool = True,
    ):
        """Initialize the training loop.

        :param model: The model to train
        :param triples_factory: The triples factory to train over
        :param optimizer: The optimizer to use while training the model
        :param negative_sampler: The class, instance, or name of the negative sampler
        :param negative_sampler_kwargs: Keyword arguments to pass to the negative sampler class on instantiation
            for every positive one
        :param automatic_memory_optimization:
            Whether to automatically optimize the sub-batch size during
            training and batch size during evaluation with regards to the hardware at hand.
        """
        super().__init__(
            model=model,
            triples_factory=triples_factory,
            optimizer=optimizer,
            automatic_memory_optimization=automatic_memory_optimization,
        )
        self.negative_sampler = negative_sampler_resolver.make(
            query=negative_sampler,
            pos_kwargs=negative_sampler_kwargs,
            triples_factory=triples_factory,
        )

    def _create_instances(self, triples_factory: CoreTriplesFactory) -> Instances:  # noqa: D102
        return triples_factory.create_slcwa_instances()

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

        # Create negative samples, shape: (batch_size, num_neg_per_pos, 3)
        negative_batch, positive_filter = self.negative_sampler.sample(positive_batch=positive_batch)

        # apply filter mask
        if positive_filter is None:
            negative_batch = negative_batch.view(-1, 3)
        else:
            negative_batch = negative_batch[positive_filter]

        # Ensure they reside on the device (should hold already for most simple negative samplers, e.g.
        # BasicNegativeSampler, BernoulliNegativeSampler
        negative_batch = negative_batch.to(self.device)

        # Compute negative and positive scores
        positive_scores = self.model.score_hrt(positive_batch)
        negative_scores = self.model.score_hrt(negative_batch).view(*negative_batch.shape[:-1])

        return self.loss.process_slcwa_scores(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            label_smoothing=label_smoothing,
            batch_filter=positive_filter,
            num_entities=self.model.num_entities,
        ) + self.model.collect_regularization_term()

    def _slice_size_search(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        training_instances: Instances,
        batch_size: int,
        sub_batch_size: int,
        supports_sub_batching: bool,
    ):  # noqa: D102
        # Slicing is not possible for sLCWA
        if supports_sub_batching:
            report = "This model supports sub-batching, but it also requires slicing, which is not possible for sLCWA"
        else:
            report = "This model doesn't support sub-batching and slicing is not possible for sLCWA"
        logger.warning(report)
        raise MemoryError("The current model can't be trained on this hardware with these parameters.")
