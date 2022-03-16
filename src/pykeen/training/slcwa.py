# -*- coding: utf-8 -*-

"""Training KGE models based on the sLCWA."""

import logging
from typing import Optional

import torch.utils.data
from class_resolver import HintOrType, OptionalKwargs
from torch.utils.data import DataLoader

from .training_loop import TrainingLoop
from ..sampling import NegativeSampler
from ..triples import CoreTriplesFactory
from ..triples.instances import SLCWABatch, SLCWASampleType

__all__ = [
    "SLCWATrainingLoop",
]

logger = logging.getLogger(__name__)


class SLCWATrainingLoop(TrainingLoop[SLCWASampleType, SLCWABatch]):
    """A training loop that uses the stochastic local closed world assumption training approach.

    [ruffinelli2020]_ call the sLCWA ``NegSamp`` in their work.
    """

    def __init__(
        self,
        negative_sampler: HintOrType[NegativeSampler] = None,
        negative_sampler_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        """Initialize the training loop.

        :param negative_sampler: The class, instance, or name of the negative sampler
        :param negative_sampler_kwargs: Keyword arguments to pass to the negative sampler class on instantiation
            for every positive one
        :param kwargs:
            Additional keyword-based parameters passed to TrainingLoop.__init__
        """
        super().__init__(**kwargs)
        self.negative_sampler = negative_sampler
        self.negative_sampler_kwargs = negative_sampler_kwargs

    def _create_training_data_loader(
        self,
        triples_factory: CoreTriplesFactory,
        batch_size: int,
        drop_last: bool,
        num_workers: int,
        pin_memory: bool,
        sampler: Optional[str],
    ) -> DataLoader[SLCWABatch]:  # noqa: D102
        return DataLoader(
            dataset=triples_factory.create_slcwa_instances(
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last,
                negative_sampler=self.negative_sampler,
                negative_sampler_kwargs=self.negative_sampler_kwargs,
                sampler=sampler,
            ),
            num_workers=num_workers,
            pin_memory=pin_memory,
            # disable automatic batching
            batch_size=None,
            batch_sampler=None,
        )

    @staticmethod
    def _get_batch_size(batch: SLCWABatch) -> int:  # noqa: D102
        return batch[0].shape[0]

    def _process_batch(
        self,
        batch: SLCWABatch,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # Slicing is not possible in sLCWA training loops
        if slice_size is not None:
            raise AttributeError("Slicing is not possible for sLCWA training loops.")

        # split batch
        positive_batch, negative_batch, positive_filter = batch

        # send to device
        positive_batch = positive_batch[start:stop].to(device=self.device)
        negative_batch = negative_batch[start:stop]
        if positive_filter is not None:
            positive_filter = positive_filter[start:stop]
            negative_batch = negative_batch[positive_filter]
        # Make it negative batch broadcastable (required for num_negs_per_pos > 1).
        negative_score_shape = negative_batch.shape[:-1]
        negative_batch = negative_batch.view(-1, 3)

        # Ensure they reside on the device (should hold already for most simple negative samplers, e.g.
        # BasicNegativeSampler, BernoulliNegativeSampler
        negative_batch = negative_batch.to(self.device)

        # Compute negative and positive scores
        positive_scores = self.model.score_hrt(positive_batch, mode=self.mode)
        negative_scores = self.model.score_hrt(negative_batch, mode=self.mode).view(*negative_score_shape)

        return (
            self.loss.process_slcwa_scores(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                label_smoothing=label_smoothing,
                batch_filter=positive_filter,
                num_entities=self.model._get_entity_len(mode=self.mode),
            )
            + self.model.collect_regularization_term()
        )

    def _slice_size_search(
        self,
        *,
        triples_factory: CoreTriplesFactory,
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
