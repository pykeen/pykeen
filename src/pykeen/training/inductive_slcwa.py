# -*- coding: utf-8 -*-

"""Training inductive KGE models based on the sLCWA."""

import logging
from typing import Any, Mapping, Optional

import torch
from class_resolver import HintOrType

from .training_loop import TrainingLoop
from ..sampling import NegativeSampler, negative_sampler_resolver
from ..triples import CoreTriplesFactory, Instances
from ..triples.instances import SLCWABatchType, SLCWASampleType
from ..typing import MappedTriples
from .slcwa import SLCWATrainingLoop

__all__ = [
    "InductiveSLCWATrainingLoop",
]

logger = logging.getLogger(__name__)

class InductiveSLCWATrainingLoop(SLCWATrainingLoop):
    """A training loop that uses the stochastic local closed world assumption training approach.

    The main difference to transductive SLCWA is
     - explicit setting of the mode=train argument
     - num entities is set to num_training_entities
    """

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
            raise AttributeError("Slicing is not possible for sLCWA training loops.")

        # Send positive batch to device
        positive_batch = batch[start:stop].to(device=self.device)

        # Create negative samples, shape: (batch_size, num_neg_per_pos, 3)
        negative_batch, positive_filter = self.negative_sampler.sample(positive_batch=positive_batch)

        # apply filter mask
        if positive_filter is None:
            negative_score_shape = negative_batch.shape[:2]
            negative_batch = negative_batch.view(-1, 3)
        else:
            negative_batch = negative_batch[positive_filter]
            negative_score_shape = negative_batch.shape[:-1]

        # Ensure they reside on the device (should hold already for most simple negative samplers, e.g.
        # BasicNegativeSampler, BernoulliNegativeSampler
        negative_batch = negative_batch.to(self.device)

        # Compute negative and positive scores
        positive_scores = self.model.score_hrt(positive_batch, mode="train")
        negative_scores = self.model.score_hrt(negative_batch, mode="train").view(*negative_score_shape)

        return (
            self.loss.process_slcwa_scores(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                label_smoothing=label_smoothing,
                batch_filter=positive_filter,
                num_entities=self.model.num_training_entities,
            )
            + self.model.collect_regularization_term()
        )