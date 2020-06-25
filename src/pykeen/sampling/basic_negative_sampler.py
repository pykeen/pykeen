# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work of of Bordes *et al.*."""

import torch

from .negative_sampler import NegativeSampler

__all__ = [
    'BasicNegativeSampler',
]


class BasicNegativeSampler(NegativeSampler):
    """A basic negative sampler."""

    #: The default strategy for optimizing the negative sampler's hyper-parameters
    hpo_default = dict(
        num_negs_per_pos=dict(type=int, low=1, high=100, q=10),
    )

    def sample(self, positive_batch: torch.LongTensor) -> torch.LongTensor:
        """Generate negative samples from the positive batch."""
        if self.num_negs_per_pos > 1:
            positive_batch = positive_batch.repeat(self.num_negs_per_pos, 1)

        # Bind number of negatives to sample
        num_negs = positive_batch.shape[0]

        # Equally corrupt head and tail
        split_idx = num_negs // 2

        # Copy positive batch for corruption.
        # Do not detach, as no gradients should flow into the indices.
        negative_batch = positive_batch.clone()

        # Sample random entities as replacement
        negative_entities = torch.randint(high=self.num_entities - 1, size=(num_negs,), device=positive_batch.device)

        # Replace heads – To make sure we don't replace the head by the original value
        # we shift all values greater or equal than the original value by one up
        # for that reason we choose the random value from [0, num_entities -1]
        filter_same_head = (negative_entities[:split_idx] >= positive_batch[:split_idx, 0])
        negative_batch[:split_idx, 0] = negative_entities[:split_idx] + filter_same_head.long()
        # Corrupt tails
        filter_same_tail = (negative_entities[split_idx:] >= positive_batch[split_idx:, 2])
        negative_batch[split_idx:, 2] = negative_entities[split_idx:] + filter_same_tail.long()

        return negative_batch
