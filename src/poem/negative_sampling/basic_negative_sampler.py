# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work of of Bordes *et al.*."""

import torch

from .base import NegativeSampler

__all__ = [
    'BasicNegativeSampler',
]


class BasicNegativeSampler(NegativeSampler):
    """A basic negative sampler."""

    def sample(self, positive_batch: torch.LongTensor) -> torch.LongTensor:
        """Generate negative samples from the positive batch."""
        # Bind batch size
        batch_size = positive_batch.shape[0]

        # Equally corrupt subject and object
        split_idx = batch_size // 2

        # Copy positive batch for corruption. Do not detach, as no gradients should flow into the indices.
        negative_batch = positive_batch.detach().clone()

        # Corrupt subjects
        negative_batch[:split_idx, 0] = torch.randint(high=self.num_entities, size=(split_idx,))
        # Corrupt objects
        negative_batch[split_idx:, 2] = torch.randint(high=self.num_entities, size=(batch_size - split_idx,))

        return negative_batch
