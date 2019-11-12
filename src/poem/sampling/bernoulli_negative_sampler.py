# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work of [wang2014]_."""

import torch

from .negative_sampler import NegativeSampler
from ..triples import TriplesFactory

__all__ = [
    'BernoulliNegativeSampler',
]


class BernoulliNegativeSampler(NegativeSampler):
    """An implementation of the bernoulli negative sampling approach proposed by [wang2014]_."""

    def __init__(self, triples_factory: TriplesFactory) -> None:
        super().__init__(triples_factory=triples_factory)

        # Preprocessing: Compute corruption probabilities
        triples = torch.tensor(self.triples_factory.mapped_triples)
        head_rel_uniq, tail_count = torch.unique(triples[:, :2], return_counts=True, dim=0)
        rel_tail_uniq, head_count = torch.unique(triples[:, 1:], return_counts=True, dim=0)

        self.corrupt_head_probability = torch.empty(self.triples_factory.num_relations)
        for r in range(self.triples_factory.num_relations):
            # compute tph, i.e. the average number of tail entities per head
            mask = (head_rel_uniq[:, 1] == r)
            tph = tail_count[mask].float().mean()

            # compute hpt, i.e. the average number of head entities per tail
            mask = (rel_tail_uniq[:, 0] == r)
            hpt = head_count[mask].float().mean()

            # Set parameter for Bernoulli distribution
            self.corrupt_head_probability[r] = tph / (tph + hpt)

    def sample(self, positive_batch: torch.LongTensor) -> torch.LongTensor:
        """Sample a negative batched based on the bern approach."""
        # Bind batch size
        batch_size = positive_batch.shape[0]

        # Copy positive batch for corruption. Do not detach, as no gradients should flow into the indices.
        negative_batch = positive_batch.clone()

        device = positive_batch.device
        # Decide whether to corrupt head or tail
        head_corruption_probability = self.corrupt_head_probability[positive_batch[:, 1]]
        head_mask = torch.rand(batch_size, device=device) < head_corruption_probability.to(device=device)

        # Tails are corrupted if heads are not corrupted
        tail_mask = ~head_mask

        # Randomly sample corruption
        corrupt_entity = torch.randint(
            self.triples_factory.num_entities,
            size=(batch_size,),
            device=positive_batch.device,
        )

        # Replace heads
        negative_batch[:, 0][head_mask] = corrupt_entity[head_mask]

        # Replace tails
        negative_batch[:, 2][tail_mask] = corrupt_entity[tail_mask]

        return negative_batch
