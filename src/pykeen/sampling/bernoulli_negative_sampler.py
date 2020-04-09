# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work of [wang2014]_."""

from typing import Optional

import torch

from .negative_sampler import NegativeSampler
from ..triples import TriplesFactory

__all__ = [
    'BernoulliNegativeSampler',
]


class BernoulliNegativeSampler(NegativeSampler):
    """An implementation of the bernoulli negative sampling approach proposed by [wang2014]_."""

    #: The default strategy for optimizing the negative sampler's hyper-parameters
    hpo_default = dict(
        num_negs_per_pos=dict(type=int, low=1, high=100, q=10),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        num_negs_per_pos: Optional[int] = None,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            num_negs_per_pos=num_negs_per_pos,
        )
        # Preprocessing: Compute corruption probabilities
        triples = self.triples_factory.mapped_triples
        head_rel_uniq, tail_count = torch.unique(triples[:, :2], return_counts=True, dim=0)
        rel_tail_uniq, head_count = torch.unique(triples[:, 1:], return_counts=True, dim=0)

        self.corrupt_head_probability = torch.empty(
            self.triples_factory.num_relations,
            device=triples_factory.mapped_triples.device,
        )

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
        if self.num_negs_per_pos > 1:
            positive_batch = positive_batch.repeat(self.num_negs_per_pos, 1)

        # Bind number of negatives to sample
        num_negs = positive_batch.shape[0]

        # Copy positive batch for corruption.
        # Do not detach, as no gradients should flow into the indices.
        negative_batch = positive_batch.clone()

        device = positive_batch.device
        # Decide whether to corrupt head or tail
        head_corruption_probability = self.corrupt_head_probability[positive_batch[:, 1]]
        head_mask = torch.rand(num_negs, device=device) < head_corruption_probability.to(device=device)

        # Tails are corrupted if heads are not corrupted
        tail_mask = ~head_mask

        # Randomly sample corruption
        negative_entities = torch.randint(
            self.triples_factory.num_entities - 1,
            size=(num_negs,),
            device=positive_batch.device,
        )

        # Replace heads – To make sure we don't replace the head by the original value
        # we shift all values greater or equal than the original value by one up
        # for that reason we choose the random value from [0, num_entities -1]
        filter_same_head = (negative_entities[head_mask] >= positive_batch[:, 0][head_mask])
        negative_batch[:, 0][head_mask] = negative_entities[head_mask] + filter_same_head.long()

        # Replace tails
        filter_same_tail = (negative_entities[tail_mask] >= positive_batch[:, 2][tail_mask])
        negative_batch[:, 2][tail_mask] = negative_entities[tail_mask] + filter_same_tail.long()

        return negative_batch
