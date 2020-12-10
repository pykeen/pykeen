# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work of of Bordes *et al.*."""

import torch

from .negative_sampler import NegativeSampler

__all__ = [
    'BasicNegativeSampler',
]


class BasicNegativeSampler(NegativeSampler):
    r"""A basic negative sampler.

    This negative sampler that corrupts positive triples $(h,r,t) \in \mathcal{K}$ by replacing either $h$ or $t$.

    Steps:

    1. Randomly (uniformly) determine whether $h$ or $t$ shall be corrupted for a positive triple
       $(h,r,t) \in \mathcal{K}$.
    2. Randomly (uniformly) sample an entity $e \in \mathcal{E}$ for selection to corrupt the triple.

       - If $h$ was selected before, the corrupted triple is $(e,r,t)$
       - If $t$ was selected before, the corrupted triple is $(h,r,e)$
    """

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

        # Equally corrupt all sides
        split_idx = num_negs // len(self._corruption_indices)

        # Copy positive batch for corruption.
        # Do not detach, as no gradients should flow into the indices.
        negative_batch = positive_batch.clone()

        # Sample random entities as replacement
        if 0 in self._corruption_indices or 2 in self._corruption_indices:
            negative_entities = torch.randint(
                high=self.num_entities - 1,
                size=(num_negs,),
                device=positive_batch.device,
            )

        # Sample random relations as replacement, if requested
        if 1 in self._corruption_indices:
            negative_relations = torch.randint(
                high=self.num_relations - 1,
                size=(num_negs,),
                device=positive_batch.device,
            )

        for index, start in zip(self._corruption_indices, range(0, num_negs, split_idx)):
            stop = min(start + split_idx, num_negs)
            if index == 1:
                # Corrupt relations
                negative_batch[start:stop, index] = negative_relations[start:stop]
            else:
                # Corrupt heads or tails
                negative_batch[start:stop, index] = negative_entities[start:stop]

            # Replace {heads, relations, tails} – To make sure we don't replace the {head, relation, tail} by the
            # original value we shift all values greater or equal than the original value by one up
            # for that reason we choose the random value from [0, num_{heads, relations, tails} -1]
            if not self.filtered:
                negative_batch[start:stop, index] += (
                        negative_batch[start:stop, index] >= positive_batch[start:stop, index]
                ).long()

        # If filtering is activated, all negative triples that are positive in the training dataset will be removed
        if self.filtered:
            negative_batch = self._filter_negative_triples(negative_batch=negative_batch)

        return negative_batch
