# -*- coding: utf-8 -*-

"""Sampling methods for temproal KGs."""

import torch
from pykeen.sampling import BasicNegativeSampler


class TemporalBasicNegativeSampler(BasicNegativeSampler):
    """A basic temporal negative sampler from BasicNegativeSampler with rewritten corrupt_batch()."""

    def __init__(self, num_negs_per_pos, **kwargs):
        """Initialize."""
        super().__init__(num_negs_per_pos=num_negs_per_pos, **kwargs)

    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:  # noqa: D102
        """Corrupt a batch of positive triples."""
        # positive_batch size [1,4]
        # batch_shape is batch length and now it is 1
        batch_shape = positive_batch.shape[:-1]
        # print(f"positive batch shape: {positive_batch.shape}")
        # print(f"positive_batch: {positive_batch}")
        # Copy positive batch for corruption.
        # Do not detach, as no gradients should flow into the indices.
        negative_batch = positive_batch.clone()
        # 1 6 4 where 6 is the hyperparameter, generate 6 negative for one positive, 3 head 3 column
        negative_batch = negative_batch.unsqueeze(dim=-2).repeat(
            *(1 for _ in batch_shape), self.num_negs_per_pos, 1
        )
        # print(f"negative batch shape: {negative_batch.shape}")
        # print(f"negative_batch: {negative_batch}")
        # print(f"corruption indices: {self._corruption_indices}")

        # a half of self.num_negs_per_pos is head; the other half is tail
        corruption_index = torch.zeros(size=(*batch_shape, self.num_negs_per_pos))
        corruption_index[0, : len(corruption_index[0]) // 2] = 2
        # for corroption targets, self._corruption_indices is [0,2], the first column and the third column
        for index in self._corruption_indices:
            # Relations have a different index maximum than entities
            # At least make sure to not replace the triples by the original value
            index_max = (self.num_relations if index == 1 else self.num_entities) - 1

            # to see current index i is the corruption target
            # mask is [[False]] if not
            mask = corruption_index == index
            # print(f"mask {mask}; corruption_index: {corruption_index}")

            # To make sure we don't replace the {head, relation, tail} by the
            # original value we shift all values greater or equal than the original value by one up
            # for that reason we choose the random value from [0, num_{heads, relations, tails} -1]
            negative_indices = torch.randint(
                high=index_max,
                size=(mask.sum().item(),),
                device=positive_batch.device,
            )
            # print(f"negative indices : {negative_indices}")

            # determine shift *before* writing the negative indices
            shift = (negative_indices >= negative_batch[mask][:, index]).long()
            negative_indices += shift
            # print(f"negative indices : {negative_indices}")

            # write the negative indices
            negative_batch[
                mask.unsqueeze(dim=-1)
                & (torch.arange(4) == index).view(*(1 for _ in batch_shape), 1, 4)
            ] = negative_indices

        return negative_batch
