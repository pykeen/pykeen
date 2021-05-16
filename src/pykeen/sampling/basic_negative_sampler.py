# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work of of Bordes *et al.*."""

import math
from typing import Collection, Optional

import torch

from .negative_sampler import NegativeSampler

__all__ = [
    'BasicNegativeSampler',
]

LOOKUP = {'h': 0, 'r': 1, 't': 2}


class BasicNegativeSampler(NegativeSampler):
    r"""A basic negative sampler.

    This negative sampler that corrupts positive triples $(h,r,t) \in \mathcal{K}$ by replacing either $h$, $r$ or $t$
    based on the chosen corruption scheme. The corruption scheme can contain $h$, $r$ and $t$ or any subset of these.

    Steps:

    1. Randomly (uniformly) determine whether $h$, $r$ or $t$ shall be corrupted for a positive triple
       $(h,r,t) \in \mathcal{K}$.
    2. Randomly (uniformly) sample an entity $e \in \mathcal{E}$ or relation $r' \in \mathcal{R}$ for selection to
       corrupt the triple.

       - If $h$ was selected before, the corrupted triple is $(e,r,t)$
       - If $r$ was selected before, the corrupted triple is $(h,r',t)$
       - If $t$ was selected before, the corrupted triple is $(h,r,e)$
    3. If ``filtered`` is set to ``True``, all proposed corrupted triples that also exist as
       actual positive triples $(h,r,t) \in \mathcal{K}$ will be removed.
    """

    def __init__(
        self,
        *,
        corruption_scheme: Optional[Collection[str]] = None,
        **kwargs,
    ) -> None:
        """Initialize the basic negative sampler with the given entities.

        :param corruption_scheme:
            What sides ('h', 'r', 't') should be corrupted. Defaults to head and tail ('h', 't').
        :param kwargs:
            Additional keyword based arguments passed to :class:`pykeen.sampling.NegativeSampler`.
        """
        super().__init__(**kwargs)
        self.corruption_scheme = corruption_scheme or ('h', 't')
        # Set the indices
        self._corruption_indices = [LOOKUP[side] for side in self.corruption_scheme]

    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:  # noqa: D102
        if self.num_negs_per_pos > 1:
            positive_batch = positive_batch.repeat_interleave(repeats=self.num_negs_per_pos, dim=0)

        # Bind number of negatives to sample
        num_negs = positive_batch.shape[0]

        # Equally corrupt all sides
        split_idx = int(math.ceil(num_negs / len(self._corruption_indices)))

        # Copy positive batch for corruption.
        # Do not detach, as no gradients should flow into the indices.
        negative_batch = positive_batch.clone()

        for index, start in zip(self._corruption_indices, range(0, num_negs, split_idx)):
            stop = min(start + split_idx, num_negs)

            # Relations have a different index maximum than entities
            # At least make sure to not replace the triples by the original value
            index_max = (self.num_relations if index == 1 else self.num_entities) - 1

            negative_batch[start:stop, index] = torch.randint(
                high=index_max,
                size=(stop - start,),
                device=positive_batch.device,
            )

            # To make sure we don't replace the {head, relation, tail} by the
            # original value we shift all values greater or equal than the original value by one up
            # for that reason we choose the random value from [0, num_{heads, relations, tails} -1]
            negative_batch[start:stop, index] += (
                negative_batch[start:stop, index] >= positive_batch[start:stop, index]
            ).long()

        return negative_batch.view(-1, self.num_negs_per_pos, 3)
