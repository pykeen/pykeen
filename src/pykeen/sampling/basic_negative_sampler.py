# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work of of Bordes *et al.*."""

from typing import Collection, Optional, Tuple

import torch

from .negative_sampler import NegativeSampler
from ..triples import TriplesFactory

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

    #: The default strategy for optimizing the negative sampler's hyper-parameters
    hpo_default = dict(
        num_negs_per_pos=dict(type=int, low=1, high=100, q=10),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        num_negs_per_pos: Optional[int] = None,
        filtered: bool = False,
        corruption_scheme: Optional[Collection[str]] = None,
    ) -> None:
        """Initialize the negative sampler with the given entities.

        :param triples_factory: The factory holding the triples to sample from
        :param num_negs_per_pos: Number of negative samples to make per positive triple. Defaults to 1.
        :param filtered: Whether proposed corrupted triples that are in the training data should be filtered.
            Defaults to False. See explanation in :func:`filter_negative_triples` for why this is
            a reasonable default.
        :param corruption_scheme: What sides ('h', 'r', 't') should be corrupted. Defaults to head and tail ('h', 't').
        """
        super().__init__(
            triples_factory=triples_factory,
            num_negs_per_pos=num_negs_per_pos,
            filtered=filtered,
        )
        self.corruption_scheme = corruption_scheme or ('h', 't')
        # Set the indices
        self._corruption_indices = [LOOKUP[side] for side in self.corruption_scheme]

    def sample(self, positive_batch: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:
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

        for index, start in zip(self._corruption_indices, range(0, num_negs, split_idx)):
            stop = min(start + split_idx, num_negs)

            # Relations have a different index maximum than entities
            index_max = self.num_relations - 1 if index == 1 else self.num_entities - 1

            negative_batch[start:stop, index] = torch.randint(
                high=index_max,
                size=(stop - start,),
                device=positive_batch.device,
            )

            # To make sure we don't replace the {head, relation, tail} by the
            # original value we shift all values greater or equal than the original value by one up
            # for that reason we choose the random value from [0, num_{heads, relations, tails} -1]
            if not self.filtered:
                negative_batch[start:stop, index] += (
                    negative_batch[start:stop, index] >= positive_batch[start:stop, index]
                ).long()

        # If filtering is activated, all negative triples that are positive in the training dataset will be removed
        if self.filtered:
            negative_batch, batch_filter = self.filter_negative_triples(negative_batch=negative_batch)
        else:
            batch_filter = None

        return negative_batch, batch_filter
