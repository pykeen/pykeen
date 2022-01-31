# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work of of Bordes *et al.*."""

from typing import Collection, Optional

import torch

from .negative_sampler import NegativeSampler
from ..constants import LABEL_HEAD, LABEL_TAIL, TARGET_TO_INDEX
from ..typing import Target

__all__ = [
    "BasicNegativeSampler",
]


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
        corruption_scheme: Optional[Collection[Target]] = None,
        **kwargs,
    ) -> None:
        """Initialize the basic negative sampler with the given entities.

        :param corruption_scheme:
            What sides ('h', 'r', 't') should be corrupted. Defaults to head and tail ('h', 't').
        :param kwargs:
            Additional keyword based arguments passed to :class:`pykeen.sampling.NegativeSampler`.
        """
        super().__init__(**kwargs)
        self.corruption_scheme = corruption_scheme or (LABEL_HEAD, LABEL_TAIL)
        # Set the indices
        self._corruption_indices = [TARGET_TO_INDEX[side] for side in self.corruption_scheme]
        self._n_corruptions = len(self._corruption_indices)

    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:  # noqa: D102
        # Copy positive batch for corruption.
        # Do not detach, as no gradients should flow into the indices.
        negative_batch = positive_batch.clone()
        negative_batch = negative_batch.unsqueeze(dim=-2).repeat(
            *(1 for _ in positive_batch.shape[:-1]), self.num_negs_per_pos, 1
        )

        corruption_index = torch.randint(self._n_corruptions, size=negative_batch.shape[:-1])
        # split_idx = int(math.ceil(num_negs / len(self._corruption_indices)))
        for index in self._corruption_indices:
            # Relations have a different index maximum than entities
            # At least make sure to not replace the triples by the original value
            index_max = (self.num_relations if index == 1 else self.num_entities) - 1

            mask = corruption_index == index
            # To make sure we don't replace the {head, relation, tail} by the
            # original value we shift all values greater or equal than the original value by one up
            # for that reason we choose the random value from [0, num_{heads, relations, tails} -1]
            negative_indices = torch.randint(
                high=index_max,
                size=(mask.sum().item(),),
                device=positive_batch.device,
            )

            # determine shift *before* writing the negative indices
            shift = (negative_indices >= negative_batch[mask][:, index]).long()
            negative_indices += shift

            # write the negative indices
            negative_batch[
                mask.unsqueeze(dim=-1) & (torch.arange(3) == index).view(*(1 for _ in negative_batch.shape[:-1]), 3)
            ] = negative_indices

        return negative_batch
