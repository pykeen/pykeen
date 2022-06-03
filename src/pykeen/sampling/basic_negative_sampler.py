# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work of of Bordes *et al.*."""

import math
from typing import Collection, Optional

import torch

from .negative_sampler import NegativeSampler
from ..constants import LABEL_HEAD, LABEL_TAIL, TARGET_TO_INDEX
from ..typing import Target

__all__ = [
    "BasicNegativeSampler",
    "random_replacement_",
]


def random_replacement_(batch: torch.LongTensor, index: int, selection: slice, size: int, max_index: int) -> None:
    """
    Replace a column of a batch of indices by random indices.

    :param batch: shape: `(*batch_dims, d)`
        the batch of indices
    :param index:
        the index (of the last axis) which to replace
    :param selection:
        a selection of the batch, e.g., a slice or a mask
    :param size:
        the size of the selection
    :param max_index:
        the maximum index value at the chosen position
    """
    # At least make sure to not replace the triples by the original value
    # To make sure we don't replace the {head, relation, tail} by the
    # original value we shift all values greater or equal than the original value by one up
    # for that reason we choose the random value from [0, num_{heads, relations, tails} -1]
    replacement = torch.randint(
        high=max_index - 1,
        size=(size,),
        device=batch.device,
    )
    replacement += (replacement >= batch[selection, index]).long()
    batch[selection, index] = replacement


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

    # docstr-coverage: inherited
    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:  # noqa: D102
        batch_shape = positive_batch.shape[:-1]

        # clone positive batch for corruption (.repeat_interleave creates a copy)
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(self.num_negs_per_pos, dim=0)

        # Bind the total number of negatives to sample in this batch
        total_num_negatives = negative_batch.shape[0]

        # Equally corrupt all sides
        split_idx = int(math.ceil(total_num_negatives / len(self._corruption_indices)))

        # Do not detach, as no gradients should flow into the indices.
        for index, start in zip(self._corruption_indices, range(0, total_num_negatives, split_idx)):
            stop = min(start + split_idx, total_num_negatives)
            random_replacement_(
                batch=negative_batch,
                index=index,
                selection=slice(start, stop),
                size=stop - start,
                max_index=self.num_relations if index == 1 else self.num_entities,
            )

        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)
