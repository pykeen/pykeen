# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work of [wang2014]_."""

from typing import Any, Mapping, Optional, Tuple

import torch
from class_resolver import HintOrType

from .filtering import Filterer
from .negative_sampler import NegativeSampler
from ..triples import TriplesFactory

__all__ = [
    'BernoulliNegativeSampler',
]


class BernoulliNegativeSampler(NegativeSampler):
    r"""An implementation of the Bernoulli negative sampling approach proposed by [wang2014]_.

    The probability of corrupting the head $h$ or tail $t$ in a relation $(h,r,t) \in \mathcal{K}$
    is determined by global properties of the relation $r$:

    - $r$ is *one-to-many* (e.g. *motherOf*): a higher probability is assigned to replace $h$
    - $r$ is *many-to-one* (e.g. *bornIn*): a higher probability is assigned to replace $t$.

    More precisely, for each relation $r \in \mathcal{R}$, the average number of tails per head
    (``tph``) and heads per tail (``hpt``) are first computed.

    Then, the head corruption probability $p_r$ is defined as $p_r = \frac{tph}{tph + hpt}$.
    The tail corruption probability is defined as $1 - p_r = \frac{hpt}{tph + hpt}$.

    For each triple $(h,r,t) \in \mathcal{K}$, the head is corrupted with probability $p_r$ and the tail is
    corrupted with probability $1 - p_r$.

    If ``filtered`` is set to ``True``, all proposed corrupted triples that also exist as
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
        filterer: HintOrType[Filterer] = None,
        filterer_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialize the bernoulli negative sampler with the given entities.

        :param triples_factory: The factory holding the triples to sample from
        :param num_negs_per_pos: Number of negative samples to make per positive triple. Defaults to 1.
        :param filtered: Whether proposed corrupted triples that are in the training data should be filtered.
            Defaults to False. See explanation in :func:`filter_negative_triples` for why this is
            a reasonable default.
        :param filterer: If filtered is set to True, this can be used to choose which filter module from
            :mod:`pykeen.sampling.filtering` is used.
        :param filterer_kwargs:
            Additional keyword-based arguments passed to the filterer upon construction.
        """
        super().__init__(
            triples_factory=triples_factory,
            num_negs_per_pos=num_negs_per_pos,
            filtered=filtered,
            filterer=filterer,
            filterer_kwargs=filterer_kwargs,
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

    def sample(self, positive_batch: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:
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

        # Randomly sample corruption. See below for explanation of
        # why this is on a range of [0, num_entities - 1]
        negative_entities = torch.randint(
            self.triples_factory.num_entities - 1,
            size=(num_negs,),
            device=positive_batch.device,
        )

        # Replace heads
        negative_batch[head_mask, 0] = negative_entities[head_mask]

        # Replace tails
        negative_batch[tail_mask, 2] = negative_entities[tail_mask]

        # If filtering is activated, all negative triples that are positive in the training dataset will be removed
        if self.filterer is not None:
            negative_batch, batch_filter = self.filterer(negative_batch=negative_batch)
        else:
            # To make sure we don't replace the head by the original value
            # we shift all values greater or equal than the original value by one up
            # for that reason we choose the random value from [0, num_entities -1]
            negative_batch[head_mask, 0] += (negative_batch[head_mask, 0] >= positive_batch[head_mask, 0]).long()
            negative_batch[tail_mask, 2] += (negative_batch[tail_mask, 2] >= positive_batch[tail_mask, 2]).long()
            batch_filter = None

        return negative_batch, batch_filter
