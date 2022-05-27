# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work of [wang2014]_."""

import torch

from .basic_negative_sampler import random_replacement_
from .negative_sampler import NegativeSampler
from ..typing import COLUMN_HEAD, COLUMN_TAIL, MappedTriples

__all__ = [
    "BernoulliNegativeSampler",
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

    def __init__(
        self,
        *,
        mapped_triples: MappedTriples,
        **kwargs,
    ) -> None:
        """Initialize the bernoulli negative sampler with the given entities.

        :param mapped_triples:
            the positive training triples
        :param kwargs:
            Additional keyword based arguments passed to :class:`pykeen.sampling.NegativeSampler`.
        """
        super().__init__(mapped_triples=mapped_triples, **kwargs)
        # Preprocessing: Compute corruption probabilities
        head_rel_uniq, tail_count = torch.unique(mapped_triples[:, :2], return_counts=True, dim=0)
        rel_tail_uniq, head_count = torch.unique(mapped_triples[:, 1:], return_counts=True, dim=0)

        self.corrupt_head_probability = torch.empty(
            self.num_relations,
            device=mapped_triples.device,
        )

        for r in range(self.num_relations):
            # compute tph, i.e. the average number of tail entities per head
            mask = head_rel_uniq[:, 1] == r
            tph = tail_count[mask].float().mean()

            # compute hpt, i.e. the average number of head entities per tail
            mask = rel_tail_uniq[:, 0] == r
            hpt = head_count[mask].float().mean()

            # Set parameter for Bernoulli distribution
            self.corrupt_head_probability[r] = tph / (tph + hpt)

    # docstr-coverage: inherited
    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:  # noqa: D102
        batch_shape = positive_batch.shape[:-1]

        # Decide whether to corrupt head or tail
        head_corruption_probability = self.corrupt_head_probability[positive_batch[..., 1]].unsqueeze(dim=-1)
        head_mask = torch.rand(
            *batch_shape, self.num_negs_per_pos, device=positive_batch.device
        ) < head_corruption_probability.to(device=positive_batch.device)

        # clone positive batch for corruption (.repeat_interleave creates a copy)
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(self.num_negs_per_pos, dim=0)
        # flatten mask
        head_mask = head_mask.view(-1)

        for index, mask in (
            (COLUMN_HEAD, head_mask),
            # Tails are corrupted if heads are not corrupted
            (COLUMN_TAIL, ~head_mask),
        ):
            random_replacement_(
                batch=negative_batch,
                index=index,
                selection=mask,
                size=mask.sum(),
                max_index=self.num_entities,
            )

        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)
