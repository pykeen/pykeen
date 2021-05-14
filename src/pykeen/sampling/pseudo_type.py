# -*- coding: utf-8 -*-

"""Pseudo-Typed negative sampling."""

import logging
import random
from itertools import product, starmap
from typing import List, Optional, Tuple, cast

import torch

from .negative_sampler import NegativeSampler
from ..triples import CoreTriplesFactory
from ..triples.analysis import create_relation_to_entity_set_mapping

__all__ = [
    "PseudoTypedNegativeSampler",
]

logger = logging.getLogger(__name__)


class PseudoTypedNegativeSampler(NegativeSampler):
    """
    A negative sampler using pseudo-types.

    To generate a corrupted head entity for triple (h, r, t), only those entities are considered which occur as a
    head entity in a triple with the relation r.
    """

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        **kwargs,
    ):
        """
        Instantiate the negative sampler.

        :param triples_factory:
            The factory holding the positive training triples
        :param kwargs:
            Additional keyword based arguments passed to NegativeSampler.
        """
        super().__init__(triples_factory=triples_factory, **kwargs)
        self.heads, self.tails = create_relation_to_entity_set_mapping(triples=triples_factory.mapped_triples.tolist())
        for r in set(self.heads.keys()).union(self.tails.keys()):
            if len(self.heads[r]) < 2 and len(self.tails[r]) < 2:
                logger.warning(f"Relation {r} does not have a sufficient number of distinct heads and tails.")

    def sample(self, positive_batch: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:  # noqa: D102
        # shape: (batch_size, neg, 3)
        negative_batch = positive_batch.unsqueeze(dim=1).repeat(1, self.num_negs_per_pos, 1)

        # TODO: Can we vectorize this? .tolist is an expensive operation which requires synchronization
        chosens = starmap(self._sample_helper, positive_batch.tolist())
        chosens = list(chosens)
        c_t = torch.as_tensor(chosens)
        k, e = c_t[..., 0], c_t[..., 1]
        negative_batch = torch.scatter(negative_batch, dim=-1, index=k, src=e)

        # TODO: Filtering
        return negative_batch.view(-1, 3), None

    def _sample_helper(self, h: int, r: int, t: int) -> List[Tuple[int, int]]:
        candidates: List[Tuple[int, int]] = [
            (position, candidate)
            for position, relation_to_candidates, current_entity in (
                (0, self.heads, h),
                (2, self.tails, t),
            )
            for candidate in relation_to_candidates[r].difference({current_entity})
        ]
        k = min(len(candidates), self.num_negs_per_pos)
        chosen: List[Tuple[int, int]] = random.sample(candidates, k=k)
        # fallback heuristic: random
        k = self.num_negs_per_pos - len(chosen)
        chosen.extend(cast(List[Tuple[int, int]], random.choices(
            list(product((0, 2), range(self.num_entities))),  # cross product of positions/candidates
            k=k,
        )))
        return chosen
