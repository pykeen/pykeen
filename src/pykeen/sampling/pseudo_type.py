# -*- coding: utf-8 -*-

"""Pseudo-Typed negative sampling."""

import logging
from typing import Optional, Tuple

import torch

from .negative_sampler import NegativeSampler
from ..triples import CoreTriplesFactory
from ..triples.analysis import create_relation_to_entity_set_mapping

__all__ = [
    "PseudoTypedNegativeSampler",
]

logger = logging.getLogger(__name__)


class PseudoTypedNegativeSampler(NegativeSampler):
    r"""
    A negative sampler using pseudo-types.

    To generate a corrupted head entity for triple (h, r, t), only those entities are considered which occur as a
    head entity in a triple with the relation r.

    Data Structure
    --------------

    heads:
        (r_1, t_1) -> {h_1^1, \ldots, h_{k_1}^1}
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
        heads, tails = create_relation_to_entity_set_mapping(triples=triples_factory.mapped_triples.tolist())

        relations = set(heads.keys()).union(tails.keys())
        for r in relations:
            if len(heads[r]) < 2 and len(tails[r]) < 2:
                logger.warning(f"Relation {r} does not have a sufficient number of distinct heads and tails.")

        # create index structure
        data = []
        offset = 0
        offsets = [offset]
        for r in range(self.num_relations):
            for m in (heads, tails):
                data.extend(sorted(m[r]))
                offset = len(data)
                offsets.append(offset)
        self.data = torch.as_tensor(data=data, dtype=torch.long)
        self.offsets = torch.as_tensor(data=offsets, dtype=torch.long)

    def sample(self, positive_batch: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:  # noqa: D102
        batch_size = positive_batch.shape[0]
        # shape: (batch_size, neg, 3)
        negative_batch = positive_batch.unsqueeze(dim=1).repeat(1, self.num_negs_per_pos, 1)
        r = positive_batch[:, 1]
        start_heads = self.offsets[2 * r].unsqueeze(dim=-1)
        start_tails = self.offsets[2 * r + 1].unsqueeze(dim=-1)
        end = self.offsets[2 * r + 2].unsqueeze(dim=-1)
        num_choices = end - start_heads
        negative_ids = start_heads + (torch.rand(size=(batch_size, self.num_negs_per_pos)) * num_choices).long()
        entity_id = self.data[negative_ids]
        triple_position = 2 * (negative_ids >= start_tails).long()
        # fallback heuristic: random
        fill_mask = torch.arange(self.num_negs_per_pos).unsqueeze(dim=0) >= num_choices
        entity_id[fill_mask] = torch.randint(self.num_entities, size=(fill_mask.sum(),), device=negative_batch.device)
        negative_batch[
            torch.arange(batch_size, device=negative_batch.device).unsqueeze(dim=-1),
            torch.arange(self.num_negs_per_pos, device=negative_batch.device).unsqueeze(dim=0),
            triple_position,
        ] = entity_id

        # TODO: Filtering
        return negative_batch.view(-1, 3), None
