# -*- coding: utf-8 -*-

"""Pseudo-Typed negative sampling."""

import itertools
import logging

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

    To generate a corrupted head entity for triple $(h, r, t)$, only those entities are considered which occur as a
    head entity in a triple with the relation $r$.

    For this sampling, we need to store for each relation the set of head / tail entities. For efficient
    vectorized sampling, the following data structure is employed, which is partially inspired by the
    CSR format of sparse matrices (cf. :class:`scipy.sparse.csr_matrix`).

    We use two arrays, ``offsets`` and ``data``. The `offsets` array is of shape ``(2 * num_relations + 1,)``.
    The ``data`` array contains the sorted set of heads and tails for each relation, i.e.
    ``data[offsets[2*i]:offsets[2*i+1]]`` are the IDs of head entities for relation ``i``, and
    ``data[offsets[2*i+1]:offsets[2*i+2]]`` the ID of tail entities.
    """

    #: The array of offsets within the data array, shape: (2 * num_relations + 1,)
    offsets: torch.LongTensor

    #: The concatenated sorted sets of head/tail entities
    data: torch.LongTensor

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
        offsets = torch.empty(2 * self.num_relations + 1, dtype=torch.long)
        offsets[0] = 0
        for i, (r, m) in enumerate(itertools.product(range(self.num_relations), (heads, tails)), start=1):
            data.extend(sorted(m[r]))
            offsets[i] = len(data)
        self.data = torch.as_tensor(data=data, dtype=torch.long)
        self.offsets = offsets

    def corrupt_batch(self, positive_batch: torch.LongTensor):  # noqa: D102
        batch_size = positive_batch.shape[0]

        # shape: (neg, batch_size, 3)
        negative_batch = positive_batch.unsqueeze(dim=0).repeat(self.num_negs_per_pos, 1, 1)

        # Uniformly sample from head/tail offsets
        r = positive_batch[:, 1]
        start_heads = self.offsets[2 * r].unsqueeze(dim=-1)
        start_tails = self.offsets[2 * r + 1].unsqueeze(dim=-1)
        end = self.offsets[2 * r + 2].unsqueeze(dim=-1)
        num_choices = end - start_heads
        negative_ids = start_heads + (torch.rand(size=(batch_size, self.num_negs_per_pos)) * num_choices).long()

        # get corresponding entity
        entity_id = self.data[negative_ids]

        # and position within triple (0: head, 2: tail)
        triple_position = 2 * (negative_ids >= start_tails).long()

        # fallback heuristic: random
        fill_mask = torch.arange(self.num_negs_per_pos).unsqueeze(dim=0) >= num_choices
        entity_id[fill_mask] = torch.randint(self.num_entities, size=(fill_mask.sum(),), device=negative_batch.device)

        # write into negative batch
        negative_batch[
            torch.arange(self.num_negs_per_pos, device=negative_batch.device).unsqueeze(dim=0),
            torch.arange(batch_size, device=negative_batch.device).unsqueeze(dim=-1),
            triple_position,
        ] = entity_id

        return negative_batch.view(-1, self.num_negs_per_pos, 3)
