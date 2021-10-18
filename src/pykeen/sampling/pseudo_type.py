# -*- coding: utf-8 -*-

"""Pseudo-Typed negative sampling."""

import itertools
import logging
from typing import Tuple

import torch

from .negative_sampler import NegativeSampler
from ..triples import CoreTriplesFactory
from ..triples.analysis import create_relation_to_entity_set_mapping

__all__ = [
    "PseudoTypedNegativeSampler",
]

logger = logging.getLogger(__name__)


def create_index(
    triples_factory: CoreTriplesFactory,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Create an index for efficient vectorized pseudo-type negative sampling.

    For this sampling, we need to store for each relation the set of head / tail entities. For efficient
    vectorized sampling, the following data structure is employed, which is partially inspired by the
    CSR format of sparse matrices (cf. :class:`scipy.sparse.csr_matrix`).

    We use two arrays, ``offsets`` and ``data``. The `offsets` array is of shape ``(2 * num_relations + 1,)``.
    The ``data`` array contains the sorted set of heads and tails for each relation, i.e.
    ``data[offsets[2*r]:offsets[2*r+1]]`` are the IDs of head entities for relation ``r``, and
    ``data[offsets[2*r+1]:offsets[2*r+2]]`` the ID of tail entities.

    :param triples_factory:
        The triples factory.

    :return:
        A pair (data, offsets) containing the compressed triples.
    """
    heads, tails = create_relation_to_entity_set_mapping(triples=triples_factory.mapped_triples.tolist())
    relations = set(heads.keys()).union(tails.keys())

    # TODO: move this warning to PseudoTypeNegativeSampler's constructor?
    for r in relations:
        if len(heads[r]) < 2 and len(tails[r]) < 2:
            logger.warning(f"Relation {r} does not have a sufficient number of distinct heads and tails.")

    # create index structure
    data = []
    offsets = torch.empty(2 * triples_factory.num_relations + 1, dtype=torch.long)
    offsets[0] = 0
    for i, (r, m) in enumerate(itertools.product(range(triples_factory.num_relations), (heads, tails)), start=1):
        data.extend(sorted(m[r]))
        offsets[i] = len(data)
    data = torch.as_tensor(data=data, dtype=torch.long)
    return data, offsets


class PseudoTypedNegativeSampler(NegativeSampler):
    r"""A sampler that accounts for which entities co-occur with a relation.

    To generate a corrupted head entity for triple $(h, r, t)$, only those entities are considered which occur as a
    head entity in a triple with the relation $r$.

    .. warning:: With this type of sampler, filtering for false negatives is more important.
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
        Instantiate the pseudo-typed negative sampler.

        :param triples_factory:
            The factory holding the positive training triples
        :param kwargs:
            Additional keyword based arguments passed to :class:`pykeen.sampling.NegativeSampler`.
        """
        super().__init__(triples_factory=triples_factory, **kwargs)
        self.data, self.offsets = create_index(triples_factory)

    def corrupt_batch(self, positive_batch: torch.LongTensor):  # noqa: D102
        batch_size = positive_batch.shape[0]

        # shape: (batch_size, num_neg_per_pos, 3)
        negative_batch = positive_batch.unsqueeze(dim=1).repeat(1, self.num_negs_per_pos, 1)

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

        # write into negative batch
        negative_batch[
            torch.arange(batch_size, device=negative_batch.device).unsqueeze(dim=-1),
            torch.arange(self.num_negs_per_pos, device=negative_batch.device).unsqueeze(dim=0),
            triple_position,
        ] = entity_id

        return negative_batch
