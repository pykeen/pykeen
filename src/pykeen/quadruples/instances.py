# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG quadruples."""

import logging
from typing import Iterable, Optional, Tuple

import numpy as np
import scipy
import torch
from class_resolver import HintOrType, OptionalKwargs
from pykeen.sampling import NegativeSampler, negative_sampler_resolver
from torch.utils import data

from ..typing import MappedQuadruples

__all__ = [
    "QuadrupleInstances",
    "SLCWAQuadrupleInstances",
    "LCWAQuadrupleInstances",
]

SLCWATemporalSampleType = Tuple[MappedQuadruples, MappedQuadruples, Optional[torch.BoolTensor]]
SLCWATemporalBatchType = Tuple[MappedQuadruples, MappedQuadruples, Optional[torch.BoolTensor]]
LCWATemporalBatchType = Tuple[MappedQuadruples, MappedQuadruples, Optional[torch.BoolTensor]]

logger = logging.getLogger(__name__)


class QuadrupleInstances(data.Dataset):
    """A Wrapper of torch Dataset for quadruples."""

    def __len__(self):
        """Return the length of dataset."""
        raise NotImplementedError

    def __getitem__(self, item):
        """Return an item given index."""
        raise NotImplementedError

    @classmethod
    def from_quadruples(
        cls,
        mapped_quadruples: MappedQuadruples,
        *,
        num_entities: int,
        num_relations: int,
        num_timestamps: int,
        **kwargs,
    ) -> "QuadrupleInstances":
        """
        Create instances from mapped quadruples.

        :param mapped_quadruples: shape: (num_quadruples, 4)
            The ID-based quadruples
        :param num_entities: >0
            The number of entities
        :param num_relations: >0
            The number of relations
        :param num_timestamps: >0
            The number of timestamps
        :param kwargs:
            Additional keyword-based parameters

        :return:
            The instances.

        # noqa:DAR202
        # noqa:DAR401
        """
        raise NotImplementedError


class SLCWAQuadrupleInstances(QuadrupleInstances):
    """SLCWA Quadruple Instance Dataset."""

    def __init__(
        self,
        *,
        mapped_quadruples: MappedQuadruples,
        num_entities: Optional[int] = None,
        num_relations: Optional[int] = None,
        negative_sampler: HintOrType[NegativeSampler] = None,
        negative_sampler_kwargs: OptionalKwargs = None,
    ):
        """
        Initialize Instances.

        :param mapped_quadruples: shape: (num_quadruples, 4)
            The ID-based quadruples, passed to the negative sampler
        :param num_entities: >0
            The number of entities, passed to the negative sampler
        :param num_relations: >0
            The number of relations, passed to the negative sampler
        :param negative_sampler:
            The negative sampler, or a hint thereof
        :param negative_sampler_kwargs:
            Additional keyword-based arguments passed to the negative sampler

        """
        self.mapped_quadruples = mapped_quadruples
        self.sampler = negative_sampler_resolver.make(
            negative_sampler,
            pos_kwargs=negative_sampler_kwargs,
            mapped_triples=mapped_quadruples,
            num_entities=num_entities,
            num_relations=num_relations,
        )

    def __len__(self) -> int:
        """Return the length of quadruples."""
        return self.mapped_quadruples.shape[0]

    def __getitem__(self, item: int) -> SLCWATemporalSampleType:
        """Rewrite __getitem__."""
        positive = self.mapped_quadruples[item].unsqueeze(dim=0)
        negative, mask = self.sampler.sample(positive_batch=positive)
        return positive, negative, mask

    @staticmethod
    def collate(samples: Iterable[SLCWATemporalSampleType]) -> SLCWATemporalBatchType:
        """Collator functions for torch.Dataset; same as the SLCWAInstances."""
        positives, negatives, masks = zip(*samples)
        positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)
        if masks[0] is None:
            assert all(m is None for m in masks)
            masks = None
        else:
            masks = torch.cat(masks, dim=0)
        return positives, negatives, masks

    def get_collator(
        self,
    ):
        """Return the collate function."""
        return self.collate

    @classmethod
    def from_quadruples(
        cls,
        mapped_quadruples: MappedQuadruples,
        *,
        num_entities: int,
        num_relations: int,
        **kwargs,
    ) -> QuadrupleInstances:
        """Build instances datasets."""
        return cls(
            mapped_quadruples=mapped_quadruples,
            num_entities=num_entities,
            num_relations=num_relations,
            **kwargs,
        )


class LCWAQuadrupleInstances(QuadrupleInstances):
    """Triples and mappings to their indices for LCWA."""

    def __init__(self, *, pairs: np.ndarray, compressed: scipy.sparse.csr_matrix):
        """Initialize the LCWA instances.

        :param pairs: The unique pairs
        :param compressed: The compressed triples in CSR format
        """
        self.pairs = pairs
        self.compressed = compressed

    @classmethod
    def from_quadruples(
        cls,
        mapped_quadruples: MappedQuadruples,
        *,
        num_entities: int,
        num_relations: int,
        num_timestamps: int,
        target: Optional[int] = None,
        **kwargs,
    ) -> QuadrupleInstances:
        """
        Create LCWA instances from quadruples.

        :param mapped_quadruples: shape: (num_quadruples, 4)
            The ID-based quadruples.
        :param num_entities:
            The number of entities.
        :param num_relations:
            The number of relations.
        :param num_timestamps:
            The number of timestamps.
        :param target:
            The column to predict.
        :param kwargs:
            Keyword arguments (thrown out).

        :return:
            The instances.
        """
        if target is None:
            target = 2
        mapped_quadruples = mapped_quadruples.numpy()
        other_columns = sorted(set(range(4)).difference({target}))
        unique_pairs, pair_idx_to_triple_idx = np.unique(
            mapped_quadruples[:, other_columns], return_inverse=True, axis=0
        )
        num_pairs = unique_pairs.shape[0]
        tails = mapped_quadruples[:, target]
        target_size = num_relations if target == 1 else num_entities
        compressed = scipy.sparse.coo_matrix(
            (
                np.ones(mapped_quadruples.shape[0], dtype=np.float32),
                (pair_idx_to_triple_idx, tails),
            ),
            shape=(num_pairs, target_size),
        )
        # convert to csr for fast row slicing
        compressed = compressed.tocsr()
        return cls(pairs=unique_pairs, compressed=compressed)

    def __len__(self) -> int:  # noqa: D105
        return self.pairs.shape[0]

    def __getitem__(self, item: int) -> LCWATemporalBatchType:  # noqa: D105
        return self.pairs[item], np.asarray(self.compressed[item, :].todense())[0, :]
