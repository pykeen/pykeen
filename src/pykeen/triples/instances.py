# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

from abc import ABC
from dataclasses import dataclass
from typing import Generic, Mapping, Optional, Tuple, TypeVar

import numpy as np
import scipy.sparse
import torch
from torch.utils import data

from ..typing import MappedTriples
from ..utils import fix_dataclass_init_docs

__all__ = [
    'Instances',
    'SLCWAInstances',
    'LCWAInstances',
    'MultimodalInstances',
    'MultimodalSLCWAInstances',
    'MultimodalLCWAInstances',
]

BatchType = TypeVar("BatchType")
LCWASampleType = Tuple[MappedTriples, torch.FloatTensor]
LCWABatchType = Tuple[MappedTriples, torch.FloatTensor]
SLCWASampleType = TypeVar('SLCWASampleType', bound=MappedTriples)
SLCWABatchType = Tuple[MappedTriples, MappedTriples, Optional[torch.BoolTensor]]


@fix_dataclass_init_docs
@dataclass
class Instances(data.Dataset, Generic[BatchType], ABC):
    """Triples and mappings to their indices."""

    def __len__(self):  # noqa:D401
        """The number of instances."""
        raise NotImplementedError

    def __getitem__(self, item: int) -> BatchType:  # noqa: D105
        raise NotImplementedError

    @classmethod
    def from_triples(cls, mapped_triples: MappedTriples, num_entities: int) -> 'Instances':
        """Create instances from mapped triples.

        :param mapped_triples: shape: (num_triples, 3)
            The ID-based triples.
        :param num_entities:
            The number of entities.
        :return:
            The instances.
        """
        raise NotImplementedError


@fix_dataclass_init_docs
@dataclass
class SLCWAInstances(Instances[MappedTriples]):
    """Triples and mappings to their indices for sLCWA."""

    #: The mapped triples, shape: (num_triples, 3)
    mapped_triples: MappedTriples

    def __len__(self):  # noqa: D105
        return self.mapped_triples.shape[0]

    def __getitem__(self, item: int) -> MappedTriples:  # noqa: D105
        return self.mapped_triples[item]

    @classmethod
    def from_triples(cls, mapped_triples: MappedTriples, num_entities: int) -> Instances:  # noqa:D102
        return cls(mapped_triples=mapped_triples)


@fix_dataclass_init_docs
@dataclass
class LCWAInstances(Instances[LCWABatchType]):
    """Triples and mappings to their indices for LCWA."""

    #: The unique pairs
    pairs: np.ndarray

    #: The compressed triples in CSR format
    compressed: scipy.sparse.csr_matrix

    @classmethod
    def from_triples(cls, mapped_triples: MappedTriples, num_entities: int) -> Instances:
        """
        Create LCWA instances from triples.

        :param mapped_triples: shape: (num_triples, 3)
            The ID-based triples.
        :param num_entities:
            The number of entities.

        :return:
            The instances.
        """
        mapped_triples = mapped_triples.numpy()
        unique_hr, pair_idx_to_triple_idx = np.unique(mapped_triples[:, :2], return_inverse=True, axis=0)
        num_pairs = unique_hr.shape[0]
        tails = mapped_triples[:, 2]
        compressed = scipy.sparse.coo_matrix(
            (np.ones(mapped_triples.shape[0], dtype=np.float32), (pair_idx_to_triple_idx, tails)),
            shape=(num_pairs, num_entities),
        )
        # convert to csr for fast row slicing
        compressed = compressed.tocsr()
        return cls(pairs=unique_hr, compressed=compressed)

    def __len__(self) -> int:  # noqa: D105
        return self.pairs.shape[0]

    def __getitem__(self, item: int) -> LCWABatchType:  # noqa: D105
        return self.pairs[item], np.asarray(self.compressed[item, :].todense())[0, :]


@fix_dataclass_init_docs
@dataclass
class MultimodalInstances(Instances):
    """Triples and mappings to their indices as well as multimodal data."""

    #: TODO: do we need these?
    numeric_literals: Mapping[str, np.ndarray]
    literals_to_id: Mapping[str, int]


@fix_dataclass_init_docs
@dataclass
class MultimodalSLCWAInstances(SLCWAInstances, MultimodalInstances):
    """Triples and mappings to their indices as well as multimodal data for sLCWA."""


@fix_dataclass_init_docs
@dataclass
class MultimodalLCWAInstances(LCWAInstances, MultimodalInstances):
    """Triples and mappings to their indices as well as multimodal data for LCWA."""
