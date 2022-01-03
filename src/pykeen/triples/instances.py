# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

from abc import ABC
from typing import Generic, Mapping, Optional, Tuple, TypeVar

import numpy as np
import scipy.sparse
import torch
from torch.utils import data

from ..typing import MappedTriples

__all__ = [
    "Instances",
    "SLCWAInstances",
    "LCWAInstances",
    "MultimodalInstances",
    "MultimodalSLCWAInstances",
    "MultimodalLCWAInstances",
]

BatchType = TypeVar("BatchType")
LCWASampleType = Tuple[MappedTriples, torch.FloatTensor]
LCWABatchType = Tuple[MappedTriples, torch.FloatTensor]
SLCWASampleType = TypeVar("SLCWASampleType", bound=MappedTriples)
SLCWABatchType = Tuple[MappedTriples, MappedTriples, Optional[torch.BoolTensor]]


class Instances(data.Dataset[BatchType], Generic[BatchType], ABC):
    """Triples and mappings to their indices."""

    def __len__(self):  # noqa:D401
        """The number of instances."""
        raise NotImplementedError

    @classmethod
    def from_triples(
        cls,
        mapped_triples: MappedTriples,
        *,
        num_entities: int,
        num_relations: int,
        **kwargs,
    ) -> "Instances":
        """Create instances from mapped triples.

        :param mapped_triples: shape: (num_triples, 3)
            The ID-based triples.
        :param num_entities:
            The number of entities.
        :param num_relations:
            The number of relations.

        :return:
            The instances.
        """
        raise NotImplementedError


class SLCWAInstances(Instances[MappedTriples]):
    """Triples and mappings to their indices for sLCWA."""

    def __init__(self, *, mapped_triples: MappedTriples):
        """Initialize the sLCWA instances.

        :param mapped_triples: The mapped triples, shape: (num_triples, 3)
        """
        self.mapped_triples = mapped_triples

    def __len__(self) -> int:  # noqa: D105
        return self.mapped_triples.shape[0]

    def __getitem__(self, item: int) -> MappedTriples:  # noqa: D105
        return self.mapped_triples[item]

    @classmethod
    def from_triples(
        cls,
        mapped_triples: MappedTriples,
        *,
        num_entities: int,
        num_relations: int,
        **kwargs,
    ) -> Instances:  # noqa:D102
        return cls(mapped_triples=mapped_triples)


class LCWAInstances(Instances[LCWABatchType]):
    """Triples and mappings to their indices for LCWA."""

    def __init__(self, *, pairs: np.ndarray, compressed: scipy.sparse.csr_matrix):
        """Initialize the LCWA instances.

        :param pairs: The unique pairs
        :param compressed: The compressed triples in CSR format
        """
        self.pairs = pairs
        self.compressed = compressed

    @classmethod
    def from_triples(
        cls,
        mapped_triples: MappedTriples,
        *,
        num_entities: int,
        num_relations: int,
        target: Optional[int] = None,
        **kwargs,
    ) -> Instances:
        """
        Create LCWA instances from triples.

        :param mapped_triples: shape: (num_triples, 3)
            The ID-based triples.
        :param num_entities:
            The number of entities.
        :param num_relations:
            The number of relations.
        :param target:
            The column to predict

        :return:
            The instances.
        """
        if target is None:
            target = 2
        mapped_triples = mapped_triples.numpy()
        other_columns = sorted(set(range(3)).difference({target}))
        unique_pairs, pair_idx_to_triple_idx = np.unique(mapped_triples[:, other_columns], return_inverse=True, axis=0)
        num_pairs = unique_pairs.shape[0]
        tails = mapped_triples[:, target]
        target_size = num_relations if target == 1 else num_entities
        compressed = scipy.sparse.coo_matrix(
            (np.ones(mapped_triples.shape[0], dtype=np.float32), (pair_idx_to_triple_idx, tails)),
            shape=(num_pairs, target_size),
        )
        # convert to csr for fast row slicing
        compressed = compressed.tocsr()
        return cls(pairs=unique_pairs, compressed=compressed)

    @staticmethod
    def _get_target_size(num_entities: int, num_relations: int) -> int:
        raise NotImplementedError

    def __len__(self) -> int:  # noqa: D105
        return self.pairs.shape[0]

    def __getitem__(self, item: int) -> LCWABatchType:  # noqa: D105
        return self.pairs[item], np.asarray(self.compressed[item, :].todense())[0, :]


class MultimodalInstances:
    """Triples and mappings to their indices as well as multimodal data."""

    def __init__(self, *, numeric_literals: Mapping[str, np.ndarray], literals_to_id: Mapping[str, int]):
        """Initialize the multimodal instances.

        :param numeric_literals: A mapping from relations to numeric literals
        :param literals_to_id: A mapping from literals to their identifiers
        """
        self.numeric_literals = numeric_literals
        self.literals_to_id = literals_to_id


class MultimodalSLCWAInstances(MultimodalInstances, SLCWAInstances):
    """Triples and mappings to their indices as well as multimodal data for sLCWA."""

    def __init__(
        self,
        *,
        mapped_triples: MappedTriples,
        numeric_literals: Mapping[str, np.ndarray],
        literals_to_id: Mapping[str, int],
    ):
        """Initialize the multimodal sLCWA instances.

        :param mapped_triples: The mapped triples, shape: (num_triples, 3)
        :param numeric_literals: A mapping from relations to numeric literals
        :param literals_to_id: A mapping from literals to their identifiers
        """
        SLCWAInstances.__init__(self, mapped_triples=mapped_triples)
        MultimodalInstances.__init__(self, numeric_literals=numeric_literals, literals_to_id=literals_to_id)


class MultimodalLCWAInstances(MultimodalInstances, LCWAInstances):
    """Triples and mappings to their indices as well as multimodal data for LCWA."""

    def __init__(
        self,
        *,
        pairs: np.ndarray,
        compressed: scipy.sparse.csr_matrix,
        numeric_literals: Mapping[str, np.ndarray],
        literals_to_id: Mapping[str, int],
    ):
        """Initialize the multimodal LCWA instances.

        :param pairs: The unique pairs
        :param compressed: The compressed triples in CSR format
        :param numeric_literals: A mapping from relations to numeric literals
        :param literals_to_id: A mapping from literals to their identifiers
        """
        LCWAInstances.__init__(self, pairs=pairs, compressed=compressed)
        MultimodalInstances.__init__(self, numeric_literals=numeric_literals, literals_to_id=literals_to_id)
