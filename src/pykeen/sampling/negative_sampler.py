# -*- coding: utf-8 -*-

"""Basic structure for a negative sampler."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, Optional, Tuple

import torch

from ..triples import TriplesFactory
from ..utils import normalize_string

__all__ = [
    'NegativeSampler',
]


class NegativeSampler(ABC):
    """A negative sampler."""

    #: The default strategy for optimizing the negative sampler's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Mapping[str, Any]]]

    def __init__(
        self,
        triples_factory: TriplesFactory,
        num_negs_per_pos: Optional[int] = None,
        filtered: bool = None,
        corruption_scheme: Tuple[str] = None,
    ) -> None:
        """Initialize the negative sampler with the given entities.

        :param triples_factory: The factory holding the triples to sample from
        :param num_negs_per_pos: Number of negative samples to make per positive triple. Defaults to 1.
        :param filtered: Whether proposed corrupted triples that are in the training data should be filtered.
            Defaults to False.
        :param filtered: Whether proposed corrupted triples that are in the training data should be filtered.
            Defaults to False.
        :param corruption_scheme: What sides (h, r, t) should be corrupted. Defaults to head and tail (h, t).
        """
        self.triples_factory = triples_factory
        self.num_negs_per_pos = num_negs_per_pos if num_negs_per_pos is not None else 1
        self.filtered = filtered if filtered is not None else False
        self.corruption_scheme = corruption_scheme if corruption_scheme is not None else ('h', 't')

        # Set the indices
        self._set_corruption_indices(corruption_scheme)

    def _set_corruption_indices(self, corruption_scheme: Tuple[str]):
        self._corruption_indices = [0 if side == 'h' else 1 if side == 'r' else 2 for side in corruption_scheme]

    @classmethod
    def get_normalized_name(cls) -> str:
        """Get the normalized name of the negative sampler."""
        return normalize_string(cls.__name__, suffix=NegativeSampler.__name__)

    @property
    def num_entities(self) -> int:  # noqa: D401
        """The number of entities to sample from."""
        return self.triples_factory.num_entities

    @property
    def num_relations(self) -> int:  # noqa: D401
        """The number of relations to sample from."""
        return self.triples_factory.num_relations

    @abstractmethod
    def sample(self, positive_batch: torch.LongTensor) -> torch.LongTensor:
        """Generate negative samples from the positive batch."""
        raise NotImplementedError
