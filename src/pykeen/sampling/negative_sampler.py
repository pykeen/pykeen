# -*- coding: utf-8 -*-

"""Basic structure for a negative sampler."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, Optional, Set, Tuple

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
        filtered: bool = False,
        corruption_scheme: Set[str] = None,
    ) -> None:
        """Initialize the negative sampler with the given entities.

        :param triples_factory: The factory holding the triples to sample from
        :param num_negs_per_pos: Number of negative samples to make per positive triple. Defaults to 1.
        :param filtered: Whether proposed corrupted triples that are in the training data should be filtered.
            Defaults to False.
        :param corruption_scheme: What sides ('h', 'r', 't') should be corrupted. Defaults to head and tail ('h', 't').
        """
        self.triples_factory = triples_factory
        self.num_negs_per_pos = num_negs_per_pos if num_negs_per_pos is not None else 1
        self.filtered = filtered
        self.corruption_scheme = corruption_scheme or ('h', 't')
        # Set the indices
        self._corruption_indices = [0 if side == 'h' else 1 if side == 'r' else 2 for side in self.corruption_scheme]
        # Tracking whether required init steps for negative sample filtering are performed
        self._filter_init = False

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
    def sample(self, positive_batch: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:
        """Generate negative samples from the positive batch."""
        raise NotImplementedError

    def _filter_negative_triples(self, negative_batch: torch.LongTensor) -> torch.Tensor:
        """Filter all proposed negative samples that are positive in the training dataset.

        Normally there is a low probability that proposed negative samples are positive in the training datasets and
        thus act as false negatives. This is expected to act as a kind of regularization, since it adds noise signal to
        the training data. However, the degree of regularization is hard to control since the added noise signal depends
        on the ratio of true triples for a given entity relation or entity entity pair. Therefore, the effects are hard
        to control and a researcher might want to exclude the possibility of having false negatives in the proposed
        negative triples.
        """
        # Make sure the mapped triples are on the right device
        if not self._filter_init:
            # Copy the mapped triples to the device for efficient filtering
            self.mapped_triples = self.triples_factory.mapped_triples.to(negative_batch.device)
            self._filter_init = True
        # Check which heads of the mapped triples are also in the negative triples
        head_filter = (self.mapped_triples[:, 0:1].view(1, -1) == negative_batch[:, 0:1]).max(axis=0)[0]
        # Reduce the search space by only using possible matches that at least contain the head we look for
        sub_mapped_triples = self.mapped_triples[head_filter]
        # Check in this subspace which relations of the mapped triples are also in the negative triples
        relation_filter = (sub_mapped_triples[:, 1:2].view(1, -1) == negative_batch[:, 1:2]).max(axis=0)[0]
        # Reduce the search space by only using possible matches that at least contain head and relation we look for
        sub_mapped_triples = sub_mapped_triples[relation_filter]
        # Create a filter indicating which of the proposed negative triples are positive in the training dataset
        final_filter = (sub_mapped_triples[:, 2:3].view(1, -1) == negative_batch[:, 2:3]).max(axis=1)[0]
        # Return only those proposed negative triples that are not positive in the training dataset
        return ~final_filter
