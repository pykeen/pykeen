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
        filtered: bool = False,
    ) -> None:
        """Initialize the negative sampler with the given entities.

        :param triples_factory: The factory holding the triples to sample from
        :param num_negs_per_pos: Number of negative samples to make per positive triple. Defaults to 1.
        :param filtered: Whether proposed corrupted triples that are in the training data should be filtered.
            Defaults to False. See explanation in :func:`filter_negative_triples` for why this is
            a reasonable default.
        """
        self.triples_factory = triples_factory
        self.num_negs_per_pos = num_negs_per_pos if num_negs_per_pos is not None else 1
        self.filtered = filtered
        # Create mapped triples attribute that is required for filtering
        self.mapped_triples = None

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

    def filter_negative_triples(self, negative_batch: torch.LongTensor) -> Tuple[torch.LongTensor, torch.Tensor]:
        """Filter all proposed negative samples that are positive in the training dataset.

        Normally there is a low probability that proposed negative samples are positive in the training datasets and
        thus act as false negatives. This is expected to act as a kind of regularization, since it adds noise signal to
        the training data. However, the degree of regularization is hard to control since the added noise signal depends
        on the ratio of true triples for a given entity relation or entity entity pair. Therefore, the effects are hard
        to control and a researcher might want to exclude the possibility of having false negatives in the proposed
        negative triples.
        Note: Filtering is a very expensive task, since every proposed negative sample has to be checked against the
        entire training dataset.

        :param negative_batch: The batch of negative triples
        """
        # Make sure the mapped triples are initiated
        if self.mapped_triples is None:
            # Copy the mapped triples to the device for efficient filtering
            self.mapped_triples = self.triples_factory.mapped_triples.to(negative_batch.device)

        try:
            # Check which heads of the mapped triples are also in the negative triples
            head_filter = (
                self.mapped_triples[:, 0:1].view(1, -1) == negative_batch[:, 0:1]  # type: ignore
            ).max(axis=0)[0]
            # Reduce the search space by only using possible matches that at least contain the head we look for
            sub_mapped_triples = self.mapped_triples[head_filter]  # type: ignore
            # Check in this subspace which relations of the mapped triples are also in the negative triples
            relation_filter = (sub_mapped_triples[:, 1:2].view(1, -1) == negative_batch[:, 1:2]).max(axis=0)[0]
            # Reduce the search space by only using possible matches that at least contain head and relation we look for
            sub_mapped_triples = sub_mapped_triples[relation_filter]
            # Create a filter indicating which of the proposed negative triples are positive in the training dataset
            final_filter = (sub_mapped_triples[:, 2:3].view(1, -1) == negative_batch[:, 2:3]).max(axis=1)[0]
        except RuntimeError as e:
            # In cases where no triples should be filtered, the subspace reduction technique above will fail
            if str(e) == (
                'cannot perform reduction function max on tensor with no elements because the operation does not '
                'have an identity'
            ):
                final_filter = torch.zeros(negative_batch.shape[0], dtype=torch.bool, device=negative_batch.device)
            else:
                raise e
        # Return only those proposed negative triples that are not positive in the training dataset
        return negative_batch[~final_filter], ~final_filter
