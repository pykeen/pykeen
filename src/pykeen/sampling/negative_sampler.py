# -*- coding: utf-8 -*-

"""Basic structure for a negative sampler."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, Optional, Tuple, Type, Union

import torch
from class_resolver import Hint

from .filtering import Filterer, filterer_resolver
from ..triples import TriplesFactory
from ..utils import normalize_string

__all__ = [
    'NegativeSampler',
]


class NegativeSampler(ABC):
    """A negative sampler."""

    #: The default strategy for optimizing the negative sampler's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Mapping[str, Any]]]

    filterer: Optional[Filterer]

    def __init__(
        self,
        triples_factory: TriplesFactory,
        num_negs_per_pos: Optional[int] = None,
        filtered: bool = False,
        filterer: Hint[Union[Filterer, Type[Filterer]]] = None,
    ) -> None:
        """Initialize the negative sampler with the given entities.

        :param triples_factory: The factory holding the triples to sample from
        :param num_negs_per_pos: Number of negative samples to make per positive triple. Defaults to 1.
        :param filtered: Whether proposed corrupted triples that are in the training data should be filtered.
            Defaults to False. See explanation in :func:`filter_negative_triples` for why this is
            a reasonable default.
        :param filterer: If filtered is set to True, this can be used to choose which filter module from
            :mod:`pykeen.sampling.filtering` is used.
        """
        self.triples_factory = triples_factory
        self.num_negs_per_pos = num_negs_per_pos if num_negs_per_pos is not None else 1
        self.filterer = filterer_resolver.make(
            filterer,
            triples_factory=triples_factory,
        ) if filtered else None

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
