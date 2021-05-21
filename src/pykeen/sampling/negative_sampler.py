# -*- coding: utf-8 -*-

"""Basic structure for a negative sampler."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, Optional, Tuple

import torch
from class_resolver import HintOrType, normalize_string

from .filtering import Filterer, filterer_resolver
from ..triples import CoreTriplesFactory

__all__ = [
    'NegativeSampler',
]


class NegativeSampler(ABC):
    """A negative sampler."""

    #: The default strategy for optimizing the negative sampler's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Mapping[str, Any]]] = dict(
        num_negs_per_pos=dict(type=int, low=1, high=100, log=True),
    )

    #: A filterer for negative batches
    filterer: Optional[Filterer]

    num_entities: int
    num_relations: int
    num_negs_per_pos: int

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        num_negs_per_pos: Optional[int] = None,
        filtered: bool = False,
        filterer: HintOrType[Filterer] = None,
        filterer_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialize the negative sampler with the given entities.

        :param triples_factory: The factory holding the positive training triples
        :param num_negs_per_pos: Number of negative samples to make per positive triple. Defaults to 1.
        :param filtered: Whether proposed corrupted triples that are in the training data should be filtered.
            Defaults to False. See explanation in :func:`filter_negative_triples` for why this is
            a reasonable default.
        :param filterer: If filtered is set to True, this can be used to choose which filter module from
            :mod:`pykeen.sampling.filtering` is used.
        :param filterer_kwargs:
            Additional keyword-based arguments passed to the filterer upon construction.
        """
        self.num_entities = triples_factory.num_entities
        self.num_relations = triples_factory.num_relations
        self.num_negs_per_pos = num_negs_per_pos if num_negs_per_pos is not None else 1
        self.filterer = filterer_resolver.make(
            filterer,
            pos_kwargs=filterer_kwargs,
            mapped_triples=triples_factory.mapped_triples,
        ) if filterer is not None or filtered else None

    @classmethod
    def get_normalized_name(cls) -> str:
        """Get the normalized name of the negative sampler."""
        return normalize_string(cls.__name__, suffix=NegativeSampler.__name__)

    def sample(self, positive_batch: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.BoolTensor]]:
        """
        Generate negative samples from the positive batch.

        :param positive_batch: shape: (batch_size, 3)
            The positive triples.

        :return:
            A pair (negative_batch, filter_mask) where

            1. negative_batch: shape: (batch_size, num_negatives, 3)
               The negative batch. ``negative_batch[i, :, :]`` contains the negative examples generated from
               ``positive_batch[i, :]``.
            2. filter_mask: shape: (batch_size, num_negatives)
               An optional filter mask. True where negative samples are valid.
        """
        # create unfiltered negative batch by corruption
        negative_batch = self.corrupt_batch(positive_batch=positive_batch)

        if self.filterer is None:
            return negative_batch, None

        # If filtering is activated, all negative triples that are positive in the training dataset will be removed
        return negative_batch, self.filterer(negative_batch=negative_batch)

    @abstractmethod
    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:
        """
        Generate negative samples from the positive batch without application of any filter.

        :param positive_batch: shape: (batch_size, 3)
            The positive triples.

        :return: shape: (batch_size, num_negs_per_pos, 3)
            The negative triples. ``result[i, :, :]`` contains the negative examples generated from
            ``positive_batch[i, :]``.
        """
        raise NotImplementedError
