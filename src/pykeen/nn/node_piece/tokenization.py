# -*- coding: utf-8 -*-

"""Tokenization algorithms for NodePiece."""

import logging
import pathlib
from abc import abstractmethod
from collections import defaultdict
from typing import Collection, Mapping, Optional, Tuple

import numpy
import torch
from class_resolver import ClassResolver, HintOrType, OptionalKwargs

from .anchor_search import AnchorSearcher, anchor_searcher_resolver
from .anchor_selection import AnchorSelection, anchor_selection_resolver
from .loader import PrecomputedTokenizerLoader, precomputed_tokenizer_loader_resolver
from .utils import random_sample_no_replacement
from ...constants import PYKEEN_MODULE
from ...typing import MappedTriples
from ...utils import format_relative_comparison, get_edge_index

__all__ = [
    # Resolver
    "tokenizer_resolver",
    # Base classes
    "Tokenizer",
    # Concrete classes
    "RelationTokenizer",
    "AnchorTokenizer",
    "PrecomputedPoolTokenizer",
]

logger = logging.getLogger(__name__)


class Tokenizer:
    """A base class for tokenizers for NodePiece representations."""

    @abstractmethod
    def __call__(
        self,
        mapped_triples: MappedTriples,
        num_tokens: int,
        num_entities: int,
        num_relations: int,
    ) -> Tuple[int, torch.LongTensor]:
        """
        Tokenize the entities contained given the triples.

        :param mapped_triples: shape: (n, 3)
            the ID-based triples
        :param num_tokens:
            the number of tokens to select for each entity
        :param num_entities:
            the number of entities
        :param num_relations:
            the number of relations

        :return: shape: (num_entities, num_tokens), -1 <= res < vocabulary_size
            the selected relation IDs for each entity. -1 is used as a padding token.
        """
        raise NotImplementedError


class RelationTokenizer(Tokenizer):
    """Tokenize entities by representing them as a bag of relations."""

    # docstr-coverage: inherited
    def __call__(
        self,
        mapped_triples: MappedTriples,
        num_tokens: int,
        num_entities: int,
        num_relations: int,
    ) -> Tuple[int, torch.LongTensor]:  # noqa: D102
        # tokenize: represent entities by bag of relations
        h, r, t = mapped_triples.t()

        # collect candidates
        e2r = defaultdict(set)
        for e, r_ in (
            torch.cat(
                [
                    torch.stack([h, r], dim=1),
                    torch.stack([t, r + num_relations], dim=1),
                ],
                dim=0,
            )
            .unique(dim=0)
            .tolist()
        ):
            e2r[e].add(r_)

        # randomly sample without replacement num_tokens relations for each entity
        return 2 * num_relations + 1, random_sample_no_replacement(pool=e2r, num_tokens=num_tokens)


class AnchorTokenizer(Tokenizer):
    """
    Tokenize entities by representing them as a bag of anchor entities.

    The entities are chosen by shortest path distance.
    """

    def __init__(
        self,
        selection: HintOrType[AnchorSelection] = None,
        selection_kwargs: OptionalKwargs = None,
        searcher: HintOrType[AnchorSearcher] = None,
        searcher_kwargs: OptionalKwargs = None,
    ) -> None:
        """
        Initialize the tokenizer.

        :param selection:
            the anchor node selection strategy.
        :param selection_kwargs:
            additional keyword-based arguments passed to the selection strategy
        :param searcher:
            the component for searching the closest anchors for each entity
        :param searcher_kwargs:
            additional keyword-based arguments passed to the searcher
        """
        self.anchor_selection = anchor_selection_resolver.make(selection, pos_kwargs=selection_kwargs)
        self.searcher = anchor_searcher_resolver.make(searcher, pos_kwargs=searcher_kwargs)

    # docstr-coverage: inherited
    def __call__(
        self,
        mapped_triples: MappedTriples,
        num_tokens: int,
        num_entities: int,
        num_relations: int,
    ) -> torch.LongTensor:  # noqa: D102
        edge_index = get_edge_index(mapped_triples=mapped_triples).numpy()
        # select anchors
        logger.info(f"Selecting anchors according to {self.anchor_selection}")
        anchors = self.anchor_selection(edge_index=edge_index)
        if len(numpy.unique(anchors)) < len(anchors):
            logger.warning(f"Only {len(numpy.unique(anchors))} out of {len(anchors)} anchors are unique")
        # find closest anchors
        logger.info(f"Searching closest anchors with {self.searcher}")
        tokens = self.searcher(edge_index=edge_index, anchors=anchors, k=num_tokens)
        num_empty = (tokens < 0).all(axis=1).sum()
        if num_empty > 0:
            logger.warning(
                f"{format_relative_comparison(part=num_empty, total=num_entities)} " f"do not have any anchor.",
            )
        # convert to torch
        return len(anchors) + 1, torch.as_tensor(tokens, dtype=torch.long)


class PrecomputedPoolTokenizer(Tokenizer):
    """A tokenizer using externally precomputed tokenization."""

    @classmethod
    def _load_pool(
        cls,
        *,
        path: Optional[pathlib.Path] = None,
        url: Optional[str] = None,
        download_kwargs: OptionalKwargs = None,
        pool: Optional[Mapping[int, Collection[int]]] = None,
        loader: HintOrType[PrecomputedTokenizerLoader] = None,
    ) -> Tuple[Mapping[int, Collection[int]], int]:
        """Load a precomputed pool via one of the supported ways."""
        if pool is not None:
            return pool, max(c for candidates in pool.values() for c in candidates) + 1 + 1  # +1 for padding
        if url is not None and path is None:
            module = PYKEEN_MODULE.module(__name__, tokenizer_resolver.normalize_cls(cls=cls))
            path = module.ensure(url=url, download_kwargs=download_kwargs)
        if path is None:
            raise ValueError("Must provide at least one of pool, path, or url.")

        if not path.is_file():
            raise FileNotFoundError(path)
        logger.info(f"Loading precomputed pools from {path}")
        return precomputed_tokenizer_loader_resolver.make(loader)(path=path)

    def __init__(
        self,
        *,
        path: Optional[pathlib.Path] = None,
        url: Optional[str] = None,
        download_kwargs: OptionalKwargs = None,
        pool: Optional[Mapping[int, Collection[int]]] = None,
        randomize_selection: bool = False,
        loader: HintOrType[PrecomputedTokenizerLoader] = None,
    ):
        r"""
        Initialize the tokenizer.

        .. note ::
            the preference order for loading the precomputed pools is (1) from the given pool (2) from the given path,
            and (3) by downloading from the given url

        :param path:
            a path for a file containing the precomputed pools
        :param url:
            an url to download the file with precomputed pools from
        :param download_kwargs:
            additional download parameters, passed to pystow.Module.ensure
        :param pool:
            the precomputed pools.
        :param randomize_selection:
            whether to randomly choose from tokens, or always take the first `num_token` precomputed tokens.
        :param loader:
            the loader to use for loading the pool
        :raises ValueError: If the pool's keys are not contiguous on $0 \dots N-1$.
        """
        self.pool, self.vocabulary_size = self._load_pool(
            path=path, url=url, pool=pool, download_kwargs=download_kwargs, loader=loader
        )
        # verify pool
        if set(self.pool.keys()) != set(range(len(self.pool))):
            raise ValueError("Expected pool to contain contiguous keys 0...(N-1)")
        self.randomize_selection = randomize_selection

    # docstr-coverage: inherited
    def __call__(
        self, mapped_triples: MappedTriples, num_tokens: int, num_entities: int, num_relations: int
    ) -> Tuple[int, torch.LongTensor]:  # noqa: D102
        if num_entities != len(self.pool):
            raise ValueError(f"Invalid number of entities ({num_entities}); expected {len(self.pool)}")
        if self.randomize_selection:
            assignment = random_sample_no_replacement(pool=self.pool, num_tokens=num_tokens)
        else:
            # choose first num_tokens
            assignment = torch.full(
                size=(len(self.pool), num_tokens),
                dtype=torch.long,
                fill_value=-1,
            )
            # TODO: vectorization?
            for idx, this_pool in self.pool.items():
                this_pool_t = torch.as_tensor(data=list(this_pool)[:num_tokens], dtype=torch.long)
                assignment[idx, : len(this_pool_t)] = this_pool_t
        return self.vocabulary_size, assignment


tokenizer_resolver: ClassResolver[Tokenizer] = ClassResolver.from_subclasses(
    base=Tokenizer,
    default=RelationTokenizer,
)
