# -*- coding: utf-8 -*-

"""Tokenization algorithms for NodePiece."""

import logging
import pathlib
from abc import abstractmethod
from collections import defaultdict
from typing import Collection, Mapping, Optional, Tuple

import more_itertools
import numpy
import torch
from class_resolver import ClassResolver, HintOrType, OptionalKwargs

from .anchor_search import AnchorSearcher, anchor_searcher_resolver
from .anchor_selection import AnchorSelection, anchor_selection_resolver
from .loader import PrecomputedTokenizerLoader, precomputed_tokenizer_loader_resolver
from .utils import prepare_edges_for_metis, random_sample_no_replacement
from ...constants import PYKEEN_MODULE
from ...typing import DeviceHint, MappedTriples
from ...utils import format_relative_comparison, get_edge_index, resolve_device

__all__ = [
    # Resolver
    "tokenizer_resolver",
    # Base classes
    "Tokenizer",
    # Concrete classes
    "RelationTokenizer",
    "AnchorTokenizer",
    "MetisAnchorTokenizer",
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
        return 2 * num_relations + 1, random_sample_no_replacement(
            pool=e2r, num_tokens=num_tokens, num_entities=num_entities
        )


class AnchorTokenizer(Tokenizer):
    """
    Tokenize entities by representing them as a bag of anchor entities.

    The entities are chosen by shortest path distance.
    """

    anchor_selection: AnchorSelection
    searcher: AnchorSearcher

    def __init__(
        self,
        # TODO: expose num_anchors?
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

    def _call(
        self,
        edge_index: torch.LongTensor,
        num_tokens: int,
        num_entities: int,
    ) -> Tuple[int, torch.LongTensor]:
        edge_index = edge_index.numpy()
        # select anchors
        logger.info(f"Selecting anchors according to {self.anchor_selection}")
        anchors = self.anchor_selection(edge_index=edge_index)
        if len(numpy.unique(anchors)) < len(anchors):
            logger.warning(f"Only {len(numpy.unique(anchors))} out of {len(anchors)} anchors are unique")
        # find closest anchors
        logger.info(f"Searching closest anchors with {self.searcher}")
        tokens = self.searcher(edge_index=edge_index, anchors=anchors, k=num_tokens, num_entities=num_entities)
        num_empty = (tokens < 0).all(axis=1).sum()
        if num_empty > 0:
            logger.warning(
                f"{format_relative_comparison(part=num_empty, total=num_entities)} " f"do not have any anchor.",
            )
        # convert to torch
        return len(anchors) + 1, torch.as_tensor(tokens, dtype=torch.long)

    # docstr-coverage: inherited
    def __call__(
        self,
        mapped_triples: MappedTriples,
        num_tokens: int,
        num_entities: int,
        num_relations: int,
    ) -> Tuple[int, torch.LongTensor]:  # noqa: D102
        return self._call(
            edge_index=get_edge_index(mapped_triples=mapped_triples),
            num_tokens=num_tokens,
            num_entities=num_entities,
        )


class MetisAnchorTokenizer(AnchorTokenizer):
    """
    An anchor tokenizer, which first partitions the graph using METIS.

    We use the binding by :mod:`torch_sparse`. The METIS graph partitioning algorithm is described here:
    http://glaros.dtc.umn.edu/gkhome/metis/metis/overview
    """

    def __init__(self, num_partitions: int = 2, device: DeviceHint = None, **kwargs):
        """Initialize the tokenizer.

        :param num_partitions:
            the number of partitions obtained through Metis.
        :param device:
            the device to use for tokenization
        :param kwargs:
            additional keyword-based parameters passed to :meth:`AnchorTokenizer.__init__`. note that there will be one
            anchor tokenizer per partition, i.e., the vocabulary size will grow respectively.
        """
        super().__init__(**kwargs)
        self.num_partitions = num_partitions
        self.device = resolve_device(device)

    # docstr-coverage: inherited
    def __call__(
        self,
        mapped_triples: MappedTriples,
        num_tokens: int,
        num_entities: int,
        num_relations: int,
    ) -> Tuple[int, torch.LongTensor]:  # noqa: D102
        try:
            import torch_sparse
        except ImportError as err:
            raise ImportError(f"{self.__class__.__name__} requires `torch_sparse` to be installed.") from err

        logger.info(f"Partitioning the graph into {self.num_partitions} partitions.")
        edge_index = get_edge_index(mapped_triples=mapped_triples)
        # To prevent possible segfaults in the METIS C code, METIS expects a graph
        # (1) without self-loops; (2) with inverse edges added; (3) with unique edges only
        # https://github.com/KarypisLab/METIS/blob/94c03a6e2d1860128c2d0675cbbb86ad4f261256/libmetis/checkgraph.c#L18
        row, col = prepare_edges_for_metis(edge_index=edge_index)
        re_ordered_adjacency, bound, perm = torch_sparse.partition(
            src=torch_sparse.SparseTensor(row=row, col=col, sparse_sizes=(num_entities, num_entities)).to(
                device=self.device
            ),
            num_parts=self.num_partitions,
            recursive=True,
        )
        re_ordered_adjacency = re_ordered_adjacency.cpu()
        sizes = bound.diff()
        logger.info(f"Partition sizes: min: {sizes.min().item()}, max: {sizes.max().item()}")

        # select independently per partition
        vocabulary_size = 0
        assignment = []
        edge_count = 0
        for low, high in more_itertools.pairwise(bound.tolist()):
            # select adjacency part;
            # note: the indices will automatically be in [0, ..., high - low), since they are *local* indices
            edge_index = re_ordered_adjacency[low:high, low:high].to_torch_sparse_coo_tensor().coalesce().indices()
            edge_count += edge_index.shape[1]
            num_entities = high - low
            this_vocabulary_size, this_assignment = super()._call(
                edge_index=edge_index, num_tokens=num_tokens, num_entities=num_entities
            )
            assert this_assignment.shape[0] == num_entities

            # offset
            mask = this_assignment < 0
            this_assignment = this_assignment + vocabulary_size
            this_assignment[mask] = -1

            # the -1 comes from the shared padding token
            vocabulary_size += this_vocabulary_size - 1

            # note: permutation will be later on reverted
            assignment.append(this_assignment)

        # add back 1 for the shared padding token
        vocabulary_size += 1
        total_edges = mapped_triples.shape[0]
        logger.info(
            f"Partitioned anchor tokenization lead to ignoring "
            f"{format_relative_comparison(part=total_edges - edge_count, total=total_edges)} connections.",
        )
        # TODO: check if perm is used correctly
        return vocabulary_size, torch.cat(assignment, dim=0)[perm]


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
            assignment = random_sample_no_replacement(pool=self.pool, num_tokens=num_tokens, num_entities=num_entities)
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
