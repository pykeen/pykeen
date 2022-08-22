# -*- coding: utf-8 -*-

"""
Anchor selection for NodePiece.

An anchor selection method selects a given number of entities from the KG which serve as *anchors* to describe other
entities. Most of these methods rely on some form of
`(graph) centrality measure <https://en.wikipedia.org/wiki/Centrality>`_ to select central entities.
"""

import logging
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence, Union

import numpy
import torch
from class_resolver import ClassResolver, HintOrType, OptionalKwargs
from torch_ppr import page_rank

from ...triples.splitting import get_absolute_split_sizes, normalize_ratios
from ...typing import OneOrSequence
from ...utils import ExtraReprMixin

__all__ = [
    # Resolver
    "anchor_selection_resolver",
    # Base classes
    "AnchorSelection",
    "SingleSelection",
    # Concrete classes
    "DegreeAnchorSelection",
    "MixtureAnchorSelection",
    "PageRankAnchorSelection",
    "RandomAnchorSelection",
]

logger = logging.getLogger(__name__)


class AnchorSelection(ExtraReprMixin, ABC):
    """Anchor entity selection strategy."""

    def __init__(self, num_anchors: int = 32) -> None:
        """
        Initialize the strategy.

        :param num_anchors:
            the number of anchor nodes to select.
            # TODO: allow relative
        """
        self.num_anchors = num_anchors

    @abstractmethod
    def __call__(
        self,
        edge_index: numpy.ndarray,
        known_anchors: Optional[numpy.ndarray] = None,
    ) -> numpy.ndarray:
        """
        Select anchor nodes.

        .. note ::
            the number of selected anchors may be smaller than $k$, if there
            are less entities present in the edge index.

        :param edge_index: shape: (m, 2)
            the edge_index, i.e., adjacency list.

        :param known_anchors: numpy.ndarray
            an array of already known anchors for getting only unique anchors

        :return: (k,)
            the selected entity ids
        """
        raise NotImplementedError

    def iter_extra_repr(self) -> Iterable[str]:
        """Extra components for __repr__."""
        yield f"num_anchors={self.num_anchors}"

    def filter_unique(
        self,
        anchor_ranking: numpy.ndarray,
        known_anchors: Optional[numpy.ndarray],
    ) -> numpy.ndarray:
        """
        Filter out already known anchors, and select from remaining ones afterwards.

        .. note ::
            the output size may be smaller, if there are not enough candidates remaining.

        :param anchor_ranking: shape: (n,)
            the anchor node IDs sorted by preference, where the first one is the most preferrable.
        :param known_anchors: shape: (m,)
            a collection of already known anchors

        :return: shape: (m + num_anchors,)
            the extended anchors, i.e., the known ones and `num_anchors` novel ones.
        """
        if known_anchors is None:
            return anchor_ranking[: self.num_anchors]

        # isin() preserves the sorted order
        unique_anchors = anchor_ranking[~numpy.isin(anchor_ranking, known_anchors)]
        unique_anchors = unique_anchors[: self.num_anchors]
        return numpy.concatenate([known_anchors, unique_anchors])


class SingleSelection(AnchorSelection, ABC):
    """Single-step selection."""

    def __call__(
        self,
        edge_index: numpy.ndarray,
        known_anchors: Optional[numpy.ndarray] = None,
    ) -> numpy.ndarray:
        """
        Select anchor nodes.

        .. note ::
            the number of selected anchors may be smaller than $k$, if there
            are less entities present in the edge index.

        :param edge_index: shape: (m, 2)
            the edge_index, i.e., adjacency list.

        :param known_anchors: numpy.ndarray
            an array of already known anchors for getting only unique anchors

        :return: (k,)
            the selected entity ids
        """
        return self.filter_unique(anchor_ranking=self.rank(edge_index=edge_index), known_anchors=known_anchors)

    @abstractmethod
    def rank(self, edge_index: numpy.ndarray) -> numpy.ndarray:
        """
        Rank nodes.

        :param edge_index: shape: (m, 2)
            the edge_index, i.e., adjacency list.

        :return: (n,)
            the node IDs sorted decreasingly by anchor selection preference.
        """
        raise NotImplementedError


class DegreeAnchorSelection(SingleSelection):
    """Select entities according to their (undirected) degree."""

    # docstr-coverage: inherited
    def rank(self, edge_index: numpy.ndarray) -> numpy.ndarray:  # noqa: D102
        unique, counts = numpy.unique(edge_index, return_counts=True)
        # sort by decreasing degree
        ids = numpy.argsort(counts)[::-1]
        return unique[ids]


class PageRankAnchorSelection(SingleSelection):
    """
    Select entities according to their page rank.

    .. seealso::
        http://web.stanford.edu/class/cs224w/slides/04-pagerank.pdf
    """

    def __init__(
        self,
        num_anchors: int = 32,
        **kwargs,
    ) -> None:
        """
        Initialize the selection strategy.

        :param num_anchors:
            the number of anchors to select
        :param kwargs:
            additional keyword-based parameters passed to :func:`page_rank`.
        """
        super().__init__(num_anchors=num_anchors)
        self.kwargs = kwargs

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        for key, value in self.kwargs.items():
            yield f"{key}={value}"

    # docstr-coverage: inherited
    @torch.inference_mode()
    def rank(self, edge_index: numpy.ndarray) -> numpy.ndarray:  # noqa: D102
        # sort by decreasing page rank
        return numpy.argsort(page_rank(edge_index=torch.as_tensor(edge_index), **self.kwargs).cpu().numpy())[::-1]


class RandomAnchorSelection(SingleSelection):
    """Random node selection."""

    def __init__(
        self,
        num_anchors: int = 32,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the selection stragegy.

        :param num_anchors:
            the number of anchors to select
        :param random_seed:
            the random seed to use.
        """
        super().__init__(num_anchors=num_anchors)
        self.generator: numpy.random.Generator = numpy.random.default_rng(random_seed)

    # docstr-coverage: inherited
    def rank(self, edge_index: numpy.ndarray) -> numpy.ndarray:  # noqa: D102
        return self.generator.permutation(edge_index.max())


class MixtureAnchorSelection(AnchorSelection):
    """A weighted mixture of different anchor selection strategies."""

    def __init__(
        self,
        selections: Sequence[HintOrType[AnchorSelection]],
        ratios: Union[None, float, Sequence[float]] = None,
        selections_kwargs: OneOrSequence[OptionalKwargs] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the selection strategy.

        :param selections:
            the individual selections.
            For the sake of selecting unique anchors, selections will be executed in the given order
            eg, ['degree', 'pagerank'] will be executed differently from ['pagerank', 'degree']
        :param ratios:
            the ratios, cf. normalize_ratios. None means uniform ratios
        :param selections_kwargs:
            additional keyword-based arguments for the individual selection strategies
        :param kwargs:
            additional keyword-based arguments passed to AnchorSelection.__init__,
            in particular, the total number of anchors.
        """
        super().__init__(**kwargs)
        n_selections = len(selections)
        # input normalization
        if selections_kwargs is None:
            selections_kwargs = [None] * n_selections
        if ratios is None:
            ratios = numpy.ones(shape=(n_selections,)) / n_selections
        # determine absolute number of anchors for each strategy
        num_anchors = get_absolute_split_sizes(n_total=self.num_anchors, ratios=normalize_ratios(ratios=ratios))
        self.selections = [
            anchor_selection_resolver.make(selection, selection_kwargs, num_anchors=num)
            for selection, selection_kwargs, num in zip(selections, selections_kwargs, num_anchors)
        ]
        # if pre-instantiated
        for selection, num in zip(self.selections, num_anchors):
            if selection.num_anchors != num:
                logger.warning(f"{selection} had wrong number of anchors. Setting to {num}")
                selection.num_anchors = num

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"selections={self.selections}"

    # docstr-coverage: inherited
    def __call__(
        self,
        edge_index: numpy.ndarray,
        known_anchors: Optional[numpy.ndarray] = None,
    ) -> numpy.ndarray:  # noqa: D102
        anchors = known_anchors or None
        for selection in self.selections:
            anchors = selection(edge_index=edge_index, known_anchors=anchors)
        return anchors


anchor_selection_resolver: ClassResolver[AnchorSelection] = ClassResolver.from_subclasses(
    base=AnchorSelection,
    default=DegreeAnchorSelection,
    skip={SingleSelection},
)
