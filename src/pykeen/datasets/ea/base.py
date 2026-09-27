"""Base classes for entity alignment datasets."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import NotRequired, Unpack

import pandas as pd
from class_resolver import HintOrType, OptionalKwargs

from .combination import GraphPairCombinator, graph_combinator_resolver
from ..base import LazyDatasetKwargs, SplittingLazyDataset, SplittingLazyDatasetKwargs
from ...triples import TriplesFactory
from ...typing import EA_SIDE_LEFT, EA_SIDES, EASide
from ...utils import format_relative_comparison

__all__ = [
    "EADataset",
    "EADatasetKwargs",
]

logger = logging.getLogger(__name__)


# TODO: support ID-only graphs


class EADatasetKwargs(LazyDatasetKwargs):
    """Keyword arguments for an entity alignment dataset."""

    side: NotRequired[EASide | None]
    combination: NotRequired[HintOrType[GraphPairCombinator]]
    combination_kwargs: NotRequired[OptionalKwargs]


class EADataset(SplittingLazyDataset, ABC):
    """Base class for entity alignment datasets."""

    side: EASide | None
    combination: GraphPairCombinator

    def __init__(
        self,
        *,
        side: EASide | None = EA_SIDE_LEFT,
        combination: HintOrType[GraphPairCombinator] = None,
        combination_kwargs: OptionalKwargs = None,
        **kwargs: Unpack[SplittingLazyDatasetKwargs],
    ) -> None:
        """Initialize the dataset.

        :param side: the side, if only a single graph should be considered, or `None` to combine the two graphs into a
            single one, using `combination`.
        :param random_state: the random state to use for reproducible splits
        :param ratios: the split ratios used to perform the train/test/validation split.
        :param combination: the graph combination. only effective if side is `None`
        :param combination_kwargs: additional keyword-based parameters for the graph combination

        :raises ValueError: if an invalid side is passed
        """
        self.side = side
        self.combination = graph_combinator_resolver.make(combination, pos_kwargs=combination_kwargs)
        self.alignment = None
        super().__init__(**kwargs)

    def _get_triples_factory(self) -> TriplesFactory:
        if self.side is None:
            # load both graphs
            left, right = (self._load_graph(side=side) for side in EA_SIDES)
            # load alignment
            alignment = self._load_alignment()
            # drop duplicates
            old = alignment.shape[0]
            alignment = alignment.drop_duplicates()
            new = alignment.shape[0]
            if new < old:
                logger.info(
                    f"Dropped {format_relative_comparison(part=old - new, total=old)} alignments "
                    f"due to being duplicates.",
                )
            tf, self.alignment = self.combination(left=left, right=right, alignment=alignment)
            return tf
        if self.side in EA_SIDES:
            return self._load_graph(side=self.side)
        raise ValueError(f"side must be one of {EA_SIDES} or None")

    @abstractmethod
    def _load_graph(self, side: EASide) -> TriplesFactory:
        """Load the graph for one side."""

    @abstractmethod
    def _load_alignment(self) -> pd.DataFrame:
        """Load the entity alignment."""

    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"self.side={self.side}"
        if self.side is None:
            yield "self.combination=None"
        else:
            yield f"self.combination={self.combination}"
