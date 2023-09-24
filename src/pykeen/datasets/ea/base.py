"""Base classes for entity alignment datasets."""

import logging
from abc import abstractmethod
from typing import Iterable, Optional, Tuple

import pandas
from class_resolver import HintOrType, OptionalKwargs

from .combination import GraphPairCombinator, graph_combinator_resolver
from ..base import EagerDataset
from ...triples import TriplesFactory
from ...typing import EA_SIDE_LEFT, EA_SIDES, EASide, TorchRandomHint
from ...utils import format_relative_comparison

__all__ = [
    "EADataset",
]

logger = logging.getLogger(__name__)


# TODO: support ID-only graphs


class EADataset(EagerDataset):
    """Base class for entity alignment datasets."""

    def __init__(
        self,
        *,
        side: Optional[EASide] = EA_SIDE_LEFT,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = 0,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        combination: HintOrType[GraphPairCombinator] = None,
        combination_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        """
        Initialize the dataset.

        :param side:
            the side, if only a single graph should be considered, or `None` to combine the two graphs into a
            single one, using `combination`.
        :param create_inverse_triples:
            whether to create inverse triples.
        :param random_state:
            the random state to use for reproducible splits
        :param split_ratios:
            the split ratios used to perform the train/test/validation split.
        :param combination:
            the graph combination. only effective if side is `None`
        :param combination_kwargs:
            additional keyword-based parameters for the graph combination
        :param kwargs:
            any additional keyword-based parameters are passed to :meth:`EagerDataset.__init__`.

        :raises ValueError:
            if an invalid side is passed
        """
        if side is None:
            # load both graphs
            left, right = [self._load_graph(side=side) for side in EA_SIDES]
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
            # combine
            self.combination = graph_combinator_resolver.make(combination, pos_kwargs=combination_kwargs)
            tf, self.alignment = self.combination(left=left, right=right, alignment=alignment)  # **kwargs
        elif side not in EA_SIDES:
            raise ValueError(f"side must be one of {EA_SIDES} or None")
        else:
            self.combination = self.alignment = None
            tf = self._load_graph(side=side)
        # store for repr
        self.side = side
        # split
        training, testing, validation = tf.split(ratios=split_ratios, random_state=random_state)
        # create inverse triples only for training
        training.create_inverse_triples = create_inverse_triples
        super().__init__(training=training, testing=testing, validation=validation, **kwargs)

    @abstractmethod
    def _load_graph(self, side: EASide) -> TriplesFactory:
        """Load the graph for one side."""
        raise NotImplementedError

    @abstractmethod
    def _load_alignment(self) -> pandas.DataFrame:
        """Load the entity alignment."""
        raise NotImplementedError

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"self.side={self.side}"
        yield f"self.combination={self.combination}"
