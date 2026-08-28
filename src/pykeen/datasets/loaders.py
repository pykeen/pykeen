"""Loaders turn dataset files into triples factories.

A :class:`Loader` encapsulates *how* the raw data becomes a mapping of named
:class:`~pykeen.triples.CoreTriplesFactory` instances, and is deliberately ignorant of where the files came from.
That is the job of a :class:`~pykeen.datasets.sources.Source`.

There are two recipes in use:

1. :class:`PreSplitLoader` for datasets which ship one file per split. The evaluation splits re-use the index of
   the training split so that the factories are comparable; which split inherits from which is described by a
   *plan*, cf. ``TRANSDUCTIVE_PLAN`` and ``INDUCTIVE_PLAN``.
2. :class:`AutoSplitLoader` for datasets which ship a single table of triples which needs to be split.
"""

from __future__ import annotations

import logging
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas

from .sources import Source
from ..triples import CoreTriplesFactory, TriplesFactory
from ..typing import TorchRandomHint

__all__ = [
    "SplitSpec",
    "TRANSDUCTIVE_PLAN",
    "INDUCTIVE_PLAN",
    "Loader",
    "PreSplitLoader",
    "AutoSplitLoader",
]

logger = logging.getLogger(__name__)

#: The default ratios used when a single table of triples has to be split.
DEFAULT_RATIOS: Sequence[float] = (0.8, 0.1, 0.1)


@dataclass(frozen=True)
class SplitSpec:
    """How to build one factory of a pre-split dataset."""

    #: The key of the factory whose entity index is re-used, or ``None`` to build a fresh one.
    entity_index_from: str | None = None
    #: The key of the factory whose relation index is re-used, or ``None`` to build a fresh one.
    relation_index_from: str | None = None
    #: Whether the dataset-level ``create_inverse_triples`` applies to this factory. It never applies to
    #: evaluation factories, since inverse triples are handled by the evaluation code.
    create_inverse_triples: bool = False


#: The plan for an ordinary transductive dataset: the evaluation splits share the training index.
TRANSDUCTIVE_PLAN: Mapping[str, SplitSpec] = {
    "training": SplitSpec(create_inverse_triples=True),
    "testing": SplitSpec(entity_index_from="training", relation_index_from="training"),
    "validation": SplitSpec(entity_index_from="training", relation_index_from="training"),
}

#: The plan for a fully inductive dataset: the inference graph shares only the *relations* of the transductive
#: training graph -- its entities are new -- and the evaluation splits share the inference index.
INDUCTIVE_PLAN: Mapping[str, SplitSpec] = {
    "transductive_training": SplitSpec(create_inverse_triples=True),
    "inductive_inference": SplitSpec(
        relation_index_from="transductive_training",
        create_inverse_triples=True,
    ),
    "inductive_testing": SplitSpec(
        entity_index_from="inductive_inference",
        relation_index_from="inductive_inference",
    ),
    "inductive_validation": SplitSpec(
        entity_index_from="inductive_inference",
        relation_index_from="inductive_inference",
    ),
}


class Loader(ABC):
    """A loader for the triples factories of a dataset."""

    @abstractmethod
    def load(self) -> Mapping[str, CoreTriplesFactory]:
        """Load the triples factories.

        :returns: A mapping from split name, e.g., ``"training"``, to the corresponding factory. Splits which the
            dataset does not provide are omitted.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}()"


class PreSplitLoader(Loader):
    """Load a dataset which ships one file per split."""

    def __init__(
        self,
        source: Source,
        *,
        plan: Mapping[str, SplitSpec] | None = None,
        create_inverse_triples: bool = False,
        factory_cls: type[TriplesFactory] = TriplesFactory,
        load_triples_kwargs: Mapping[str, Any] | None = None,
        factory_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the loader.

        :param source: The source providing one path per split.
        :param plan: The index-sharing plan, cf. ``TRANSDUCTIVE_PLAN``, which is also the default. It must be
            ordered such that a split occurs after the splits it inherits an index from.
        :param create_inverse_triples: Whether to create inverse triples. This only applies to the splits whose
            :attr:`SplitSpec.create_inverse_triples` is set, since inverse triples for evaluation splits are
            handled by the evaluation code.
        :param factory_cls: The triples factory class to instantiate.
        :param load_triples_kwargs: Keyword arguments to pass through to
            :meth:`~pykeen.triples.TriplesFactory.from_path` and ultimately to
            :func:`~pykeen.triples.utils.load_triples`.
        :param factory_kwargs: Additional keyword arguments for every ``from_path`` call, e.g., the path to the
            numeric literals for :class:`~pykeen.triples.TriplesNumericLiteralsFactory`.
        """
        self.source = source
        self.plan = TRANSDUCTIVE_PLAN if plan is None else plan
        self.create_inverse_triples = create_inverse_triples
        self.factory_cls = factory_cls
        self.load_triples_kwargs = load_triples_kwargs
        self.factory_kwargs = dict(factory_kwargs or {})

    # docstr-coverage: inherited
    def load(self) -> Mapping[str, CoreTriplesFactory]:  # noqa: D102
        paths = self.source.paths()
        factories: dict[str, TriplesFactory] = {}
        for key, spec in self.plan.items():
            path = paths.get(key)
            if path is None:
                logger.debug("no path for %s; skipping", key)
                continue
            index_kwargs: dict[str, Any] = {}
            if spec.entity_index_from is not None:
                index_kwargs["entity_to_id"] = factories[spec.entity_index_from].entity_to_id
            if spec.relation_index_from is not None:
                index_kwargs["relation_to_id"] = factories[spec.relation_index_from].relation_to_id
            if self.load_triples_kwargs is not None:
                # note: not all factory classes accept this parameter, so only pass it when it is set
                index_kwargs["load_triples_kwargs"] = self.load_triples_kwargs
            factories[key] = self.factory_cls.from_path(
                path=path,
                create_inverse_triples=self.create_inverse_triples and spec.create_inverse_triples,
                **self.factory_kwargs,
                **index_kwargs,
            )
        return factories

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(source={self.source})"


class AutoSplitLoader(Loader):
    """Load a single table of labeled triples and split it into training, testing, and validation."""

    def __init__(
        self,
        read_df: Callable[[], pandas.DataFrame],
        *,
        get_path: Callable[[], pathlib.Path | None] = lambda: None,
        ratios: Sequence[float] = DEFAULT_RATIOS,
        random_state: TorchRandomHint = None,
        create_inverse_triples: bool = False,
        factory_cls: type[TriplesFactory] = TriplesFactory,
    ) -> None:
        """Initialize the loader.

        :param read_df: A callable returning the table of labeled triples, with the head, relation, and tail in
            this order in the first three columns.
        :param get_path: A callable returning the path the data was read from, if any. It is only used as metadata.
        :param ratios: The training / testing / validation split ratios.
        :param random_state: The random state, to make the split reproducible.
        :param create_inverse_triples: Whether to create inverse triples for the training factory.
        :param factory_cls: The triples factory class to instantiate.
        """
        self.read_df = read_df
        self.get_path = get_path
        self.ratios = ratios
        self.random_state = random_state
        self.create_inverse_triples = create_inverse_triples
        self.factory_cls = factory_cls

    # docstr-coverage: inherited
    def load(self) -> Mapping[str, CoreTriplesFactory]:  # noqa: D102
        df = self.read_df()
        path = self.get_path()
        tf = self.factory_cls.from_labeled_triples(
            triples=df.values,
            create_inverse_triples=self.create_inverse_triples,
            metadata={"path": path} if path else None,
        )
        training, testing, validation = cast(
            tuple[TriplesFactory, TriplesFactory, TriplesFactory],
            tf.split(ratios=self.ratios, random_state=self.random_state),
        )
        logger.info("done splitting data from %s", path)
        return {"training": training, "testing": testing, "validation": validation}
