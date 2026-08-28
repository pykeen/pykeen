"""Utility classes for constructing datasets.

The dataset classes here are deliberately thin: a dataset is a *named set of triples factories*, and everything
below that is composed rather than inherited. Where the files come from is a
:class:`~pykeen.datasets.sources.Source`, and how they become factories is a
:class:`~pykeen.datasets.loaders.Loader`. The classes in this module pick a source and a loader and give the
combination a name, so that a new dataset is usually a five-line subclass.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable, Collection, Iterable, Mapping, MutableMapping, Sequence
from typing import Any, ClassVar, cast

import click
import docdata
import pandas as pd
import torch
from more_click import verbose_option
from pystow.utils import download, name_from_url
from tabulate import tabulate
from typing_extensions import Self

from .loaders import DEFAULT_RATIOS, AutoSplitLoader, Loader, PreSplitLoader, SplitSpec
from .sources import ArchiveSource, LocalSource, RemoteSource, Source, TarArchiveSource, ZipArchiveSource
from ..constants import PYKEEN_DATASETS
from ..triples import CoreTriplesFactory, TriplesFactory
from ..triples.deteriorate import deteriorate
from ..triples.remix import remix
from ..triples.triples_factory import splits_similarity
from ..typing import MappedTriples, TorchRandomHint
from ..utils import ExtraReprMixin, format_relative_comparison, normalize_path, normalize_string

__all__ = [
    # Base classes
    "DatasetBase",
    "LazyFactoryMixin",
    "Dataset",
    "EagerDataset",
    "LazyDataset",
    "PathDataset",
    "RemoteDataset",
    "UnpackedRemoteDataset",
    "TarFileRemoteDataset",
    "PackedZipRemoteDataset",
    "CompressedSingleDataset",
    "TarFileSingleDataset",
    "ZipSingleDataset",
    "TabbedDataset",
    "SingleTabbedDataset",
    # Utilities
    "dataset_similarity",
]

logger = logging.getLogger(__name__)


def dataset_similarity(a: Dataset, b: Dataset, metric: str | None = None) -> float:
    """Calculate the similarity between two datasets.

    :param a: The reference dataset
    :param b: The target dataset
    :param metric: The similarity metric to use. Defaults to `tanimoto`. Could either be a symmetric or asymmetric
        metric.

    :returns: A scalar value between 0 and 1 where closer to 1 means the datasets are more similar based on the metric.

    :raises ValueError: if an invalid metric type is passed. Right now, there's only `tanimoto`, but this could change
        in later.
    """
    if metric == "tanimoto" or metric is None:
        return splits_similarity(a._tup(), b._tup())
    raise ValueError(f"invalid metric: {metric}")


def _map_ids(x: torch.Tensor, kept_old_ids: torch.Tensor) -> torch.Tensor:
    """Vectorized re-mapping of ids."""
    # note: this needs `O(old_max_id)` memory.
    # note: this is quite similar to pykeen.triples.triples_factory._map_triples_elements_to_ids
    old_max_id = int(x.max())
    new_max_id = len(kept_old_ids)
    map_t = torch.full(size=(old_max_id + 1,), fill_value=-1)
    map_t[kept_old_ids] = torch.arange(new_max_id)
    return map_t[x]


def _filter_mapped_triples(
    mapped_triples: MappedTriples,
    kept_old_entity_ids_t: torch.Tensor,
    kept_old_relation_ids_t: torch.Tensor,
) -> MappedTriples:
    heads, tails = _map_ids(mapped_triples[:, ::2], kept_old_ids=kept_old_entity_ids_t).unbind(dim=-1)
    relations = _map_ids(mapped_triples[:, 1], kept_old_ids=kept_old_relation_ids_t)
    mapped_triples = cast(MappedTriples, torch.stack([heads, relations, tails], dim=-1))
    # We can only keep triples where none of the IDs have been filtered.
    keep_mask = (mapped_triples >= 0).all(dim=-1)
    logger.info(f"keeping {format_relative_comparison(keep_mask.sum().item(), keep_mask.numel())} triples.")
    return mapped_triples[keep_mask]


def _update_eval_triples_factory(
    factory: TriplesFactory,
    kept_old_entity_ids_t: torch.Tensor,
    kept_old_relation_ids_t: torch.Tensor,
    entity_to_id: Mapping[str, int],
    relation_to_id: Mapping[str, int],
) -> TriplesFactory:
    mapped_triples = _filter_mapped_triples(
        mapped_triples=factory.mapped_triples,
        kept_old_entity_ids_t=kept_old_entity_ids_t,
        kept_old_relation_ids_t=kept_old_relation_ids_t,
    )
    return TriplesFactory(
        mapped_triples=mapped_triples,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        create_inverse_triples=factory.create_inverse_triples,
        metadata=factory.metadata,
        num_entities=len(kept_old_entity_ids_t),
        num_relations=len(kept_old_relation_ids_t),
    )


def _update_eval_core_factory(
    factory: CoreTriplesFactory, kept_old_entity_ids_t: torch.Tensor, kept_old_relation_ids_t: torch.Tensor
) -> CoreTriplesFactory:
    mapped_triples = _filter_mapped_triples(
        mapped_triples=factory.mapped_triples,
        kept_old_entity_ids_t=kept_old_entity_ids_t,
        kept_old_relation_ids_t=kept_old_relation_ids_t,
    )
    return CoreTriplesFactory(
        mapped_triples=mapped_triples,
        create_inverse_triples=factory.create_inverse_triples,
        metadata=factory.metadata,
        num_entities=len(kept_old_entity_ids_t),
        num_relations=len(kept_old_relation_ids_t),
    )


def _restrict_mapping(id_to_label: Mapping[int, str], kept_ids: Sequence[int]) -> Mapping[str, int]:
    return {id_to_label[old_id]: new_id for new_id, old_id in enumerate(kept_ids)}


def _reorder_columns(df: pd.DataFrame, usecols: Sequence[Any] | None) -> pd.DataFrame:
    """Restore the column order requested via ``usecols``, which :func:`pandas.read_csv` does not honor."""
    if usecols is None:
        return df
    logger.info("reordering columns: %s", usecols)
    return df[usecols]


class DatasetBase(ExtraReprMixin):
    """A named collection of triples factories.

    This holds everything which does not depend on *which* splits a dataset has, so that the transductive
    :class:`Dataset` and the fully inductive :class:`~pykeen.datasets.inductive.base.InductiveDataset` can share
    it instead of maintaining two copies. Subclasses declare their splits via ``_factory_keys``.
    """

    #: The names of the factories, in canonical order. The first one is the *reference* factory, whose entity and
    #: relation index the dataset reports and from which examples are shown in the summary.
    _factory_keys: ClassVar[tuple[str, ...]]

    #: Additional metadata to store inside the dataset
    metadata: Mapping[str, Any] | None = None

    metadata_file_name: ClassVar[str] = "metadata.pth"
    triples_factory_cls: ClassVar[type[CoreTriplesFactory]] = TriplesFactory

    @classmethod
    def _eager_cls(cls) -> type[DatasetBase]:
        """Return the eager counterpart of this dataset family, used when loading from a binary directory."""
        raise NotImplementedError

    @property
    def factory_dict(self) -> Mapping[str, CoreTriplesFactory]:
        """Return a dictionary of the factories, keyed by split name, omitting absent optional splits.

        .. note::

            Lazy datasets override this to *be* the primitive, cf. :class:`LazyFactoryMixin`, and expose the
            individual splits as properties reading from it.
        """
        return {key: factory for key in self._factory_keys if (factory := getattr(self, key, None)) is not None}

    @property
    def _reference_factory(self) -> CoreTriplesFactory:
        return self.factory_dict[self._factory_keys[0]]

    def __eq__(self, other: object) -> bool:  # noqa: D105
        return (
            isinstance(other, DatasetBase)
            and self._factory_keys == other._factory_keys
            and self.factory_dict == other.factory_dict
            and self.create_inverse_triples == other.create_inverse_triples
        )

    @property
    def entity_to_id(self):  # noqa: D401
        """The mapping of entity labels to IDs."""
        factory = self._reference_factory
        if not isinstance(factory, TriplesFactory):
            raise AttributeError(f"{factory.__class__} does not have labeling information.")
        return factory.entity_to_id

    @property
    def relation_to_id(self):  # noqa: D401
        """The mapping of relation labels to IDs."""
        factory = self._reference_factory
        if not isinstance(factory, TriplesFactory):
            raise AttributeError(f"{factory.__class__} does not have labeling information.")
        return factory.relation_to_id

    @property
    def num_entities(self):  # noqa: D401
        """The number of entities."""
        return self._reference_factory.num_entities

    @property
    def num_relations(self):  # noqa: D401
        """The number of relations."""
        return self._reference_factory.num_relations

    @property
    def create_inverse_triples(self):
        """Return whether inverse triples are created *for the reference factory*."""
        return self._reference_factory.create_inverse_triples

    @classmethod
    def docdata(cls, *parts: str) -> Any:
        """Get docdata for this class."""
        rv = docdata.get_docdata(cls)
        for part in parts:
            rv = rv[part]
        return rv

    @staticmethod
    def triples_sort_key(cls: type[DatasetBase]) -> int:
        """Get the number of triples for sorting."""
        return cls.docdata("statistics", "triples")

    @classmethod
    def triples_pair_sort_key(cls, pair: tuple[str, type[DatasetBase]]) -> int:
        """Get the number of triples for sorting in an iterator context."""
        return cls.triples_sort_key(pair[1])

    def _summary_rows(self):
        return [
            (key.replace("_", " ").title(), factory.num_entities, factory.num_relations, factory.num_triples)
            for key, factory in self.factory_dict.items()
        ]

    def summary_str(self, title: str | None = None, show_examples: int | None = 5, end="\n") -> str:
        """Make a summary string of all of the factories."""
        rows = self._summary_rows()
        n_triples = sum(count for *_, count in rows)
        rows.append(("Total", "-", "-", n_triples))
        t = tabulate(rows, headers=["Name", "Entities", "Relations", "Triples"])
        rv = f"{title or self.__class__.__name__} (create_inverse_triples={self.create_inverse_triples})\n{t}"
        if show_examples:
            factory = self._reference_factory
            if not isinstance(factory, TriplesFactory):
                raise AttributeError(f"{factory.__class__} does not have labeling information.")
            examples = tabulate(
                factory.label_triples(factory.mapped_triples[:show_examples]),
                headers=["Head", "Relation", "tail"],
            )
            rv += "\n" + examples
        return rv + end

    def summarize(self, title: str | None = None, show_examples: int | None = 5, file=None) -> None:
        """Print a summary of the dataset."""
        print(self.summary_str(title=title, show_examples=show_examples), file=file)  # noqa:T201

    def iter_extra_repr(self) -> Iterable[str]:
        """Yield extra entries for the instance's string representation."""
        yield f"num_entities={self.num_entities}"
        yield f"num_relations={self.num_relations}"
        yield f"create_inverse_triples={self.create_inverse_triples}"

    @classmethod
    def from_directory_binary(cls, path: str | pathlib.Path) -> DatasetBase:
        """Load a dataset from a directory.

        :param path: The directory a dataset was stored to with
            :meth:`~pykeen.datasets.base.DatasetBase.to_directory_binary`.

        :returns: An eager dataset with the stored factories.

        :raises NotADirectoryError: If the path does not refer to a directory.
        """
        path = pathlib.Path(path)

        if not path.is_dir():
            raise NotADirectoryError(path)

        tfs = {}
        for key in cls._factory_keys:
            tf_path = path.joinpath(key)
            if tf_path.is_dir():
                tfs[key] = cls.triples_factory_cls.from_path_binary(path=tf_path)
            else:
                logger.warning(f"{tf_path.as_uri()} does not exist.")
        metadata_path = path.joinpath(cls.metadata_file_name)
        # TODO: consider restricting metadata to JSON
        metadata = torch.load(metadata_path, weights_only=False) if metadata_path.is_file() else None
        eager_cls = cast("Callable[..., DatasetBase]", cls._eager_cls())
        return eager_cls(**tfs, metadata=metadata)

    def to_directory_binary(self, path: str | pathlib.Path) -> None:
        """Store a dataset to a path in binary format."""
        path = pathlib.Path(path)
        for key, factory in self.factory_dict.items():
            tf_path = path.joinpath(key)
            factory.to_path_binary(tf_path)
            logger.info(f"Stored {key} factory to {tf_path.as_uri()}")
        metadata = dict(self.metadata or {})
        metadata.setdefault("name", self.get_normalized_name())
        torch.save(metadata, path.joinpath(self.metadata_file_name))

    @classmethod
    def cli(cls) -> None:
        """Run the CLI."""

        @click.command(help=f"{cls.__name__} Dataset CLI.")
        @verbose_option
        def main():
            """Run the dataset CLI."""
            click.secho(f"Loading {cls.__name__}", fg="green", bold=True)
            click.echo(cls().summary_str())

        main()

    def get_normalized_name(self) -> str:
        """Get the normalized name of the dataset."""
        return normalize_string((self.metadata or {}).get("name") or self.__class__.__name__)


class LazyFactoryMixin:
    """Lazily materialize a mapping of named triples factories.

    This is the single lazy-loading primitive: subclasses implement ``_load_factories``, which is called at
    most once, and everything else goes through :attr:`factory_dict`.
    """

    #: The loaded factories, or ``None`` if they have not been loaded yet
    _factories: MutableMapping[str, CoreTriplesFactory] | None = None
    #: The directory in which the cached data is stored
    cache_root: pathlib.Path
    #: The loader used to materialize the factories
    loader: Loader | None = None

    def __init__(
        self,
        loader: Loader | None = None,
        *,
        eager: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the lazy dataset.

        :param loader: The loader producing the triples factories. May be ``None`` for subclasses which override
            ``_load_factories`` instead.
        :param eager: Whether to load the data immediately rather than on first access.
        :param metadata: Additional metadata to store inside the dataset.
        """
        self.loader = loader
        self.metadata = metadata
        if eager:
            _ = self.factory_dict

    def _load_factories(self) -> Mapping[str, CoreTriplesFactory]:
        """Load all triples factories of this dataset.

        :returns: A mapping from split name to factory, omitting splits the dataset does not provide.

        :raises NotImplementedError: If there is neither a loader nor an override of this method.
        """
        if self.loader is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} has neither a loader nor an implementation of `_load_factories`."
            )
        return self.loader.load()

    @property
    def _loaded(self) -> bool:
        """Whether the factories have already been loaded."""
        return self._factories is not None

    @property
    def factory_dict(self) -> Mapping[str, CoreTriplesFactory]:
        """Return the triples factories, loading them on first access."""
        if self._factories is None:
            self._factories = dict(self._load_factories())
        return self._factories

    def _help_cache(self, cache_root: None | str | pathlib.Path) -> pathlib.Path:
        """Get the appropriate cache root directory.

        :param cache_root: If none is passed, defaults to a subfolder of the PyKEEN home directory defined in
            :data:`~pykeen.constants.PYKEEN_HOME`. The subfolder is named based on the class inheriting from
            :class:`~pykeen.datasets.base.Dataset`.

        :returns: A path object for the calculated cache root directory
        """
        cache_root = normalize_path(cache_root, *self._cache_sub_directories(), mkdir=True, default=PYKEEN_DATASETS)
        logger.debug("using cache root at %s", cache_root.as_uri())
        return cache_root

    def _cache_sub_directories(self) -> Iterable[str]:
        """Iterate over appropriate cache sub-directory."""
        # TODO: use class-resolver normalize?
        yield self.__class__.__name__.lower()


class Dataset(DatasetBase):
    """The base dataset class."""

    _factory_keys: ClassVar[tuple[str, ...]] = ("training", "testing", "validation")

    #: A factory wrapping the training triples
    training: CoreTriplesFactory
    #: A factory wrapping the testing triples, that share indices with the training triples
    testing: CoreTriplesFactory
    #: A factory wrapping the validation triples, that share indices with the training triples
    validation: CoreTriplesFactory | None

    # docstr-coverage: inherited
    @classmethod
    def _eager_cls(cls) -> type[DatasetBase]:  # noqa: D102
        return EagerDataset

    # docstr-coverage: inherited
    @classmethod
    def from_directory_binary(cls, path: str | pathlib.Path) -> Dataset:  # noqa: D102
        return cast(Dataset, super().from_directory_binary(path))

    @classmethod
    def from_path(cls, path: str | pathlib.Path, ratios: list[float] | None = None) -> Dataset:
        """Create a dataset from a single triples factory by splitting it in 3."""
        tf = TriplesFactory.from_path(path=path)
        return cls.from_tf(tf=tf, ratios=ratios)

    @staticmethod
    def from_tf(tf: TriplesFactory, ratios: list[float] | None = None) -> Dataset:
        """Create a dataset from a single triples factory by splitting it in 3."""
        training, testing, validation = cast(
            tuple[TriplesFactory, TriplesFactory, TriplesFactory],
            tf.split(ratios or [0.8, 0.1, 0.1]),
        )
        return EagerDataset(training=training, testing=testing, validation=validation)

    def remix(self, random_state: TorchRandomHint = None, **kwargs) -> Dataset:
        """Remix a dataset using :func:`~pykeen.triples.remix.remix`."""
        return EagerDataset(
            *remix(
                *self._tup(),
                random_state=random_state,
                **kwargs,
            ),
        )

    def deteriorate(self, n: int | float, random_state: TorchRandomHint = None) -> Dataset:
        """Deteriorate n triples from the dataset's training with :func:`~pykeen.triples.deteriorate.deteriorate`."""
        return EagerDataset(
            *deteriorate(
                *self._tup(),
                n=n,
                random_state=random_state,
            )
        )

    def similarity(self, other: Dataset, metric: str | None = None) -> float:
        """Compute the similarity between two shuffles of the same dataset.

        :param other: The other shuffling of the dataset
        :param metric: The metric to use. Defaults to `tanimoto`.

        :returns: A float of the similarity

        .. seealso::

            :func:`~pykeen.triples.splits_similarity`.
        """
        return dataset_similarity(self, other, metric=metric)

    def _tup(self):
        if self.validation is None:
            return self.training, self.testing
        return self.training, self.testing, self.validation

    def restrict(
        self,
        entities: None | Collection[int] | Collection[str] = None,
        relations: None | Collection[int] | Collection[str] = None,
        invert_entity_selection: bool = False,
        invert_relation_selection: bool = False,
    ) -> EagerDataset | Self:
        """Restrict a dataset to the given entities/relations.

        >>> from pykeen.datasets import get_dataset
        >>> full_dataset = get_dataset(dataset="nations")
        >>> restricted_dataset = full_dataset.restrict(entities={"burma", "china", "india", "indonesia"})

        :param entities: The entities to keep (or discard, cf. `invert_entity_selection`). `None` corresponds to
            selecting all entities (but is handled more efficiently).
        :param relations: The relations to keep (or discard, cf. `invert_relation_selection`). `None` corresponds to
            selecting all relations (but is handled more efficiently).
        :param invert_entity_selection: Whether to invert the entity selection, i.e., discard the selected entities
            rather than all remaining ones.
        :param invert_relation_selection: Whether to invert the relation selection, i.e., discard the selected relations
            rather than all remaining ones.

        :returns: a new dataset with different entity and relation mappins and a restricted set of triples.

        .. warning::

            This is different to :meth:`~pykeen.triples.CoreTriplesFactory.new_with_restriction` as it
            does modify the label to id mapping.
        """
        # early termination for simple case
        if entities is None and relations is None:
            return self

        # restrict triples factories (without modifying the entity to id mapping)
        training = self.training.new_with_restriction(
            entities=entities,
            relations=relations,
            invert_entity_selection=invert_entity_selection,
            invert_relation_selection=invert_relation_selection,
        )

        # collapse entity and relation ids
        kept_entity_ids_t, entity_ids_inv_t = training.mapped_triples[:, 0::2].unique(return_inverse=True)
        kept_relation_ids_t, relation_ids_inv_t = training.mapped_triples[:, 1].unique(return_inverse=True)
        num_entities = len(kept_entity_ids_t)
        num_relations = len(kept_relation_ids_t)
        new_training_triples = torch.stack([entity_ids_inv_t[:, 0], relation_ids_inv_t, entity_ids_inv_t[:, 1]], dim=-1)

        # help mypy
        testing: CoreTriplesFactory
        validation: CoreTriplesFactory | None
        # update factories
        if isinstance(training, TriplesFactory):
            assert isinstance(self.testing, TriplesFactory)
            assert self.validation is None or isinstance(self.validation, TriplesFactory)
            entity_to_id = _restrict_mapping(
                id_to_label=training.entity_id_to_label, kept_ids=kept_entity_ids_t.tolist()
            )
            relation_to_id = _restrict_mapping(
                id_to_label=training.relation_id_to_label, kept_ids=kept_relation_ids_t.tolist()
            )
            training = TriplesFactory(
                mapped_triples=cast(MappedTriples, new_training_triples),
                entity_to_id=entity_to_id,
                relation_to_id=relation_to_id,
                create_inverse_triples=training.create_inverse_triples,
                metadata=training.metadata,
                num_entities=num_entities,
                num_relations=num_relations,
            )
            # also update testing and validation
            testing = _update_eval_triples_factory(
                factory=self.testing,
                kept_old_entity_ids_t=kept_entity_ids_t,
                kept_old_relation_ids_t=kept_relation_ids_t,
                entity_to_id=entity_to_id,
                relation_to_id=relation_to_id,
            )
            validation = (
                None
                if self.validation is None
                else _update_eval_triples_factory(
                    factory=self.validation,
                    kept_old_entity_ids_t=kept_entity_ids_t,
                    kept_old_relation_ids_t=kept_relation_ids_t,
                    entity_to_id=entity_to_id,
                    relation_to_id=relation_to_id,
                )
            )
        else:
            training = CoreTriplesFactory(
                mapped_triples=cast(MappedTriples, new_training_triples),
                create_inverse_triples=training.create_inverse_triples,
                metadata=training.metadata,
                num_entities=num_entities,
                num_relations=num_relations,
            )
            testing = _update_eval_core_factory(
                factory=self.testing,
                kept_old_entity_ids_t=kept_entity_ids_t,
                kept_old_relation_ids_t=kept_relation_ids_t,
            )
            validation = (
                None
                if self.validation is None
                else _update_eval_core_factory(
                    factory=self.validation,
                    kept_old_entity_ids_t=kept_entity_ids_t,
                    kept_old_relation_ids_t=kept_relation_ids_t,
                )
            )

        # update metadata
        metadata = dict(self.metadata or {})
        restriction_meta = {"base": metadata.pop("name", None) or self.get_normalized_name()}
        if entities:
            # note:
            # - we convert to list to make sure that the metadata is JSON-serializable
            # - we sort because the order does not matter for the functionality of this method
            restriction_meta |= {"entities": sorted(entities), "invert_entity_selection": invert_entity_selection}
        if relations:
            restriction_meta |= {"relations": sorted(relations), "invert_relation_selection": invert_relation_selection}
        metadata["restriction"] = restriction_meta

        # compose restricted dataset
        return EagerDataset(training=training, testing=testing, validation=validation, metadata=metadata)

    def merged(self) -> CoreTriplesFactory:
        """Return a single triples factory with all triples."""
        training, *rest = self._tup()
        return training.merge(*rest)


class EagerDataset(Dataset):
    """A dataset whose training, testing, and optional validation factories are pre-loaded."""

    def __init__(
        self,
        training: CoreTriplesFactory,
        testing: CoreTriplesFactory,
        validation: CoreTriplesFactory | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the eager dataset.

        :param training: A pre-defined triples factory with training triples
        :param testing: A pre-defined triples factory with testing triples
        :param validation: A pre-defined triples factory with validation triples
        :param metadata: additional metadata to store inside the dataset
        """
        self.training = training
        self.testing = testing
        self.validation = validation
        self.metadata = metadata

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"metadata={self.metadata}"


class LazyDataset(LazyFactoryMixin, Dataset):
    """A dataset whose training, testing, and optional validation factories are lazily loaded.

    The loading itself is delegated to a :class:`~pykeen.datasets.loaders.Loader`. Subclasses which cannot express
    their loading that way may instead override ``_load_factories``.
    """

    @property
    def training(self) -> TriplesFactory:  # type: ignore[override]  # noqa: D401
        """The training triples factory."""
        return cast(TriplesFactory, self.factory_dict["training"])

    @property
    def testing(self) -> TriplesFactory:  # type: ignore[override]  # noqa: D401
        """The testing triples factory that shares indices with the training triples factory."""
        return cast(TriplesFactory, self.factory_dict["testing"])

    @property
    def validation(self) -> TriplesFactory | None:  # type: ignore[override]  # noqa: D401
        """The validation triples factory that shares indices with the training triples factory."""
        return cast("TriplesFactory | None", self.factory_dict.get("validation"))


class PathDataset(LazyDataset):
    """A dataset which stores each split in its own file.

    The files may be local, downloaded one-by-one, or extracted from an archive; that choice is made by the
    :class:`~pykeen.datasets.sources.Source` and does not affect the loading itself.
    """

    def __init__(
        self,
        training_path: None | str | pathlib.Path = None,
        testing_path: None | str | pathlib.Path = None,
        validation_path: None | str | pathlib.Path = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        load_triples_kwargs: Mapping[str, Any] | None = None,
        *,
        source: Source | None = None,
        plan: Mapping[str, SplitSpec] | None = None,
        factory_cls: type[TriplesFactory] | None = None,
        factory_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the dataset.

        :param training_path: Path to the training triples file or training triples file.
        :param testing_path: Path to the testing triples file or testing triples file.
        :param validation_path: Path to the validation triples file or validation triples file.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param load_triples_kwargs: Arguments to pass through to :func:`~pykeen.triples.TriplesFactory.from_path`
            and ultimately through to :func:`~pykeen.triples.utils.load_triples`.
        :param source: An explicit source for the files, as an alternative to the three path parameters. Used by
            the subclasses which download their files.
        :param plan: The index-sharing plan, cf. ``pykeen.datasets.loaders.TRANSDUCTIVE_PLAN``, which is also the
            default.
        :param factory_cls: The triples factory class. Defaults to ``triples_factory_cls``.
        :param factory_kwargs: Additional keyword-based arguments for every triples factory.

        :raises ValueError: If neither a source nor a training and testing path are given.
        """
        if source is None:
            if training_path is None or testing_path is None:
                raise ValueError("must give either a source, or both a training_path and a testing_path")
            source = LocalSource(training=training_path, testing=testing_path, validation=validation_path)
        self.source = source
        self.load_triples_kwargs = load_triples_kwargs
        super().__init__(
            loader=PreSplitLoader(
                source=source,
                plan=plan,
                create_inverse_triples=create_inverse_triples,
                factory_cls=factory_cls or cast(type[TriplesFactory], self.triples_factory_cls),
                load_triples_kwargs=load_triples_kwargs,
                factory_kwargs=factory_kwargs,
            ),
            eager=eager,
        )

    @property
    def training_path(self) -> pathlib.Path | None:
        """The path of the training triples file."""
        return self.source.expected_paths().get("training")

    @property
    def testing_path(self) -> pathlib.Path | None:
        """The path of the testing triples file."""
        return self.source.expected_paths().get("testing")

    @property
    def validation_path(self) -> pathlib.Path | None:
        """The path of the validation triples file, if any."""
        return self.source.expected_paths().get("validation")

    def __repr__(self) -> str:  # noqa: D105
        return (
            f'{self.__class__.__name__}(training_path="{self.training_path}", testing_path="{self.testing_path}",'
            f' validation_path="{self.validation_path}")'
        )


class UnpackedRemoteDataset(PathDataset):
    """A dataset with all three of train, test, and validation sets as URLs."""

    def __init__(
        self,
        training_url: str,
        testing_url: str,
        validation_url: str,
        cache_root: str | None = None,
        force: bool = False,
        eager: bool = False,
        create_inverse_triples: bool = False,
        load_triples_kwargs: Mapping[str, Any] | None = None,
        download_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize dataset.

        :param training_url: The URL of the training file
        :param testing_url: The URL of the testing file
        :param validation_url: The URL of the validation file
        :param cache_root: An optional directory to store the extracted files. Is none is given, the default PyKEEN
            directory is used. This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to
            ``~/.data/pykeen``.
        :param force: If true, redownload any cached files
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param load_triples_kwargs: Arguments to pass through to :func:`~pykeen.triples.TriplesFactory.from_path`
            and ultimately through to :func:`~pykeen.triples.utils.load_triples`.
        :param download_kwargs: Keyword arguments to pass to :func:`pystow.utils.download`
        """
        self.cache_root = self._help_cache(cache_root)

        self.training_url = training_url
        self.testing_url = testing_url
        self.validation_url = validation_url

        super().__init__(
            source=RemoteSource(
                urls={"training": training_url, "testing": testing_url, "validation": validation_url},
                cache_root=self.cache_root,
                force=force,
                download_kwargs=download_kwargs,
            ),
            eager=eager,
            create_inverse_triples=create_inverse_triples,
            load_triples_kwargs=load_triples_kwargs,
        )


class RemoteDataset(PathDataset):
    """A dataset whose splits are packed into a single remote archive."""

    #: The source class used to unpack the archive
    source_cls: ClassVar[type[ArchiveSource]]
    #: Whether the whole archive is unpacked, rather than only the requested members
    extract_all: ClassVar[bool] = True

    def __init__(
        self,
        url: str,
        relative_training_path: str | pathlib.PurePath,
        relative_testing_path: str | pathlib.PurePath,
        relative_validation_path: str | pathlib.PurePath,
        cache_root: str | None = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        timeout=None,
    ) -> None:
        """Initialize dataset.

        :param url: The url where to download the dataset from.
        :param relative_training_path: The path inside the cache root where the training path gets extracted
        :param relative_testing_path: The path inside the cache root where the testing path gets extracted
        :param relative_validation_path: The path inside the cache root where the validation path gets extracted
        :param cache_root: An optional directory to store the extracted files. Is none is given, the default PyKEEN
            directory is used. This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to
            ``~/.data/pykeen``.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param timeout: The timeout number of seconds for waiting to download the dataset. Defaults to 60.
        """
        self.cache_root = self._help_cache(cache_root)

        self.url = url
        self.timeout = timeout if timeout is not None else 60
        self._relative_training_path = pathlib.PurePath(relative_training_path)
        self._relative_testing_path = pathlib.PurePath(relative_testing_path)
        self._relative_validation_path = pathlib.PurePath(relative_validation_path)

        # note: requests cannot handle file:// URLs, which are convenient for testing
        download_kwargs: Mapping[str, Any] = (
            {"backend": "requests", "timeout": self.timeout}
            if url.startswith(("http://", "https://"))
            else {"backend": "urllib"}
        )
        super().__init__(
            source=self.source_cls(
                members={
                    "training": self._relative_training_path,
                    "testing": self._relative_testing_path,
                    "validation": self._relative_validation_path,
                },
                cache_root=self.cache_root,
                url=url,
                extract_all=self.extract_all,
                download_kwargs=download_kwargs,
            ),
            eager=eager,
            create_inverse_triples=create_inverse_triples,
        )

    def _get_paths(self) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:  # noqa: D401
        """Get the paths where the extracted files can be found."""
        paths = self.source.expected_paths()
        return paths["training"], paths["testing"], paths["validation"]


class TarFileRemoteDataset(RemoteDataset):
    """A remote dataset stored as a tar file."""

    source_cls = TarArchiveSource


class PackedZipRemoteDataset(PathDataset):
    """A dataset whose splits are packed into a single remote zip archive."""

    head_column: int = 0
    relation_column: int = 1
    tail_column: int = 2
    sep = "\t"
    header = None

    def __init__(
        self,
        relative_training_path: str | pathlib.PurePath,
        relative_testing_path: str | pathlib.PurePath,
        relative_validation_path: str | pathlib.PurePath,
        url: str | None = None,
        name: str | None = None,
        cache_root: str | None = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
    ) -> None:
        """Initialize dataset.

        :param relative_training_path: The path inside the zip file for the training data
        :param relative_testing_path: The path inside the zip file for the testing data
        :param relative_validation_path: The path inside the zip file for the validation data
        :param url: The url where to download the dataset from
        :param name: The name of the file. If not given, tries to get the name from the end of the URL
        :param cache_root: An optional directory to store the extracted files. Is none is given, the default PyKEEN
            directory is used. This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to
            ``~/.pykeen``.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.

        :raises ValueError: if the ``header`` class attribute was overridden, or if there's no URL specified and
            there is no data already at the calculated path
        """
        if self.header is not None:
            raise ValueError(f"{self.__class__.__name__} does not support a header row; got header={self.header}")

        self.cache_root = self._help_cache(cache_root)

        self.relative_training_path = pathlib.PurePath(relative_training_path)
        self.relative_testing_path = pathlib.PurePath(relative_testing_path)
        self.relative_validation_path = pathlib.PurePath(relative_validation_path)

        self.url = url
        self.name = name or (name_from_url(url) if url else None)
        if self.name is None:
            raise ValueError("must give at least one of name or URL")
        self.path = self.cache_root.joinpath(self.name)
        logger.debug("file path at %s", self.path)
        if not self.path.is_file() and not self.url:
            raise ValueError(f"must specify url to download from since path does not exist: {self.path}")

        super().__init__(
            source=ZipArchiveSource(
                members={
                    "training": self.relative_training_path,
                    "testing": self.relative_testing_path,
                    "validation": self.relative_validation_path,
                },
                cache_root=self.cache_root,
                url=url,
                name=self.name,
            ),
            eager=eager,
            create_inverse_triples=create_inverse_triples,
            load_triples_kwargs={
                "delimiter": self.sep,
                "column_remapping": (self.head_column, self.relation_column, self.tail_column),
            },
        )


class TabbedDataset(LazyDataset):
    """A dataset which ships a single table of triples and gets split automatically."""

    ratios: ClassVar[Sequence[float]] = DEFAULT_RATIOS

    def __init__(
        self,
        cache_root: str | None = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = None,
        *,
        factory_cls: type[TriplesFactory] | None = None,
    ):
        """Initialize dataset.

        :param cache_root: An optional directory to store the extracted files. Is none is given, the default PyKEEN
            directory is used. This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to
            ``~/.pykeen``.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param random_state: An optional random state to make the training/testing/validation split reproducible.
        :param factory_cls: The triples factory class. Defaults to ``triples_factory_cls``.
        """
        self.cache_root = self._help_cache(cache_root)
        self.random_state = random_state

        super().__init__(
            loader=AutoSplitLoader(
                # note: bound methods, so that subclasses overriding the hooks still take effect
                read_df=self._get_df,
                get_path=self._get_path,
                ratios=self.ratios,
                random_state=random_state,
                create_inverse_triples=create_inverse_triples,
                factory_cls=factory_cls or cast(type[TriplesFactory], self.triples_factory_cls),
            ),
            eager=eager,
        )

    def _get_path(self) -> pathlib.Path | None:
        """Get the path of the data if there's a single file."""
        return None

    def _get_df(self) -> pd.DataFrame:
        """Get the dataframe of labeled triples."""
        raise NotImplementedError


class SingleTabbedDataset(TabbedDataset):
    """A dataset which ships a single, downloaded table of triples and gets split automatically."""

    #: URL to the data to download
    url: str

    def __init__(
        self,
        url: str,
        name: str | None = None,
        cache_root: str | None = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = None,
        download_kwargs: dict[str, Any] | None = None,
        read_csv_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize dataset.

        :param url: The url where to download the dataset from
        :param name: The name of the file. If not given, tries to get the name from the end of the URL
        :param cache_root: An optional directory to store the extracted files. Is none is given, the default PyKEEN
            directory is used. This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to
            ``~/.pykeen``.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param random_state: An optional random state to make the training/testing/validation split reproducible.
        :param download_kwargs: Keyword arguments to pass through to :func:`pystow.utils.download`.
        :param read_csv_kwargs: Keyword arguments to pass through to :func:`pandas.read_csv`.
        """
        # note: these are set before `super().__init__`, since an eager load reads them
        self.url = url
        self.name = name or name_from_url(url)
        self.download_kwargs = download_kwargs or {}
        self.read_csv_kwargs = dict(read_csv_kwargs or {})
        self.read_csv_kwargs.setdefault("sep", "\t")

        super().__init__(
            cache_root=cache_root,
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            eager=eager,
        )

    # docstr-coverage: inherited
    def _get_path(self) -> pathlib.Path:  # noqa: D102
        return self.cache_root.joinpath(self.name)

    # docstr-coverage: inherited
    def _get_df(self) -> pd.DataFrame:  # noqa: D102
        path = self._get_path()
        if not path.is_file():
            if not self.url:
                raise ValueError(f"must specify url to download from since path does not exist: {path}")
            logger.info("downloading data from %s to %s", self.url, path)
            download(url=self.url, path=path, **self.download_kwargs)  # noqa:S310
        df = pd.read_csv(path, **self.read_csv_kwargs)
        return _reorder_columns(df, self.read_csv_kwargs.get("usecols"))


class CompressedSingleDataset(TabbedDataset):
    """A dataset which ships a single table of triples inside an archive and gets split automatically."""

    #: The source class used to unpack the archive
    source_cls: ClassVar[type[ArchiveSource]]

    def __init__(
        self,
        url: str,
        relative_path: str | pathlib.PurePosixPath,
        name: str | None = None,
        cache_root: str | None = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        delimiter: str | None = None,
        random_state: TorchRandomHint = None,
        read_csv_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize dataset.

        :param url: The url where to download the dataset from
        :param relative_path: The path inside the archive to the contained dataset.
        :param name: The name of the file. If not given, tries to get the name from the end of the URL
        :param cache_root: An optional directory to store the extracted files. Is none is given, the default PyKEEN
            directory is used. This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to
            ``~/.pykeen``.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param random_state: An optional random state to make the training/testing/validation split reproducible.
        :param delimiter: The delimiter for the contained dataset.
        :param read_csv_kwargs: Keyword arguments to pass through to :func:`pandas.read_csv`.
        """
        # note: these are set before `super().__init__`, since an eager load reads them
        self.url = url
        self.name = name or name_from_url(url)
        self.delimiter = delimiter or "\t"
        self._relative_path = pathlib.PurePosixPath(relative_path)
        self.read_csv_kwargs = read_csv_kwargs or {}
        self.read_csv_kwargs.setdefault("sep", self.delimiter)

        super().__init__(
            cache_root=cache_root,
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            eager=eager,
        )

    # docstr-coverage: inherited
    def _get_path(self) -> pathlib.Path:  # noqa: D102
        """Get the path of the *archive*, which is also used as the dataset's metadata path."""
        return self.cache_root.joinpath(self.name)

    def _get_source(self) -> ArchiveSource:
        """Build the source for the single member of the archive."""
        return self.source_cls(
            members={"data": self._relative_path},
            cache_root=self.cache_root,
            url=self.url,
            # note: goes through `_get_path` so that subclasses can point at a pre-existing archive
            archive_path=self._get_path(),
        )

    # docstr-coverage: inherited
    def _get_df(self) -> pd.DataFrame:  # noqa: D102
        path = self._get_source().paths()["data"]
        df = pd.read_csv(path, **self.read_csv_kwargs)
        return _reorder_columns(df, self.read_csv_kwargs.get("usecols"))


class ZipSingleDataset(CompressedSingleDataset):
    """Loads a dataset that's a single file inside a zip archive."""

    source_cls = ZipArchiveSource


class TarFileSingleDataset(CompressedSingleDataset):
    """Loads a dataset that's a single file inside a tar.gz archive."""

    source_cls = TarArchiveSource
