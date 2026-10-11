"""Utility classes for constructing datasets."""

from __future__ import annotations

import logging
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import (
    IO,
    Any,
    ClassVar,
    NotRequired,
    Self,
    TypeAlias,
    TypedDict,
    Unpack,
    cast,
)

import click
import docdata
import pandas as pd
import torch
from more_click import verbose_option
from pystow.utils import ArchiveType, DownloadKwargs, name_from_url
from tabulate import tabulate

from .source import RemoteArchivedSource, RemoteSimpleSource, SimpleSource, Source
from ..constants import PYKEEN_DATASETS
from ..triples import CoreTriplesFactory, TriplesFactory
from ..triples.deteriorate import deteriorate
from ..triples.remix import remix
from ..triples.triples_factory import splits_similarity
from ..typing import MappedTriples, TorchRandomHint, TransductiveRatiosHint
from ..utils import (
    ExtraReprMixin,
    format_relative_comparison,
    normalize_path,
    normalize_string,
)

__all__ = [
    "CompressedSingleDataset",
    "CompressedSingleDatasetKwargs",
    "Dataset",
    "EagerDataset",
    "LazyDataset",
    "LazyDatasetKwargs",
    "PackedRemoteDataSet",
    "PackedRemoteDataSetKwargs",
    "PackedZipRemoteDataset",
    "PathDataset",
    "PathDatasetKwargs",
    "SingleTabbedDataset",
    "SingleTabbedDatasetKwargs",
    "SourceDataSet",
    "SplittingLazyDataset",
    "SplittingLazyDatasetKwargs",
    "TabbedDataset",
    "TabbedDatasetKwargs",
    "TarFileRemoteDataset",
    "TarFileSingleDataset",
    "UnpackedRemoteDataSetKwargs",
    "UnpackedRemoteDataset",
    "ZipSingleDataset",
    "dataset_similarity",
]

logger = logging.getLogger(__name__)

DEFAULT_RATIOS: tuple[float, float, float] = (0.8, 0.1, 0.1)


class LazyDatasetKwargs(TypedDict):
    """Keyword arguments for a lazy dataset."""

    eager: NotRequired[bool]
    metadata: NotRequired[Metadata | None]


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
        metadata=factory.metadata,
        num_entities=len(kept_old_entity_ids_t),
        num_relations=len(kept_old_relation_ids_t),
    )


def _update_eval_core_factory(
    factory: CoreTriplesFactory,
    kept_old_entity_ids_t: torch.Tensor,
    kept_old_relation_ids_t: torch.Tensor,
) -> CoreTriplesFactory:
    mapped_triples = _filter_mapped_triples(
        mapped_triples=factory.mapped_triples,
        kept_old_entity_ids_t=kept_old_entity_ids_t,
        kept_old_relation_ids_t=kept_old_relation_ids_t,
    )
    return CoreTriplesFactory(
        mapped_triples=mapped_triples,
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


Metadata: TypeAlias = Mapping[str, Any]


class Dataset(ExtraReprMixin):
    """The base dataset class."""

    #: A factory wrapping the training triples
    training: CoreTriplesFactory
    #: A factory wrapping the testing triples, that share indices with the training triples
    testing: CoreTriplesFactory
    #: A factory wrapping the validation triples, that share indices with the training triples
    validation: CoreTriplesFactory | None
    #: the dataset's name
    metadata: Metadata | None

    metadata_file_name: ClassVar[str] = "metadata.pth"
    triples_factory_cls: ClassVar[type[CoreTriplesFactory]] = TriplesFactory

    def __eq__(self, __o: object, /) -> bool:
        return (
            isinstance(__o, Dataset)
            and (self.training == __o.training)
            and (self.testing == __o.testing)
            and ((self.validation is None and __o.validation is None) or (self.validation == __o.validation))
        )

    # defining __eq__ implicitly sets __hash__ to None; make this explicit
    __hash__ = None  # type: ignore[assignment]

    @property
    def factory_dict(self) -> Mapping[str, CoreTriplesFactory]:
        """Return a dictionary of the three factories."""
        rv = {
            "training": self.training,
            "testing": self.testing,
        }
        if self.validation:
            rv["validation"] = self.validation
        return rv

    @property
    def entity_to_id(self) -> Mapping[str, int]:
        """The mapping of entity labels to IDs."""
        if not isinstance(self.training, TriplesFactory):
            raise TypeError(f"{self.training.__class__} does not have labeling information.")
        return self.training.entity_to_id

    @property
    def relation_to_id(self) -> Mapping[str, int]:
        """The mapping of relation labels to IDs."""
        if not isinstance(self.training, TriplesFactory):
            raise TypeError(f"{self.training.__class__} does not have labeling information.")
        return self.training.relation_to_id

    @property
    def num_entities(self) -> int:
        """The number of entities."""
        return self.training.num_entities

    @property
    def num_relations(self) -> int:
        """The number of relations."""
        return self.training.num_relations

    @classmethod
    def docdata(cls, *parts: str) -> Any:
        """Get docdata for this class."""
        rv = docdata.get_docdata(cls)
        for part in parts:
            rv = rv[part]
        return rv

    @staticmethod
    def triples_sort_key(dataset_cls: type[Dataset]) -> int:
        """Get the number of triples for sorting."""
        return dataset_cls.docdata("statistics", "triples")

    @classmethod
    def triples_pair_sort_key(cls, pair: tuple[str, type[Dataset]]) -> int:
        """Get the number of triples for sorting in an iterator context."""
        return cls.triples_sort_key(pair[1])

    def _summary_rows(self) -> list[tuple[str, Any, Any, int]]:
        return [
            (
                label,
                triples_factory.num_entities,
                triples_factory.num_relations,
                triples_factory.num_triples,
            )
            for label, triples_factory in zip(
                ("Training", "Testing", "Validation"),
                (self.training, self.testing, self.validation),
                strict=True,
            )
            # note: the validation factory is optional
            if triples_factory is not None
        ]

    def summary_str(self, title: str | None = None, show_examples: int | None = 5, end: str = "\n") -> str:
        """Make a summary string of all of the factories."""
        rows = self._summary_rows()
        n_triples = sum(count for *_, count in rows)
        rows.append(("Total", "-", "-", n_triples))
        t = tabulate(rows, headers=["Name", "Entities", "Relations", "Triples"])
        rv = f"{title or self.__class__.__name__}\n{t}"
        if show_examples:
            if not isinstance(self.training, TriplesFactory):
                raise AttributeError(f"{self.training.__class__} does not have labeling information.")
            examples = tabulate(
                self.training.label_triples(self.training.mapped_triples[:show_examples]),
                headers=["Head", "Relation", "tail"],
            )
            rv += "\n" + examples
        return rv + end

    def summarize(
        self,
        title: str | None = None,
        show_examples: int | None = 5,
        file: IO[str] | None = None,
    ) -> None:
        """Print a summary of the dataset."""
        print(self.summary_str(title=title, show_examples=show_examples), file=file)

    def iter_extra_repr(self) -> Iterable[str]:
        """Yield extra entries for the instance's string representation."""
        yield f"num_entities={self.num_entities}"
        yield f"num_relations={self.num_relations}"

    @classmethod
    def from_path(
        cls,
        path: str | pathlib.Path | IO[str],
        *,
        ratios: TransductiveRatiosHint | None = None,
    ) -> Dataset:
        """Create a dataset from a single triples factory by splitting it in 3."""
        tf = TriplesFactory.from_path(path=path)
        return cls.from_tf(tf=tf, ratios=ratios)

    @classmethod
    def from_directory_binary(cls, path: str | pathlib.Path) -> Dataset:
        """Load a dataset from a directory."""
        path = pathlib.Path(path)

        if not path.is_dir():
            raise NotADirectoryError(path)

        tfs = {}
        # TODO: Make a constant for the names
        for key in ("training", "testing", "validation"):
            tf_path = path.joinpath(key)
            if tf_path.is_dir():
                tfs[key] = cls.triples_factory_cls.from_path_binary(path=tf_path)
            else:
                logger.warning(f"{tf_path.as_uri()} does not exist.")
        metadata_path = path.joinpath(cls.metadata_file_name)
        # TODO: consider restricting metadata to JSON
        metadata = torch.load(metadata_path, weights_only=False) if metadata_path.is_file() else None
        return EagerDataset(**tfs, metadata=metadata)

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

    @staticmethod
    def from_tf(
        tf: TriplesFactory,
        *,
        ratios: TransductiveRatiosHint | None = None,
        metadata: Metadata | None = None,
    ) -> Dataset:
        """Create a dataset from a single triples factory by splitting it in 3."""
        training, testing, validation = cast(
            tuple[TriplesFactory, TriplesFactory, TriplesFactory],
            tf.split(ratios or DEFAULT_RATIOS),
        )
        return EagerDataset(training=training, testing=testing, validation=validation, metadata=metadata)

    @classmethod
    def cli(cls) -> None:
        """Run the CLI."""

        @click.command(help=f"{cls.__name__} Dataset CLI.")
        @verbose_option
        def main() -> None:
            """Run the dataset CLI."""
            click.secho(f"Loading {cls.__name__}", fg="green", bold=True)
            click.echo(cls().summary_str())

        main()

    def get_normalized_name(self) -> str:
        """Get the normalized name of the dataset."""
        return normalize_string((self.metadata or {}).get("name") or self.__class__.__name__)

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

    def _tup(self) -> tuple[CoreTriplesFactory, ...]:
        if self.validation is None:
            return self.training, self.testing
        return self.training, self.testing, self.validation

    def restrict(
        self,
        entities: Collection[int] | Collection[str] | None = None,
        relations: Collection[int] | Collection[str] | None = None,
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
            if not isinstance(self.testing, TriplesFactory):
                raise TypeError(
                    f"Expected self.testing to be of type TriplesFactory, but got {type(self.testing).__name__}."
                )
            if not (self.validation is None or isinstance(self.validation, TriplesFactory)):
                raise TypeError(
                    f"Expected self.validation to be None or of type TriplesFactory, "
                    f"but got {type(self.validation).__name__}."
                )
            entity_to_id = _restrict_mapping(
                id_to_label=training.entity_id_to_label,
                kept_ids=kept_entity_ids_t.tolist(),
            )
            relation_to_id = _restrict_mapping(
                id_to_label=training.relation_id_to_label,
                kept_ids=kept_relation_ids_t.tolist(),
            )
            training = TriplesFactory(
                mapped_triples=cast(MappedTriples, new_training_triples),
                entity_to_id=entity_to_id,
                relation_to_id=relation_to_id,
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
            restriction_meta |= {
                "entities": sorted(entities),
                "invert_entity_selection": invert_entity_selection,
            }
        if relations:
            restriction_meta |= {
                "relations": sorted(relations),
                "invert_relation_selection": invert_relation_selection,
            }
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
        metadata: Metadata | None = None,
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

    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"metadata={self.metadata}"


class LazyDataset(Dataset, ABC):
    """A dataset whose training, testing, and optional validation factories are lazily loaded."""

    #: The actual instance of the training factory, which is exposed to the user through `training`
    _training: TriplesFactory | None = None
    #: The actual instance of the testing factory, which is exposed to the user through `testing`
    _testing: TriplesFactory | None = None
    #: The actual instance of the validation factory, which is exposed to the user through `validation`
    _validation: TriplesFactory | None = None

    def __init__(
        self,
        metadata: Metadata | None = None,
        eager: bool = False,
    ) -> None:
        """Construct the lazy dataset."""
        self.metadata = metadata
        if eager:
            self._load()
            self._load_validation()

    @property
    def training(self) -> TriplesFactory:  # type: ignore[override]
        """The training triples factory."""
        if not self._loaded:
            self._load()
        if self._training is None:
            raise RuntimeError("The dataset did not load its training factory.")
        return self._training

    @property
    def testing(self) -> TriplesFactory:  # type: ignore[override]
        """The testing triples factory that shares indices with the training triples factory."""
        if not self._loaded:
            self._load()
        if self._testing is None:
            raise RuntimeError("The dataset did not load its testing factory.")
        return self._testing

    @property
    def validation(self) -> TriplesFactory | None:  # type: ignore[override]
        """The validation triples factory that shares indices with the training triples factory."""
        if not self._loaded:
            self._load()
        if not self._loaded_validation:
            self._load_validation()
        return self._validation

    @property
    def _loaded(self) -> bool:
        return self._training is not None and self._testing is not None

    @property
    def _loaded_validation(self):
        return self._validation is not None

    @abstractmethod
    def _load(self) -> None:
        """Load the training and testing triples factories."""

    @abstractmethod
    def _load_validation(self) -> None:
        """Load the validation triples factory."""

    def _help_cache(self, cache_root: str | pathlib.Path | None) -> pathlib.Path:
        """Get the appropriate cache root directory.

        :param cache_root: If none is passed, defaults to a subfolder of the PyKEEN home directory defined in
            :data:`~pykeen.constants.PYKEEN_HOME`. The subfolder is named based on the class inheriting from
            :class:`~pykeen.datasets.base.Dataset`.

        :returns: A path object for the calculated cache root directory
        """
        cache_root = normalize_path(
            cache_root,
            *self._cache_sub_directories(),
            mkdir=True,
            default=PYKEEN_DATASETS,
        )
        logger.debug("using cache root at %s", cache_root.as_uri())
        return cache_root

    def _cache_sub_directories(self) -> Iterable[str]:
        """Iterate over appropriate cache sub-directory."""
        # TODO: use class-resolver normalize?
        yield self.__class__.__name__.lower()


class PathDatasetKwargs(LazyDatasetKwargs):
    """Keyword arguments for a path dataset."""

    load_triples_kwargs: NotRequired[Mapping[str, Any] | None]


class SourceDataSet(LazyDataset):
    """A lazy dataset for arbitrary sources."""

    def __init__(
        self,
        training_source: Source,
        testing_source: Source,
        validation_source: Source | None = None,
        *,
        load_triples_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Unpack[LazyDatasetKwargs],
    ) -> None:
        """Initialize a dataset from sources."""
        self.training_source = training_source
        self.testing_source = testing_source
        self.validation_source = validation_source
        self.load_triples_kwargs = load_triples_kwargs
        super().__init__(**kwargs)

    def _load(self) -> None:
        with self.training_source.open() as training_file:
            self._training = TriplesFactory.from_path(
                training_file,
                load_triples_kwargs=self.load_triples_kwargs,
            )
        with self.testing_source.open() as testing_file:
            self._testing = TriplesFactory.from_path(
                testing_file,
                entity_to_id=self._training.entity_to_id,  # share entity index with training
                relation_to_id=self._training.relation_to_id,  # share relation index with training
                load_triples_kwargs=self.load_triples_kwargs,
            )

    def _load_validation(self) -> None:
        # don't call this function by itself. assumes called through the `validation`
        # property and the _training factory has already been loaded
        if self._training is None:
            raise RuntimeError("The training factory has to be loaded before the validation factory.")
        if self.validation_source is None:
            self._validation = None
        else:
            with self.validation_source.open() as file:
                self._validation = TriplesFactory.from_path(
                    file,
                    entity_to_id=self._training.entity_to_id,  # share entity index with training
                    relation_to_id=self._training.relation_to_id,  # share relation index with training
                    load_triples_kwargs=self.load_triples_kwargs,
                )

    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield f'training_path="{self.training_source.path}'
        yield f'testing_path="{self.testing_source.path}'
        if self.validation_source:
            yield f'validation_path="{self.validation_source.path}")'


class PathDataset(SourceDataSet):
    """Contains a lazy reference to a training, testing, and validation dataset."""

    def __init__(
        self,
        training_path: str | pathlib.Path,
        testing_path: str | pathlib.Path,
        validation_path: str | pathlib.Path | None = None,
        **kwargs: Unpack[PathDatasetKwargs],
    ) -> None:
        """Initialize the dataset.

        :param training_path: Path to the training triples file or training triples file.
        :param testing_path: Path to the testing triples file or testing triples file.
        :param validation_path: Path to the validation triples file or validation triples file.
        """
        super().__init__(
            training_source=SimpleSource(pathlib.Path(training_path)),
            testing_source=SimpleSource(pathlib.Path(testing_path)),
            validation_source=(SimpleSource(pathlib.Path(validation_path)) if validation_path else None),
            **kwargs,
        )


class UnpackedRemoteDataSetKwargs(PathDatasetKwargs):
    """Keyword arguments for an unpacked remote dataset."""

    cache_root: NotRequired[str | pathlib.Path | None]
    force: NotRequired[bool]
    download_kwargs: NotRequired[DownloadKwargs | None]


class UnpackedRemoteDataset(SourceDataSet):
    """A dataset with all three of train, test, and validation sets as URLs."""

    def __init__(
        self,
        training_url: str,
        testing_url: str,
        validation_url: str | None = None,
        *,
        cache_root: str | pathlib.Path | None = None,
        force: bool = False,
        download_kwargs: DownloadKwargs | None = None,
        **kwargs: Unpack[PathDatasetKwargs],
    ) -> None:
        """Initialize dataset.

        :param training_url: The URL of the training file
        :param testing_url: The URL of the testing file
        :param validation_url: The URL of the validation file
        :param cache_root: An optional directory to store the extracted files. Is none is given, the default PyKEEN
            directory is used. This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to
            ``~/.pykeen``.
        :param force: If true, redownload any cached files
        :param download_kwargs: Keyword arguments to pass to :func:`pystow.utils.download`
        """
        if download_kwargs is None:
            download_kwargs = {}
        download_kwargs.setdefault("backend", "urllib")
        cache_root_ = self._help_cache(cache_root)

        def _get_source(url: str) -> Source:
            return RemoteSimpleSource(
                path=cache_root_.joinpath(name_from_url(url)),
                url=url,
                force=force,
                download_kwargs=cast(DownloadKwargs, download_kwargs),
            )

        super().__init__(
            training_source=_get_source(training_url),
            testing_source=_get_source(testing_url),
            validation_source=_get_source(validation_url) if validation_url is not None else None,
            **kwargs,
        )


class PackedRemoteDataSetKwargs(PathDatasetKwargs):
    """Keyword arguments for a packed remote dataset."""

    force: NotRequired[bool]
    cache_root: NotRequired[str | pathlib.Path | None]


class PackedRemoteDataSet(SourceDataSet):
    """An abstract base class for packed remote datasets."""

    archive_type: ClassVar[ArchiveType]

    def __init__(
        self,
        url: str,
        relative_training_path: str | pathlib.PurePath,
        relative_testing_path: str | pathlib.PurePath,
        relative_validation_path: str | pathlib.PurePath,
        *,
        force: bool = False,
        cache_root: str | pathlib.Path | None = None,
        **kwargs: Unpack[PathDatasetKwargs],
    ) -> None:
        """Initialize dataset.

        :param url: The url where to download the dataset from.
        :param relative_training_path: The path inside the cache root where the training path gets extracted
        :param relative_testing_path: The path inside the cache root where the testing path gets extracted
        :param relative_validation_path: The path inside the cache root where the validation path gets extracted
        :param force: whether files should be re-downloaded
        :param cache_root: An optional directory to store the extracted files. Is none is given, the default PyKEEN
            directory is used. This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to
            ``~/.pykeen``.
        """
        name = name_from_url(url)
        path = self._help_cache(cache_root).joinpath(name)

        def _get_source(inner_path: str | pathlib.PurePath) -> RemoteArchivedSource:
            return RemoteArchivedSource(
                archive_type=self.archive_type,
                url=url,
                force=force,
                path=path,
                inner_path=inner_path,
            )

        super().__init__(
            training_source=_get_source(relative_training_path),
            testing_source=_get_source(relative_testing_path),
            validation_source=_get_source(relative_validation_path),
            **kwargs,
        )


class TarFileRemoteDataset(PackedRemoteDataSet):
    """A remote dataset packed with a TAR archive."""

    archive_type = "tar"


class PackedZipRemoteDataset(PackedRemoteDataSet):
    """A remote dataset packed with a ZIP archive."""

    archive_type = "zip"


class SplittingLazyDatasetKwargs(LazyDatasetKwargs):
    """Keyword arguments for a splitting lazy dataset."""

    random_state: NotRequired[TorchRandomHint]
    ratios: NotRequired[TransductiveRatiosHint | None]


class SplittingLazyDataset(LazyDataset, ABC):
    """A dataset that splits."""

    def __init__(
        self,
        *,
        random_state: TorchRandomHint = None,
        ratios: TransductiveRatiosHint | None = None,
        **kwargs: Unpack[LazyDatasetKwargs],
    ) -> None:
        """Initialize the dataset."""
        self.random_state = random_state
        self.ratios = ratios or DEFAULT_RATIOS
        super().__init__(**kwargs)

    @abstractmethod
    def _get_triples_factory(self) -> TriplesFactory:
        """Get the triples factory that will be split."""

    def _load(self) -> None:
        tf = self._get_triples_factory()
        self._training, self._testing, self._validation = tf.split(ratios=self.ratios, random_state=self.random_state)

    def _load_validation(self) -> None:
        pass  # already loaded by _load()


class TabbedDatasetKwargs(SplittingLazyDatasetKwargs):
    """Keyword arguments for a tabbed dataset."""

    read_csv_kwargs: NotRequired[dict[str, Any] | None]
    delimiter: NotRequired[str | None]


class TabbedDataset(SplittingLazyDataset):
    """This class is for when you've got a single TSV of edges and want them to get auto-split."""

    def __init__(
        self,
        source: Source,
        *,
        read_csv_kwargs: dict[str, Any] | None = None,
        delimiter: str | None = None,
        **kwargs: Unpack[SplittingLazyDatasetKwargs],
    ) -> None:
        """Initialize dataset."""
        self.source = source
        self.read_csv_kwargs = read_csv_kwargs or {}
        self.read_csv_kwargs.setdefault("sep", delimiter or "\t")
        super().__init__(**kwargs)

    def _get_path(self) -> pathlib.Path | None:
        """Get the path of the data if there's a single file."""
        return self.source.path

    def _get_triples_factory(self) -> TriplesFactory:
        with self.source.open() as file:
            df = pd.read_csv(file, **self.read_csv_kwargs)
        df = _reorder_columns(df, self.read_csv_kwargs.get("usecols"))
        path = self._get_path()
        return TriplesFactory.from_labeled_triples(
            triples=df.values,
            metadata={"path": path} if path else None,
        )


class CompressedSingleDatasetKwargs(TabbedDatasetKwargs):
    """Keyword arguments for a compressed single file dataset."""

    name: NotRequired[str | None]
    cache_root: NotRequired[str | pathlib.Path | None]
    download_kwargs: NotRequired[DownloadKwargs | None]


class CompressedSingleDataset(TabbedDataset):
    """Loads a dataset that's a single file inside an archive."""

    archive_type: ClassVar[ArchiveType]

    def __init__(
        self,
        url: str,
        relative_path: str | pathlib.PurePosixPath,
        *,
        name: str | None = None,
        cache_root: str | pathlib.Path | None = None,
        download_kwargs: DownloadKwargs | None = None,
        force: bool = False,
        **kwargs: Unpack[TabbedDatasetKwargs],
    ) -> None:
        """Initialize dataset.

        :param url: The url where to download the dataset from
        :param relative_path: The path inside the archive to the contained dataset.
        :param name: The name of the file. If not given, tries to get the name from the end of the URL
        :param cache_root: An optional directory to store the extracted files. Is none is given, the default PyKEEN
            directory is used. This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to
            ``~/.pykeen``.
        :param download_kwargs: Keyword arguments to pass through to :func:`pystow.utils.download`.
        :param force: whether files should be re-downloaded
        """
        if not name:
            name = name_from_url(url) if url else pathlib.PurePath(relative_path).name
        source = RemoteArchivedSource(
            archive_type=self.archive_type,
            url=url,
            path=self._help_cache(cache_root).joinpath(name),
            inner_path=relative_path,
            download_kwargs=download_kwargs,
            force=force,
        )
        super().__init__(source, **kwargs)


class ZipSingleDataset(CompressedSingleDataset):
    """Loads a dataset that's a single file inside a zip archive."""

    archive_type = "zip"


class TarFileSingleDataset(CompressedSingleDataset):
    """Loads a dataset that's a single file inside a tar.gz archive."""

    archive_type = "tar"


class SingleTabbedDatasetKwargs(TabbedDatasetKwargs):
    """Keyword arguments for a single file tabbed dataset."""

    #: The name of the file
    name: NotRequired[str | None]
    #: An override for where the files are cached
    cache_root: NotRequired[str | pathlib.Path | None]
    #: An override for configuration of the download workflow with :func:`pystow.utils.download`
    download_kwargs: NotRequired[DownloadKwargs | None]
    #: If given as true, will re-download the file
    force: NotRequired[bool]


class SingleTabbedDataset(TabbedDataset):
    """This class is for when you've got a single TSV of edges and want them to get auto-split."""

    def __init__(
        self,
        url: str,
        *,
        name: str | None = None,
        cache_root: str | pathlib.Path | None = None,
        download_kwargs: DownloadKwargs | None = None,
        force: bool = False,
        **kwargs: Unpack[TabbedDatasetKwargs],
    ) -> None:
        """Initialize dataset.

        :param url: The url where to download the dataset from
        :param name: The name of the file. If not given, tries to get the name from the end of the URL
        :param cache_root: An optional directory to store the extracted files. Is none is given, the default PyKEEN
            directory is used. This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to
            ``~/.pykeen``.
        :param download_kwargs: Keyword arguments to pass through to :func:`pystow.utils.download`.
        :param force: whether files should be re-downloaded
        """
        name = name or name_from_url(url)
        path = self._help_cache(cache_root).joinpath(name)
        source = RemoteSimpleSource(
            url=url,
            force=force,
            path=path,
            download_kwargs=download_kwargs,
        )
        super().__init__(source, **kwargs)
