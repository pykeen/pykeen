# -*- coding: utf-8 -*-

"""Utility classes for constructing datasets."""

from __future__ import annotations

import logging
import pathlib
import tarfile
import zipfile
from abc import abstractmethod
from io import BytesIO
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union, cast

import click
import docdata
import pandas as pd
import requests
import torch
from more_click import verbose_option
from pystow.utils import download, name_from_url
from tabulate import tabulate

from ..constants import PYKEEN_DATASETS
from ..triples import CoreTriplesFactory, TriplesFactory
from ..triples.deteriorate import deteriorate
from ..triples.remix import remix
from ..triples.triples_factory import splits_similarity
from ..typing import TorchRandomHint
from ..utils import normalize_path, normalize_string

__all__ = [
    # Base classes
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


def dataset_similarity(a: Dataset, b: Dataset, metric: Optional[str] = None) -> float:
    """Calculate the similarity between two datasets.

    :param a: The reference dataset
    :param b: The target dataset
    :param metric: The similarity metric to use. Defaults to `tanimoto`. Could either be a symmetric
        or asymmetric metric.
    :returns: A scalar value between 0 and 1 where closer to 1 means the datasets are more
        similar based on the metric.

    :raises ValueError: if an invalid metric type is passed. Right now, there's only `tanimoto`,
        but this could change in later.
    """
    if metric == "tanimoto" or metric is None:
        return splits_similarity(a._tup(), b._tup())
    raise ValueError(f"invalid metric: {metric}")


class Dataset:
    """The base dataset class."""

    #: A factory wrapping the training triples
    training: CoreTriplesFactory
    #: A factory wrapping the testing triples, that share indices with the training triples
    testing: CoreTriplesFactory
    #: A factory wrapping the validation triples, that share indices with the training triples
    validation: Optional[CoreTriplesFactory]
    #: the dataset's name
    metadata: Optional[Mapping[str, Any]] = None

    metadata_file_name: ClassVar[str] = "metadata.pth"
    triples_factory_cls: ClassVar[Type[CoreTriplesFactory]] = TriplesFactory

    def __eq__(self, __o: object) -> bool:  # noqa: D105
        return (
            isinstance(__o, Dataset)
            and (self.training == __o.training)
            and (self.testing == __o.testing)
            and ((self.validation is None and __o.validation is None) or (self.validation == __o.validation))
            and (self.create_inverse_triples == __o.create_inverse_triples)
        )

    @property
    def factory_dict(self) -> Mapping[str, CoreTriplesFactory]:
        """Return a dictionary of the three factories."""
        rv = dict(
            training=self.training,
            testing=self.testing,
        )
        if self.validation:
            rv["validation"] = self.validation
        return rv

    @property
    def entity_to_id(self):  # noqa: D401
        """The mapping of entity labels to IDs."""
        if not isinstance(self.training, TriplesFactory):
            raise AttributeError(f"{self.training.__class__} does not have labeling information.")
        return self.training.entity_to_id

    @property
    def relation_to_id(self):  # noqa: D401
        """The mapping of relation labels to IDs."""
        if not isinstance(self.training, TriplesFactory):
            raise AttributeError(f"{self.training.__class__} does not have labeling information.")
        return self.training.relation_to_id

    @property
    def num_entities(self):  # noqa: D401
        """The number of entities."""
        return self.training.num_entities

    @property
    def num_relations(self):  # noqa: D401
        """The number of relations."""
        return self.training.num_relations

    @property
    def create_inverse_triples(self):
        """Return whether inverse triples are created *for the training factory*."""
        return self.training.create_inverse_triples

    @classmethod
    def docdata(cls, *parts: str) -> Any:
        """Get docdata for this class."""
        rv = docdata.get_docdata(cls)
        for part in parts:
            rv = rv[part]
        return rv

    @staticmethod
    def triples_sort_key(cls: Type[Dataset]) -> int:
        """Get the number of triples for sorting."""
        return cls.docdata("statistics", "triples")

    @classmethod
    def triples_pair_sort_key(cls, pair: Tuple[str, Type[Dataset]]) -> int:
        """Get the number of triples for sorting in an iterator context."""
        return cls.triples_sort_key(pair[1])

    def _summary_rows(self):
        return [
            (label, triples_factory.num_entities, triples_factory.num_relations, triples_factory.num_triples)
            for label, triples_factory in zip(
                ("Training", "Testing", "Validation"), (self.training, self.testing, self.validation)
            )
        ]

    def summary_str(self, title: Optional[str] = None, show_examples: Optional[int] = 5, end="\n") -> str:
        """Make a summary string of all of the factories."""
        rows = self._summary_rows()
        n_triples = sum(count for *_, count in rows)
        rows.append(("Total", "-", "-", n_triples))
        t = tabulate(rows, headers=["Name", "Entities", "Relations", "Triples"])
        rv = f"{title or self.__class__.__name__} (create_inverse_triples={self.create_inverse_triples})\n{t}"
        if show_examples:
            if not isinstance(self.training, TriplesFactory):
                raise AttributeError(f"{self.training.__class__} does not have labeling information.")
            examples = tabulate(
                self.training.label_triples(self.training.mapped_triples[:show_examples]),
                headers=["Head", "Relation", "tail"],
            )
            rv += "\n" + examples
        return rv + end

    def summarize(self, title: Optional[str] = None, show_examples: Optional[int] = 5, file=None) -> None:
        """Print a summary of the dataset."""
        print(self.summary_str(title=title, show_examples=show_examples), file=file)  # noqa:T201

    def _extra_repr(self) -> Iterable[str]:
        """Yield extra entries for the instance's string representation."""
        yield f"num_entities={self.num_entities}"
        yield f"num_relations={self.num_relations}"
        yield f"create_inverse_triples={self.create_inverse_triples}"

    def __str__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}({', '.join(self._extra_repr())})"

    @classmethod
    def from_path(cls, path: Union[str, pathlib.Path], ratios: Optional[List[float]] = None) -> "Dataset":
        """Create a dataset from a single triples factory by splitting it in 3."""
        tf = TriplesFactory.from_path(path=path)
        return cls.from_tf(tf=tf, ratios=ratios)

    @classmethod
    def from_directory_binary(cls, path: Union[str, pathlib.Path]) -> "Dataset":
        """Load a dataset from a directory."""
        path = pathlib.Path(path)

        if not path.is_dir():
            raise NotADirectoryError(path)

        tfs = dict()
        # TODO: Make a constant for the names
        for key in ("training", "testing", "validation"):
            tf_path = path.joinpath(key)
            if tf_path.is_dir():
                tfs[key] = cls.triples_factory_cls.from_path_binary(path=tf_path)
            else:
                logger.warning(f"{tf_path.as_uri()} does not exist.")
        metadata_path = path.joinpath(cls.metadata_file_name)
        metadata = torch.load(metadata_path) if metadata_path.is_file() else None
        return EagerDataset(**tfs, metadata=metadata)

    def to_directory_binary(self, path: Union[str, pathlib.Path]) -> None:
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
    def from_tf(tf: TriplesFactory, ratios: Optional[List[float]] = None) -> "Dataset":
        """Create a dataset from a single triples factory by splitting it in 3."""
        training, testing, validation = cast(
            Tuple[TriplesFactory, TriplesFactory, TriplesFactory],
            tf.split(ratios or [0.8, 0.1, 0.1]),
        )
        return EagerDataset(training=training, testing=testing, validation=validation)

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

    def remix(self, random_state: TorchRandomHint = None, **kwargs) -> Dataset:
        """Remix a dataset using :func:`pykeen.triples.remix.remix`."""
        return EagerDataset(
            *remix(
                *self._tup(),
                random_state=random_state,
                **kwargs,
            ),
        )

    def deteriorate(self, n: Union[int, float], random_state: TorchRandomHint = None) -> Dataset:
        """Deteriorate n triples from the dataset's training with :func:`pykeen.triples.deteriorate.deteriorate`."""
        return EagerDataset(
            *deteriorate(
                *self._tup(),
                n=n,
                random_state=random_state,
            )
        )

    def similarity(self, other: Dataset, metric: Optional[str] = None) -> float:
        """Compute the similarity between two shuffles of the same dataset.

        :param other: The other shuffling of the dataset
        :param metric: The metric to use. Defaults to `tanimoto`.
        :return: A float of the similarity

        .. seealso:: :func:`pykeen.triples.triples_factory.splits_similarity`.
        """
        return dataset_similarity(self, other, metric=metric)

    def _tup(self):
        if self.validation is None:
            return self.training, self.testing
        return self.training, self.testing, self.validation


class EagerDataset(Dataset):
    """A dataset whose training, testing, and optional validation factories are pre-loaded."""

    def __init__(
        self,
        training: CoreTriplesFactory,
        testing: CoreTriplesFactory,
        validation: Optional[CoreTriplesFactory] = None,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
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
    def _extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super()._extra_repr()
        yield f"metadata={self.metadata}"


class LazyDataset(Dataset):
    """A dataset whose training, testing, and optional validation factories are lazily loaded."""

    #: The actual instance of the training factory, which is exposed to the user through `training`
    _training: Optional[TriplesFactory] = None
    #: The actual instance of the testing factory, which is exposed to the user through `testing`
    _testing: Optional[TriplesFactory] = None
    #: The actual instance of the validation factory, which is exposed to the user through `validation`
    _validation: Optional[TriplesFactory] = None
    #: The directory in which the cached data is stored
    cache_root: pathlib.Path

    @property
    def training(self) -> TriplesFactory:  # type:ignore # noqa: D401
        """The training triples factory."""
        if not self._loaded:
            self._load()
        assert self._training is not None
        return self._training

    @property
    def testing(self) -> TriplesFactory:  # type:ignore # noqa: D401
        """The testing triples factory that shares indices with the training triples factory."""
        if not self._loaded:
            self._load()
        assert self._testing is not None
        return self._testing

    @property
    def validation(self) -> Optional[TriplesFactory]:  # type:ignore # noqa: D401
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

    def _load(self) -> None:
        raise NotImplementedError

    def _load_validation(self) -> None:
        raise NotImplementedError

    def _help_cache(self, cache_root: Union[None, str, pathlib.Path]) -> pathlib.Path:
        """Get the appropriate cache root directory.

        :param cache_root: If none is passed, defaults to a subfolder of the
            PyKEEN home directory defined in :data:`pykeen.constants.PYKEEN_HOME`.
            The subfolder is named based on the class inheriting from
            :class:`pykeen.datasets.base.Dataset`.
        :returns: A path object for the calculated cache root directory
        """
        cache_root = normalize_path(cache_root, *self._cache_sub_directories(), mkdir=True, default=PYKEEN_DATASETS)
        logger.debug("using cache root at %s", cache_root.as_uri())
        return cache_root

    def _cache_sub_directories(self) -> Iterable[str]:
        """Iterate over appropriate cache sub-directory."""
        # TODO: use class-resolver normalize?
        yield self.__class__.__name__.lower()


class PathDataset(LazyDataset):
    """Contains a lazy reference to a training, testing, and validation dataset."""

    def __init__(
        self,
        training_path: Union[str, pathlib.Path],
        testing_path: Union[str, pathlib.Path],
        validation_path: Union[None, str, pathlib.Path],
        eager: bool = False,
        create_inverse_triples: bool = False,
        load_triples_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialize the dataset.

        :param training_path: Path to the training triples file or training triples file.
        :param testing_path: Path to the testing triples file or testing triples file.
        :param validation_path: Path to the validation triples file or validation triples file.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param load_triples_kwargs: Arguments to pass through to :func:`TriplesFactory.from_path`
            and ultimately through to :func:`pykeen.triples.utils.load_triples`.
        """
        self.training_path = pathlib.Path(training_path)
        self.testing_path = pathlib.Path(testing_path)
        self.validation_path = pathlib.Path(validation_path) if validation_path else None

        self._create_inverse_triples = create_inverse_triples
        self.load_triples_kwargs = load_triples_kwargs

        if eager:
            self._load()
            self._load_validation()

    def _load(self) -> None:
        self._training = TriplesFactory.from_path(
            path=self.training_path,
            create_inverse_triples=self._create_inverse_triples,
            load_triples_kwargs=self.load_triples_kwargs,
        )
        self._testing = TriplesFactory.from_path(
            path=self.testing_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
            # do not explicitly create inverse triples for testing; this is handled by the evaluation code
            create_inverse_triples=False,
            load_triples_kwargs=self.load_triples_kwargs,
        )

    def _load_validation(self) -> None:
        # don't call this function by itself. assumes called through the `validation`
        # property and the _training factory has already been loaded
        assert self._training is not None
        if self.validation_path is None:
            self._validation = None
        else:
            self._validation = TriplesFactory.from_path(
                path=self.validation_path,
                entity_to_id=self._training.entity_to_id,  # share entity index with training
                relation_to_id=self._training.relation_to_id,  # share relation index with training
                # do not explicitly create inverse triples for testing; this is handled by the evaluation code
                create_inverse_triples=False,
                load_triples_kwargs=self.load_triples_kwargs,
            )

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
        cache_root: Optional[str] = None,
        force: bool = False,
        eager: bool = False,
        create_inverse_triples: bool = False,
        load_triples_kwargs: Optional[Mapping[str, Any]] = None,
        download_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """Initialize dataset.

        :param training_url: The URL of the training file
        :param testing_url: The URL of the testing file
        :param validation_url: The URL of the validation file
        :param cache_root:
            An optional directory to store the extracted files. Is none is given, the default PyKEEN directory is used.
            This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to ``~/.data/pykeen``.
        :param force: If true, redownload any cached files
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param load_triples_kwargs: Arguments to pass through to :func:`TriplesFactory.from_path`
            and ultimately through to :func:`pykeen.triples.utils.load_triples`.
        :param download_kwargs: Keyword arguments to pass to :func:`pystow.utils.download`
        """
        self.cache_root = self._help_cache(cache_root)

        self.training_url = training_url
        self.testing_url = testing_url
        self.validation_url = validation_url

        training_path = self.cache_root.joinpath(name_from_url(self.training_url))
        testing_path = self.cache_root.joinpath(name_from_url(self.testing_url))
        validation_path = self.cache_root.joinpath(name_from_url(self.validation_url))

        download_kwargs = {} if download_kwargs is None else dict(download_kwargs)
        download_kwargs.setdefault("backend", "urllib")

        for url, path in [
            (self.training_url, training_path),
            (self.testing_url, testing_path),
            (self.validation_url, validation_path),
        ]:
            if force or not path.is_file():
                download(url, path, **download_kwargs)

        super().__init__(
            training_path=training_path,
            testing_path=testing_path,
            validation_path=validation_path,
            eager=eager,
            create_inverse_triples=create_inverse_triples,
            load_triples_kwargs=load_triples_kwargs,
        )


class RemoteDataset(PathDataset):
    """Contains a lazy reference to a remote dataset that is loaded if needed."""

    def __init__(
        self,
        url: str,
        relative_training_path: Union[str, pathlib.PurePath],
        relative_testing_path: Union[str, pathlib.PurePath],
        relative_validation_path: Union[str, pathlib.PurePath],
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
    ):
        """Initialize dataset.

        :param url:
            The url where to download the dataset from.
        :param relative_training_path: The path inside the cache root where the training path gets extracted
        :param relative_testing_path: The path inside the cache root where the testing path gets extracted
        :param relative_validation_path: The path inside the cache root where the validation path gets extracted
        :param cache_root:
            An optional directory to store the extracted files. Is none is given, the default PyKEEN directory is used.
            This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to ``~/.data/pykeen``.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        """
        self.cache_root = self._help_cache(cache_root)

        self.url = url
        self._relative_training_path = pathlib.PurePath(relative_training_path)
        self._relative_testing_path = pathlib.PurePath(relative_testing_path)
        self._relative_validation_path = pathlib.PurePath(relative_validation_path)

        training_path, testing_path, validation_path = self._get_paths()
        super().__init__(
            training_path=training_path,
            testing_path=testing_path,
            validation_path=validation_path,
            eager=eager,
            create_inverse_triples=create_inverse_triples,
        )

    def _get_paths(self) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:  # noqa: D401
        """The paths where the extracted files can be found."""
        return (
            self.cache_root.joinpath(self._relative_training_path),
            self.cache_root.joinpath(self._relative_testing_path),
            self.cache_root.joinpath(self._relative_validation_path),
        )

    @abstractmethod
    def _extract(self, archive_file: BytesIO) -> None:
        """Extract from the downloaded file."""
        raise NotImplementedError

    def _get_bytes(self) -> BytesIO:
        logger.info(f"Requesting dataset from {self.url}")
        res = requests.get(url=self.url)
        res.raise_for_status()
        return BytesIO(res.content)

    # docstr-coverage: inherited
    def _load(self) -> None:  # noqa: D102
        all_unpacked = all(path.is_file() for path in self._get_paths())

        if not all_unpacked:
            archive_file = self._get_bytes()
            self._extract(archive_file=archive_file)
            logger.info(f"Extracted to {self.cache_root}.")

        super()._load()


class TarFileRemoteDataset(RemoteDataset):
    """A remote dataset stored as a tar file."""

    # docstr-coverage: inherited
    def _extract(self, archive_file: BytesIO) -> None:  # noqa: D102
        with tarfile.open(fileobj=archive_file) as tf:
            tf.extractall(path=self.cache_root)


class PackedZipRemoteDataset(LazyDataset):
    """Contains a lazy reference to a remote dataset that is loaded if needed."""

    head_column: int = 0
    relation_column: int = 1
    tail_column: int = 2
    sep = "\t"
    header = None

    def __init__(
        self,
        relative_training_path: Union[str, pathlib.PurePath],
        relative_testing_path: Union[str, pathlib.PurePath],
        relative_validation_path: Union[str, pathlib.PurePath],
        url: Optional[str] = None,
        name: Optional[str] = None,
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
    ):
        """Initialize dataset.

        :param relative_training_path: The path inside the zip file for the training data
        :param relative_testing_path: The path inside the zip file for the testing data
        :param relative_validation_path: The path inside the zip file for the validation data
        :param url:
            The url where to download the dataset from
        :param name:
            The name of the file. If not given, tries to get the name from the end of the URL
        :param cache_root:
            An optional directory to store the extracted files. Is none is given, the default PyKEEN directory is used.
            This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to ``~/.pykeen``.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.

        :raises ValueError: if there's no URL specified and there is no data already at the calculated path
        """
        self.cache_root = self._help_cache(cache_root)

        self.name = name or name_from_url(url)
        self.path = self.cache_root.joinpath(self.name)
        logger.debug("file path at %s", self.path)

        self.url = url
        if not self.path.is_file() and not self.url:
            raise ValueError(f"must specify url to download from since path does not exist: {self.path}")

        self.relative_training_path = pathlib.PurePath(relative_training_path)
        self.relative_testing_path = pathlib.PurePath(relative_testing_path)
        self.relative_validation_path = pathlib.PurePath(relative_validation_path)
        self._create_inverse_triples = create_inverse_triples
        if eager:
            self._load()
            self._load_validation()

    # docstr-coverage: inherited
    def _load(self) -> None:  # noqa: D102
        self._training = self._load_helper(self.relative_training_path)
        self._testing = self._load_helper(
            self.relative_testing_path,
            entity_to_id=self._training.entity_to_id,
            relation_to_id=self._training.relation_to_id,
        )

    def _load_validation(self) -> None:
        assert self._training is not None
        self._validation = self._load_helper(
            self.relative_validation_path,
            entity_to_id=self._training.entity_to_id,
            relation_to_id=self._training.relation_to_id,
        )

    def _load_helper(
        self,
        relative_path: pathlib.PurePath,
        entity_to_id: Optional[Mapping[str, Any]] = None,
        relation_to_id: Optional[Mapping[str, Any]] = None,
    ) -> TriplesFactory:
        if not self.path.is_file():
            if self.url is None:
                raise ValueError("url should be set")
            logger.info("downloading data from %s to %s", self.url, self.path)
            download(url=self.url, path=self.path)

        with zipfile.ZipFile(file=self.path) as zf:
            # relative paths within zip file's always follow Posix path, even on Windows
            with zf.open(relative_path.as_posix()) as file:
                logger.debug("loading %s", relative_path)
                df = pd.read_csv(
                    file,
                    usecols=[self.head_column, self.relation_column, self.tail_column],
                    header=self.header,
                    sep=self.sep,
                )
                return TriplesFactory.from_labeled_triples(
                    triples=df.values,
                    create_inverse_triples=self._create_inverse_triples,
                    metadata={"path": relative_path},
                    entity_to_id=entity_to_id,
                    relation_to_id=relation_to_id,
                )


class CompressedSingleDataset(LazyDataset):
    """Loads a dataset that's a single file inside an archive."""

    ratios = (0.8, 0.1, 0.1)

    def __init__(
        self,
        url: str,
        relative_path: Union[str, pathlib.PurePosixPath],
        name: Optional[str] = None,
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        delimiter: Optional[str] = None,
        random_state: TorchRandomHint = None,
    ):
        """Initialize dataset.

        :param url:
            The url where to download the dataset from
        :param relative_path:
            The path inside the archive to the contained dataset.
        :param name:
            The name of the file. If not given, tries to get the name from the end of the URL
        :param cache_root:
            An optional directory to store the extracted files. Is none is given, the default PyKEEN directory is used.
            This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to ``~/.pykeen``.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param random_state: An optional random state to make the training/testing/validation split reproducible.
        :param delimiter:
            The delimiter for the contained dataset.
        """
        self.cache_root = self._help_cache(cache_root)

        self.name = name or name_from_url(url)
        self.random_state = random_state
        self.delimiter = delimiter or "\t"
        self.url = url
        self._create_inverse_triples = create_inverse_triples
        self._relative_path = pathlib.PurePosixPath(relative_path)

        if eager:
            self._load()

    def _get_path(self) -> pathlib.Path:
        return self.cache_root.joinpath(self.name)

    def _load(self) -> None:
        df = self._get_df()
        tf_path = self._get_path()
        tf = TriplesFactory.from_labeled_triples(
            triples=df.values,
            create_inverse_triples=self._create_inverse_triples,
            metadata={"path": tf_path},
        )
        self._training, self._testing, self._validation = cast(
            Tuple[TriplesFactory, TriplesFactory, TriplesFactory],
            tf.split(
                ratios=self.ratios,
                random_state=self.random_state,
            ),
        )
        logger.info("[%s] done splitting data from %s", self.__class__.__name__, tf_path)

    def _get_df(self) -> pd.DataFrame:
        raise NotImplementedError

    def _load_validation(self) -> None:
        pass  # already loaded by _load()


class ZipSingleDataset(CompressedSingleDataset):
    """Loads a dataset that's a single file inside a zip archive."""

    def _get_df(self) -> pd.DataFrame:
        path = self._get_path()
        if not path.is_file():
            download(self.url, self._get_path())  # noqa:S310

        with zipfile.ZipFile(path) as zip_file:
            with zip_file.open(self._relative_path.as_posix()) as file:
                df = pd.read_csv(file, sep=self.delimiter)
        return df


class TarFileSingleDataset(CompressedSingleDataset):
    """Loads a dataset that's a single file inside a tar.gz archive."""

    def _get_df(self) -> pd.DataFrame:
        if not self._get_path().is_file():
            download(self.url, self._get_path())  # noqa:S310

        _actual_path = self.cache_root.joinpath(self._relative_path)
        if not _actual_path.is_file():
            logger.error(
                "[%s] untaring from %s (%s) to %s",
                self.__class__.__name__,
                self._get_path(),
                self._relative_path,
                _actual_path,
            )
            with tarfile.open(self._get_path()) as tar_file:
                # tarfile does not like pathlib
                tar_file.extract(str(self._relative_path), self.cache_root)

        df = pd.read_csv(_actual_path, sep=self.delimiter)
        return df


class TabbedDataset(LazyDataset):
    """This class is for when you've got a single TSV of edges and want them to get auto-split."""

    ratios: ClassVar[Sequence[float]] = (0.8, 0.1, 0.1)
    _triples_factory: Optional[TriplesFactory]

    def __init__(
        self,
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = None,
    ):
        """Initialize dataset.

        :param cache_root:
            An optional directory to store the extracted files. Is none is given, the default PyKEEN directory is used.
            This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to ``~/.pykeen``.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param random_state: An optional random state to make the training/testing/validation split reproducible.
        """
        self.cache_root = self._help_cache(cache_root)

        self._triples_factory = None
        self.random_state = random_state
        self._create_inverse_triples = create_inverse_triples
        self._training = None
        self._testing = None
        self._validation = None

        if eager:
            self._load()

    def _get_path(self) -> Optional[pathlib.Path]:
        """Get the path of the data if there's a single file."""

    def _get_df(self) -> pd.DataFrame:
        raise NotImplementedError

    def _load(self) -> None:
        df = self._get_df()
        path = self._get_path()
        tf = TriplesFactory.from_labeled_triples(
            triples=df.values,
            create_inverse_triples=self._create_inverse_triples,
            metadata=dict(path=path) if path else None,
        )
        self._training, self._testing, self._validation = cast(
            Tuple[TriplesFactory, TriplesFactory, TriplesFactory],
            tf.split(
                ratios=self.ratios,
                random_state=self.random_state,
            ),
        )

    def _load_validation(self) -> None:
        pass  # already loaded by _load()


class SingleTabbedDataset(TabbedDataset):
    """This class is for when you've got a single TSV of edges and want them to get auto-split."""

    ratios: ClassVar[Sequence[float]] = (0.8, 0.1, 0.1)
    _triples_factory: Optional[TriplesFactory]

    #: URL to the data to download
    url: str

    def __init__(
        self,
        url: str,
        name: Optional[str] = None,
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = None,
        download_kwargs: Optional[Dict[str, Any]] = None,
        read_csv_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize dataset.

        :param url:
            The url where to download the dataset from
        :param name:
            The name of the file. If not given, tries to get the name from the end of the URL
        :param cache_root:
            An optional directory to store the extracted files. Is none is given, the default PyKEEN directory is used.
            This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to ``~/.pykeen``.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param random_state: An optional random state to make the training/testing/validation split reproducible.
        :param download_kwargs: Keyword arguments to pass through to :func:`pystow.utils.download`.
        :param read_csv_kwargs: Keyword arguments to pass through to :func:`pandas.read_csv`.

        :raises ValueError: if there's no URL specified and there is no data already at the calculated path
        """
        super().__init__(
            cache_root=cache_root,
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            eager=False,  # because it gets hooked below
        )

        self.name = name or name_from_url(url)

        self.download_kwargs = download_kwargs or {}
        self.read_csv_kwargs = read_csv_kwargs or {}
        self.read_csv_kwargs.setdefault("sep", "\t")

        self.url = url
        if not self._get_path().is_file() and not self.url:
            raise ValueError(f"must specify url to download from since path does not exist: {self._get_path()}")

        if eager:
            self._load()

    def _get_path(self) -> pathlib.Path:
        return self.cache_root.joinpath(self.name)

    def _get_df(self) -> pd.DataFrame:
        if not self._get_path().is_file():
            logger.info("downloading data from %s to %s", self.url, self._get_path())
            download(url=self.url, path=self._get_path(), **self.download_kwargs)  # noqa:S310
        df = pd.read_csv(self._get_path(), **self.read_csv_kwargs)

        usecols = self.read_csv_kwargs.get("usecols")
        if usecols is not None:
            logger.info("reordering columns: %s", usecols)
            df = df[usecols]

        return df
