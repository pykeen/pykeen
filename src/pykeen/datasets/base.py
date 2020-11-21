# -*- coding: utf-8 -*-

"""Utility classes for constructing datasets."""

import logging
import os
import pathlib
import shutil
import tarfile
import zipfile
from abc import abstractmethod
from io import BytesIO
from typing import List, Optional, TextIO, Tuple, Union
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import requests
from tabulate import tabulate

from ..constants import PYKEEN_HOME
from ..triples import TriplesFactory
from ..utils import normalize_string

__all__ = [
    'DataSet',
    'EagerDataset',
    'LazyDataSet',
    'PathDataSet',
    'RemoteDataSet',
    'UnpackedRemoteDataSet',
    'TarFileRemoteDataSet',
    'ZipFileRemoteDataSet',
    'PackedZipRemoteDataSet',
    'TarFileSingleDataset',
    'SingleTabbedDataset',
]

logger = logging.getLogger(__name__)


class DataSet:
    """Contains a lazy reference to a training, testing, and validation data set."""

    #: A factory wrapping the training triples
    training: TriplesFactory
    #: A factory wrapping the testing triples, that share indices with the training triples
    testing: TriplesFactory
    #: A factory wrapping the validation triples, that share indices with the training triples
    validation: TriplesFactory
    #: All data sets should take care of inverse triple creation
    create_inverse_triples: bool

    @property
    def entity_to_id(self):  # noqa: D401
        """The mapping of entity labels to IDs."""
        return self.training.entity_to_id

    @property
    def relation_to_id(self):  # noqa: D401
        """The mapping of relation labels to IDs."""
        return self.training.relation_to_id

    @property
    def num_entities(self):  # noqa: D401
        """The number of entities."""
        return self.training.num_entities

    @property
    def num_relations(self):  # noqa: D401
        """The number of relations."""
        return self.training.num_relations

    def summary_str(self, end='\n') -> str:
        """Make a summary string of all of the factories."""
        rows = [
            (label, triples_factory.num_entities, triples_factory.num_relations, triples_factory.num_triples)
            for label, triples_factory in
            zip(('Training', 'Testing', 'Validation'), (self.training, self.testing, self.validation))
        ]
        n_triples = sum(
            triples_factory.num_triples
            for triples_factory in (self.training, self.testing, self.validation)
        )
        rows.append(('Total', '-', '-', n_triples))
        t = tabulate(rows, headers=['Name', 'Entities', 'Relations', 'Triples'])
        return f'{self.__class__.__name__} (create_inverse_triples={self.create_inverse_triples})\n{t}{end}'

    def summarize(self) -> None:
        """Print a summary of the dataset."""
        print(self.summary_str())

    def __str__(self) -> str:  # noqa: D105
        return f'{self.__class__.__name__}(num_entities={self.num_entities}, num_relations={self.num_relations})'

    @classmethod
    def from_path(cls, path: str, ratios: Optional[List[float]] = None) -> 'DataSet':
        """Create a dataset from a single triples factory by splitting it in 3."""
        tf = TriplesFactory(path=path)
        return cls.from_tf(tf=tf, ratios=ratios)

    @staticmethod
    def from_tf(tf: TriplesFactory, ratios: Optional[List[float]] = None) -> 'DataSet':
        """Create a dataset from a single triples factory by splitting it in 3."""
        training, testing, validation = tf.split(ratios or [0.8, 0.1, 0.1])
        return EagerDataset(training=training, testing=testing, validation=validation)

    @classmethod
    def get_normalized_name(cls) -> str:
        """Get the normalized name of the dataset."""
        return normalize_string(cls.__name__)


class EagerDataset(DataSet):
    """A dataset that has already been loaded."""

    def __init__(self, training: TriplesFactory, testing: TriplesFactory, validation: TriplesFactory) -> None:
        self.training = training
        self.testing = testing
        self.validation = validation
        self.create_inverse_triples = (
            training.create_inverse_triples
            and testing.create_inverse_triples
            and self.validation.create_inverse_triples
        )


class LazyDataSet(DataSet):
    """A data set that has lazy loading."""

    #: The actual instance of the training factory, which is exposed to the user through `training`
    _training: Optional[TriplesFactory] = None
    #: The actual instance of the testing factory, which is exposed to the user through `testing`
    _testing: Optional[TriplesFactory] = None
    #: The actual instance of the validation factory, which is exposed to the user through `validation`
    _validation: Optional[TriplesFactory] = None
    #: The directory in which the cached data is stored
    cache_root: pathlib.Path

    @property
    def training(self) -> TriplesFactory:  # noqa: D401
        """The training triples factory."""
        if not self._loaded:
            self._load()
        return self._training

    @property
    def testing(self) -> TriplesFactory:  # noqa: D401
        """The testing triples factory that shares indices with the training triples factory."""
        if not self._loaded:
            self._load()
        return self._testing

    @property
    def validation(self) -> TriplesFactory:  # noqa: D401
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

    def _help_cache(self, cache_root: Optional[str]) -> pathlib.Path:
        """Get the appropriate cache root directory.

        :param cache_root: If none is passed, defaults to a subfolder of the
            PyKEEN home directory defined in :data:`pykeen.constants.PYKEEN_HOME`.
            The subfolder is named based on the class inheriting from
            :class:`pykeen.datasets.base.DataSet`.
        """
        if cache_root is None:
            cache_root = PYKEEN_HOME
        cache_root = pathlib.Path(cache_root) / self.__class__.__name__.lower()
        cache_root.mkdir(parents=True, exist_ok=True)
        logger.debug('using cache root at %s', cache_root)
        return cache_root


class PathDataSet(LazyDataSet):
    """Contains a lazy reference to a training, testing, and validation data set."""

    def __init__(
        self,
        training_path: Union[str, TextIO],
        testing_path: Union[str, TextIO],
        validation_path: Union[str, TextIO],
        eager: bool = False,
        create_inverse_triples: bool = False,
    ) -> None:
        """Initialize the data set.

        :param training_path: Path to the training triples file or training triples file.
        :param testing_path: Path to the testing triples file or testing triples file.
        :param validation_path: Path to the validation triples file or validation triples file.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        """
        self.training_path = training_path
        self.testing_path = testing_path
        self.validation_path = validation_path

        self.create_inverse_triples = create_inverse_triples

        if eager:
            self._load()
            self._load_validation()

    def _load(self) -> None:
        self._training = TriplesFactory(
            path=self.training_path,
            create_inverse_triples=self.create_inverse_triples,
        )
        self._testing = TriplesFactory(
            path=self.testing_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
        )

    def _load_validation(self) -> None:
        # don't call this function by itself. assumes called through the `validation`
        # property and the _training factory has already been loaded
        self._validation = TriplesFactory(
            path=self.validation_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f'{self.__class__.__name__}(training_path="{self.training_path}", testing_path="{self.testing_path}",'
            f' validation_path="{self.validation_path}")'
        )


def _name_from_url(url: str) -> str:
    """Get the filename form the end of the URL."""
    parse_result = urlparse(url)
    path = pathlib.PurePosixPath(parse_result.path)
    name = path.name
    logger.debug('parsed name from URL: %s', name)
    return name


def _urlretrieve(url: str, path: str, clean_on_failure: bool = True, stream: bool = True) -> None:
    """Download a file from a given URL.

    :param url: URL to download
    :param path: Path to download the file to
    :param clean_on_failure: If true, will delete the file on any exception raised during download
    """
    if not stream:
        logger.info('downloading from %s to %s', url, path)
        urlretrieve(url, path)  # noqa:S310
    else:
        # see https://requests.readthedocs.io/en/master/user/quickstart/#raw-response-content
        # pattern from https://stackoverflow.com/a/39217788/5775947
        try:
            with requests.get(url, stream=True) as response, open(path, 'wb') as file:
                logger.info('downloading (streaming) from %s to %s', url, path)
                shutil.copyfileobj(response.raw, file)
        except (Exception, KeyboardInterrupt):
            if clean_on_failure:
                os.remove(path)
            raise


class UnpackedRemoteDataSet(PathDataSet):
    """A dataset with all three of train, test, and validation sets as URLs."""

    def __init__(
        self,
        training_url: str,
        testing_url: str,
        validation_url: str,
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        stream: bool = True,
        force: bool = False,
    ):
        """Initialize dataset.

        :param training_url: The URL of the training file
        :param testing_url: The URL of the testing file
        :param validation_url: The URL of the validation file
        :param cache_root:
            An optional directory to store the extracted files. Is none is given, the default PyKEEN directory is used.
            This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to ``~/.pykeen``.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param stream:
        :param force:
        """
        self.cache_root = self._help_cache(cache_root)

        self.training_url = training_url
        self.testing_url = testing_url
        self.validation_url = validation_url

        training_path = os.path.join(self.cache_root, _name_from_url(self.training_url))
        testing_path = os.path.join(self.cache_root, _name_from_url(self.testing_url))
        validation_path = os.path.join(self.cache_root, _name_from_url(self.validation_url))

        for url, path in [
            (self.training_url, training_path),
            (self.testing_url, testing_path),
            (self.validation_url, validation_path),
        ]:
            if os.path.exists(path) and not force:
                continue
            _urlretrieve(url, path, stream=stream)

        super().__init__(
            training_path=training_path,
            testing_path=testing_path,
            validation_path=validation_path,
            eager=eager,
            create_inverse_triples=create_inverse_triples,
        )


class RemoteDataSet(PathDataSet):
    """Contains a lazy reference to a remote dataset that is loaded if needed."""

    def __init__(
        self,
        url: str,
        relative_training_path: str,
        relative_testing_path: str,
        relative_validation_path: str,
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
    ):
        """Initialize dataset.

        :param url:
            The url where to download the dataset from.
        :param cache_root:
            An optional directory to store the extracted files. Is none is given, the default PyKEEN directory is used.
            This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to ``~/.pykeen``.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        """
        self.cache_root = self._help_cache(cache_root)

        self.url = url
        self._relative_training_path = relative_training_path
        self._relative_testing_path = relative_testing_path
        self._relative_validation_path = relative_validation_path

        training_path, testing_path, validation_path = self._get_paths()
        super().__init__(
            training_path=training_path,
            testing_path=testing_path,
            validation_path=validation_path,
            eager=eager,
            create_inverse_triples=create_inverse_triples,
        )

    def _get_paths(self) -> Tuple[str, str, str]:  # noqa: D401
        """The paths where the extracted files can be found."""
        return (
            os.path.join(self.cache_root, self._relative_training_path),
            os.path.join(self.cache_root, self._relative_testing_path),
            os.path.join(self.cache_root, self._relative_validation_path),
        )

    @abstractmethod
    def _extract(self, archive_file: BytesIO) -> None:
        """Extract from the downloaded file."""
        raise NotImplementedError

    def _get_bytes(self) -> BytesIO:
        logger.info(f'Requesting dataset from {self.url}')
        res = requests.get(url=self.url)
        res.raise_for_status()
        return BytesIO(res.content)

    def _load(self) -> None:  # noqa: D102
        all_unpacked = all(
            os.path.exists(path) and os.path.isfile(path)
            for path in self._get_paths()
        )

        if not all_unpacked:
            archive_file = self._get_bytes()
            self._extract(archive_file=archive_file)
            logger.info(f'Extracted to {self.cache_root}.')

        super()._load()


class TarFileRemoteDataSet(RemoteDataSet):
    """A remote dataset stored as a tar file."""

    def _extract(self, archive_file: BytesIO) -> None:  # noqa: D102
        with tarfile.open(fileobj=archive_file) as tf:
            tf.extractall(path=self.cache_root)


# TODO replace this with the new zip remote dataset class
class ZipFileRemoteDataSet(RemoteDataSet):
    """A remote dataset stored as a zip file."""

    def _extract(self, archive_file: BytesIO) -> None:  # noqa: D102
        with zipfile.ZipFile(file=archive_file) as zf:
            zf.extractall(path=self.cache_root)


class PackedZipRemoteDataSet(LazyDataSet):
    """Contains a lazy reference to a remote dataset that is loaded if needed."""

    head_column: int = 0
    relation_column: int = 1
    tail_column: int = 2
    sep = '\t'
    header = None

    def __init__(
        self,
        relative_training_path: str,
        relative_testing_path: str,
        relative_validation_path: str,
        url: Optional[str] = None,
        name: Optional[str] = None,
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
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
        """
        self.cache_root = self._help_cache(cache_root)

        self.name = name or _name_from_url(url)
        self.path = os.path.join(self.cache_root, self.name)
        logger.debug('file path at %s', self.path)

        self.url = url
        if not os.path.exists(self.path) and not self.url:
            raise ValueError(f'must specify url to download from since path does not exist: {self.path}')

        self.relative_training_path = relative_training_path
        self.relative_testing_path = relative_testing_path
        self.relative_validation_path = relative_validation_path
        self.create_inverse_triples = create_inverse_triples
        if eager:
            self._load()
            self._load_validation()

    def _load(self) -> None:  # noqa: D102
        self._training = self._load_helper(self.relative_training_path)
        self._testing = self._load_helper(self.relative_testing_path)

    def _load_validation(self) -> None:
        self._validation = self._load_helper(self.relative_validation_path)

    def _load_helper(self, relative_path) -> TriplesFactory:
        if not os.path.exists(self.path):
            logger.info('downloading data from %s to %s', self.url, self.path)
            _urlretrieve(self.url, self.path)  # noqa:S310

        with zipfile.ZipFile(file=self.path) as zf:
            with zf.open(relative_path) as file:
                logger.debug('loading %s', relative_path)
                df = pd.read_csv(
                    file,
                    usecols=[self.head_column, self.relation_column, self.tail_column],
                    header=self.header,
                    sep=self.sep,
                )
                rv = TriplesFactory(
                    triples=df.values,
                    create_inverse_triples=self.create_inverse_triples,
                )
                rv.path = relative_path
                return rv


class TarFileSingleDataset(LazyDataSet):
    """Loads a dataset that's a single file inside a tar.gz archive."""

    ratios = (0.8, 0.1, 0.1)
    _triples_factory: Optional[TriplesFactory]

    def __init__(
        self,
        url: str,
        relative_path: str,
        name: Optional[str] = None,
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        delimiter: Optional[str] = None,
        random_state: Union[None, int, np.random.RandomState] = None,
        randomize_cleanup: bool = False,
    ):
        """Initialize dataset.

        :param url:
            The url where to download the dataset from
        :param name:
            The name of the file. If not given, tries to get the name from the end of the URL
        :param cache_root:
            An optional directory to store the extracted files. Is none is given, the default PyKEEN directory is used.
            This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to ``~/.pykeen``.
        :param relative_path:
            The path inside the archive to the contained dataset.
        :param random_state:
            An optional random state to make the training/testing/validation split reproducible.
        :param delimiter:
            The delimiter for the contained dataset.
        """
        self.cache_root = self._help_cache(cache_root)

        self.name = name or _name_from_url(url)
        self.random_state = random_state
        self.delimiter = delimiter or '\t'
        self.randomize_cleanup = randomize_cleanup
        self.url = url
        self.create_inverse_triples = create_inverse_triples
        self._relative_path = relative_path

        if eager:
            self._load()

    def _get_path(self) -> str:
        return os.path.join(self.cache_root, self.name)

    def _load(self) -> None:
        if not os.path.exists(self._get_path()):
            _urlretrieve(self.url, self._get_path())  # noqa:S310

        _actual_path = os.path.join(self.cache_root, self._relative_path)
        if not os.path.exists(_actual_path):
            logger.error(
                '[%s] untaring from %s (%s) to %s',
                self.__class__.__name__,
                self._get_path(),
                self._relative_path,
                _actual_path,
            )
            with tarfile.open(self._get_path()) as tf:
                tf.extract(self._relative_path, self.cache_root)

        df = pd.read_csv(_actual_path, sep=self.delimiter)
        tf = TriplesFactory(triples=df.values, create_inverse_triples=self.create_inverse_triples)
        tf.path = self._get_path()
        self._training, self._testing, self._validation = tf.split(
            ratios=self.ratios,
            random_state=self.random_state,
            randomize_cleanup=self.randomize_cleanup,
        )
        logger.info('[%s] done splitting data from %s', self.__class__.__name__, tf.path)

    def _load_validation(self) -> None:
        pass  # already loaded by _load()


class SingleTabbedDataset(LazyDataSet):
    """This class is for when you've got a single TSV of edges and want them to get auto-split."""

    ratios = (0.8, 0.1, 0.1)
    _triples_factory: Optional[TriplesFactory]

    def __init__(
        self,
        url: str,
        name: Optional[str] = None,
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        delimiter: Optional[str] = None,
        random_state: Union[None, int, np.random.RandomState] = None,
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
        """
        self.cache_root = self._help_cache(cache_root)

        self.name = name or _name_from_url(url)

        self._triples_factory = None
        self.random_state = random_state
        self.delimiter = delimiter or '\t'

        self.url = url
        if not os.path.exists(self._get_path()) and not self.url:
            raise ValueError(f'must specify url to download from since path does not exist: {self._get_path()}')

        self.create_inverse_triples = create_inverse_triples
        self._training = None
        self._testing = None
        self._validation = None

        if eager:
            self._load()

    def _get_path(self) -> str:
        return os.path.join(self.cache_root, self.name)

    def _load(self) -> None:
        if not os.path.exists(self._get_path()):
            logger.info('downloading data from %s to %s', self.url, self._get_path())
            _urlretrieve(self.url, self._get_path())  # noqa:S310
        df = pd.read_csv(self._get_path(), sep=self.delimiter)
        tf = TriplesFactory(triples=df.values, create_inverse_triples=self.create_inverse_triples)
        tf.path = self._get_path()
        self._training, self._testing, self._validation = tf.split(
            ratios=self.ratios,
            random_state=self.random_state,
        )

    def _load_validation(self) -> None:
        pass  # already loaded by _load()
