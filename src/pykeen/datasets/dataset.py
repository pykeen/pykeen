# -*- coding: utf-8 -*-

"""Sample datasets for use with PyKEEN, borrowed from https://github.com/ZhenfengLei/KGDatasets."""

import logging
import os
import shutil
import tarfile
import zipfile
from abc import abstractmethod
from io import BytesIO
from typing import Optional, TextIO, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from tabulate import tabulate

from ..constants import PYKEEN_HOME
from ..triples import TriplesFactory

__all__ = [
    'DataSet',
    'PathDataSet',
    'RemoteDataSet',
    'TarFileRemoteDataSet',
    'ZipFileRemoteDataSet',
    'PackedZipRemoteDataSet',
    'SingleTabbedDataset',
]

logger = logging.getLogger(__name__)


class DataSet:
    """Contains a lazy reference to a training, testing, and validation data set."""

    #: A factory wrapping the training triples
    training: TriplesFactory
    #: A factory wrapping the testing triples, that share indexes with the training triples
    testing: TriplesFactory
    #: A factory wrapping the validation triples, that share indexes with the training triples
    validation: TriplesFactory
    #: All data sets should take care of inverse triple creation
    create_inverse_triples: bool

    @property
    def factories(self) -> Tuple[TriplesFactory, TriplesFactory, TriplesFactory]:
        """Return a tuple of three factories in order (training, testing, validation)."""
        return self.training, self.testing, self.validation

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

    def summary_str(self) -> str:
        """Make a summary string of all of the factories."""
        t = tabulate(
            [
                (label, triples_factory.num_entities, triples_factory.num_relations, triples_factory.num_triples)
                for label, triples_factory in
                zip(('Training', 'Testing', 'Validation'), (self.training, self.testing, self.validation))
            ],
            headers=['Name', 'Entities', 'Relations', 'Triples'],
        )
        return f'{self.__class__.__name__} (create_inverse_triples={self.create_inverse_triples})\n{t}'

    def summarize(self) -> None:
        """Print a summary of the dataset."""
        print(self.summary_str())

    def __str__(self) -> str:  # noqa: D105
        return f'{self.__class__.__name__}(num_entities={self.num_entities}, num_relations={self.num_relations})'


class LazyDataSet(DataSet):
    """A data set that has lazy loading."""

    #: The actual instance of the training factory, which is exposed to the user through `training`
    _training: Optional[TriplesFactory] = None
    #: The actual instance of the testing factory, which is exposed to the user through `testing`
    _testing: Optional[TriplesFactory] = None
    #: The actual instance of the validation factory, which is exposed to the user through `validation`
    _validation: Optional[TriplesFactory] = None

    @property
    def training(self) -> TriplesFactory:  # noqa: D401
        """The training triples factory."""
        if not self._loaded:
            self._load()
        return self._training

    @property
    def testing(self) -> TriplesFactory:  # noqa: D401
        """The testing triples factory that shares indexes with the training triples factory."""
        if not self._loaded:
            self._load()
        return self._testing

    @property
    def validation(self) -> TriplesFactory:  # noqa: D401
        """The validation triples factory that shares indexes with the training triples factory."""
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
        return f'{self.__class__.__name__}(training_path="{self.training_path}", testing_path="{self.testing_path}",' \
               f' validation_path="{self.validation_path}")'


def _urlretrieve(url, path, clean_on_failure: bool = True) -> None:
    """Download a file from a given URL.

    :param url: URL to download
    :param path: Path to download the file to
    :param clean_on_failure: If true, will delete the file on any exception raised during download
    """
    # see https://requests.readthedocs.io/en/master/user/quickstart/#raw-response-content
    # pattern from https://stackoverflow.com/a/39217788/5775947
    try:
        with requests.get(url, stream=True) as response, open(path, 'wb') as file:
            shutil.copyfileobj(response.raw, file)
    except (Exception, KeyboardInterrupt):
        if clean_on_failure:
            os.remove(path)
        raise


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
        """
        if cache_root is None:
            cache_root = PYKEEN_HOME
        self.cache_root = os.path.join(cache_root, self.__class__.__name__.lower())
        os.makedirs(cache_root, exist_ok=True)

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

    def _load(self) -> None:  # noqa: D102
        all_unpacked = all(
            os.path.exists(path) and os.path.isfile(path)
            for path in self._get_paths()
        )

        if not all_unpacked:
            logger.info(f'Requesting dataset from {self.url}')

            r = requests.get(url=self.url)
            assert r.status_code == requests.codes.ok
            archive_file = BytesIO(r.content)
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
        """
        if cache_root is None:
            cache_root = os.path.join(PYKEEN_HOME, self.__class__.__name__.lower())
        self.cache_root = cache_root
        os.makedirs(cache_root, exist_ok=True)
        logger.debug('using cache root at %s', cache_root)

        if name is None:
            parse_result = urlparse(url)
            name = os.path.basename(parse_result.path)
            logger.info('parsed name from URL: %s', name)
        self.name = name
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
                    create_inverse_triples=self.create_inverse_triples
                )
                rv.path = relative_path
                return rv


class SingleTabbedDataset(LazyDataSet):
    """This class is for when you've got a single TSV of edges and want them to get auto-split."""

    ratios = (0.8, 0.1, 0.1)
    _triples_factory: Optional[TriplesFactory]

    def __init__(
        self,
        url: Optional[str] = None,
        name: Optional[str] = None,
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
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
        """
        if cache_root is None:
            cache_root = os.path.join(PYKEEN_HOME, self.__class__.__name__.lower())
        self.cache_root = cache_root
        os.makedirs(cache_root, exist_ok=True)
        logger.debug('using cache root at %s', cache_root)

        if name is None:
            parse_result = urlparse(url)
            name = os.path.basename(parse_result.path)
            logger.info('parsed name from URL: %s', name)
        self.name = name
        self.path = os.path.join(self.cache_root, self.name)
        logger.debug('file path at %s', self.path)

        self._triples_factory = None
        self.random_state = random_state

        self.url = url
        if not os.path.exists(self.path) and not self.url:
            raise ValueError(f'must specify url to download from since path does not exist: {self.path}')

        self.create_inverse_triples = create_inverse_triples
        self._training = None
        self._testing = None
        self._validation = None

        if eager:
            self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            logger.info('downloading data from %s to %s', self.url, self.path)
            _urlretrieve(self.url, self.path)  # noqa:S310
        df = pd.read_csv(self.path, sep='\t')
        tf = TriplesFactory(triples=df.values, create_inverse_triples=self.create_inverse_triples)
        tf.path = self.path
        self._training, self._testing, self._validation = tf.split(
            ratios=self.ratios,
            random_state=self.random_state,
        )

    def _load_validation(self) -> None:
        pass  # already loaded by _load()
