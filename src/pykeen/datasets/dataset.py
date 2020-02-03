# -*- coding: utf-8 -*-

"""Sample datasets for use with POEM, borrowed from https://github.com/ZhenfengLei/KGDatasets."""

import logging
import os
import tarfile
import zipfile
from abc import abstractmethod
from io import BytesIO
from typing import Optional, TextIO, Tuple, Union

import requests

from ..triples import TriplesFactory

__all__ = [
    'DataSet',
    'RemoteDataSet',
    'TarFileRemoteDataSet',
    'ZipFileRemoteDataSet',
]

logger = logging.getLogger(__name__)


class DataSet:
    """Contains a lazy reference to a training, testing, and validation data set."""

    _training: Optional[TriplesFactory]
    _testing: Optional[TriplesFactory]
    _validation: Optional[TriplesFactory]

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

        self._create_inverse_triples = create_inverse_triples
        self._training = None
        self._testing = None
        self._validation = None

        if eager:
            self._load()
            self._load_validation()

    @property
    def _loaded(self) -> bool:
        return self._training is not None and self._testing is not None

    @property
    def _loaded_validation(self):
        return self._validation is not None

    def _load(self) -> None:
        self._training = TriplesFactory(
            path=self.training_path,
            create_inverse_triples=self._create_inverse_triples,
        )
        self._testing = TriplesFactory(
            path=self.testing_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
        )

    def _load_validation(self) -> None:
        if not self._loaded:
            self._load()

        self._validation = TriplesFactory(
            path=self.validation_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
        )

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
        if not self._loaded_validation:
            self._load_validation()
        return self._validation

    @property
    def factories(self) -> Tuple[TriplesFactory, TriplesFactory, TriplesFactory]:
        """Return all three factories."""
        return self.training, self.testing, self.validation

    @property
    def entity_to_id(self):  # noqa: D401
        """Mapping of entity labels to IDs."""
        return self.training.entity_to_id

    @property
    def relation_to_id(self):  # noqa: D401
        """Mapping of relation labels to IDs."""
        return self.training.relation_to_id

    @property
    def num_entities(self):  # noqa: D401
        """The number of entities."""
        return self.training.num_entities

    @property
    def num_relations(self):  # noqa: D401
        """The number of relations."""
        return self.training.num_relations

    def __str__(self) -> str:  # noqa: D105
        return f'{self.__class__.__name__}(num_entities={self.num_entities}, num_relations={self.num_relations})'

    def __repr__(self) -> str:  # noqa: D105
        return f'{self.__class__.__name__}(training_path="{self.training_path}", testing_path="{self.testing_path}",' \
               f' validation_path="{self.validation_path}")'


class RemoteDataSet(DataSet):
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
            An optional directory to store the extracted files. Is none is given, the default tmp directory is used.
        """
        if cache_root is None:
            cache_root = os.path.join(os.path.expanduser('~'), 'pykeen')
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


class ZipFileRemoteDataSet(RemoteDataSet):
    """A remote dataset stored as a zip file."""

    def _extract(self, archive_file: BytesIO) -> None:  # noqa: D102
        with zipfile.ZipFile(file=archive_file) as zf:
            zf.extractall(path=self.cache_root)
