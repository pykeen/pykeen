# -*- coding: utf-8 -*-
"""Sample datasets for use with POEM, borrowed from https://github.com/ZhenfengLei/KGDatasets."""

import logging
import os
import tempfile
from abc import abstractmethod
from io import BytesIO
from typing import Optional, Tuple

import requests

from ..instance_creation_factories import TriplesFactory

__all__ = [
    'DataSet',
    'RemoteDataSet',
]


class DataSet:
    """Contains a lazy reference to a training, testing, and validation data set."""

    training: Optional[TriplesFactory]
    testing: Optional[TriplesFactory]
    validation: Optional[TriplesFactory]

    _loaded: bool = False

    def __init__(
        self,
        training_path: str,
        testing_path: str,
        validation_path: str,
    ):
        self._loaded = False
        self.training_path = training_path
        self.testing_path = testing_path
        self.validation_path = validation_path
        self.training = None
        self.testing = None
        self.validation = None

    def load(self) -> 'DataSet':
        """Load the data sets."""
        if not self._loaded:
            self.training = TriplesFactory(path=self.training_path)
            self.testing = TriplesFactory(
                path=self.testing_path,
                entity_to_id=self.training.entity_to_id,
                relation_to_id=self.training.relation_to_id,
            )
            self.validation = TriplesFactory(
                path=self.validation_path,
                entity_to_id=self.training.entity_to_id,
                relation_to_id=self.training.relation_to_id,
            )
            self._loaded = True
        return self

    @property
    def entity_to_id(self):
        """Mapping of entity labels to IDs."""  # noqa: D401
        return self.training.entity_to_id

    @property
    def relation_to_id(self):
        """Mapping of relation labels to IDs."""  # noqa: D401
        return self.training.relation_to_id

    @property
    def num_entities(self):
        """The number of entities."""  # noqa: D401
        return self.training.num_entities

    @property
    def num_relations(self):
        """The number of relations."""  # noqa: D401
        return self.training.num_relations


class RemoteDataSet(DataSet):
    """Contains a lazy reference to a remove dataset that is loaded if needed."""

    def __init__(
        self,
        url: str,
        relative_training_path: str,
        relative_testing_path: str,
        relative_validation_path: str,
        cache_root: str = None,
    ):
        """
        Constructor.

        :param url:
            The url where to download the dataset from.
        :param cache_root:
            An optional directory to store the extracted files. Is none is given, the default tmp directory is used.
        """
        if cache_root is None:
            cache_root = tempfile.gettempdir()
        self.cache_root = os.path.join(cache_root, 'poem', self.__class__.__name__)
        self.url = url
        self._relative_training_path = relative_training_path
        self._relative_testing_path = relative_testing_path
        self._relative_validation_path = relative_validation_path

        training_path, testing_path, validation_path = self._get_paths()
        super().__init__(
            training_path=training_path,
            testing_path=testing_path,
            validation_path=validation_path,
        )

    def _get_paths(self) -> Tuple[str, str, str]:  # noqa: D401
        """The paths where the extracted files can be found."""
        return (
            self._get_rel_path(self._relative_training_path),
            self._get_rel_path(self._relative_testing_path),
            self._get_rel_path(self._relative_validation_path),
        )

    def _get_rel_path(self, path: str) -> str:
        """Construct a path relative to the cache root."""
        return os.path.join(self.cache_root, path)

    @abstractmethod
    def _extract(self, archive_file: BytesIO):
        """Extract from the downloaded file."""
        raise NotImplementedError

    def load(self) -> DataSet:  # noqa: D102
        if not all(map(os.path.isfile, self._get_paths())):
            logging.info(f'Requesting dataset from {self.url}')
            r = requests.get(url=self.url)
            assert r.status_code == requests.codes.ok
            archive_file = BytesIO(r.content)
            self._extract(archive_file=archive_file)
            logging.info(f'Extracted to {self.cache_root}.')
        return super().load()
