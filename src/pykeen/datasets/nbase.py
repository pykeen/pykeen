"""Utility classes for constructing datasets."""


import base64
import dataclasses
import hashlib
import logging
import pathlib
from abc import abstractmethod
from typing import Any, ClassVar, Mapping, MutableMapping, Optional, Tuple, Union

import torch
from class_resolver import OptionalKwargs, normalize_string

from ..constants import PYKEEN_DATASETS
from ..triples import CoreTriplesFactory, TriplesFactory

logger = logging.getLogger(__name__)


KEYS = ("training", "testing", "validation")


@dataclasses.dataclass
class Dataset:
    """A dataset is a collection of training and testing triples, and optionally validation triples."""

    training: CoreTriplesFactory
    testing: CoreTriplesFactory
    validation: Optional[CoreTriplesFactory] = None
    metadata: MutableMapping[str, Any] = dataclasses.Field(default_factory=dict)

    @staticmethod
    def meta_path(root: pathlib.Path) -> pathlib.Path:
        return root.joinpath("metadata.pth")

    @property
    def factory_tuple(self) -> Tuple[CoreTriplesFactory, ...]:
        """Return a tuple of the three factories."""
        res = (self.training, self.testing)
        if self.validation:
            res = res + (self.validation,)
        return res

    @property
    def factory_dict(self) -> Mapping[str, CoreTriplesFactory]:
        """Return a dictionary of the three factories."""
        return dict(zip(KEYS, self.factory_tuple))

    @classmethod
    def from_directory_binary(cls, path: Union[str, pathlib.Path]) -> "Dataset":
        """Load a dataset from a directory."""
        path = pathlib.Path(path)

        if not path.is_dir():
            raise NotADirectoryError(path)

        tfs = dict()
        for key in KEYS:
            tf_path = path.joinpath(key)
            if tf_path.is_dir():
                tfs[key] = TriplesFactory.from_path_binary(path=tf_path)
            else:
                logger.warning(f"{tf_path.as_uri()} does not exist.")
        metadata_path = cls.meta_path(root=path)
        metadata = torch.load(metadata_path)
        return Dataset(**tfs, metadata=metadata)

    def to_directory_binary(self, path: Union[str, pathlib.Path]) -> None:
        """Store a dataset to a path in binary format."""
        path = pathlib.Path(path)
        for key, factory in self.factory_dict.items():
            tf_path = path.joinpath(key)
            factory.to_path_binary(tf_path)
            logger.info(f"Stored {key} factory to {tf_path.as_uri()}")
        torch.save(self.metadata, self.meta_path(root=path))


def _digest_kwargs(dataset_kwargs: Mapping[str, Any]) -> str:
    digester = hashlib.sha256()
    for key in sorted(dataset_kwargs.keys()):
        digester.update(key.encode(encoding="utf8"))
        digester.update(str(dataset_kwargs[key]).encode(encoding="utf8"))
    return base64.urlsafe_b64encode(digester.digest()).decode("utf8")[:32]


@dataclasses.dataclass
class DatasetLoader:
    """A loader for datasets."""

    create_inverse_triples: bool = False

    def _digest(self) -> str:
        dataset_kwargs = ...
        return _digest_kwargs(dataset_kwargs)

    @property
    def name(self) -> str:
        """The canonical dataset name."""
        return normalize_string(self.__class__.__name__, suffix=DatasetLoader.__class__.__name__)

    def load(self, force: bool = False) -> Dataset:
        """Load the dataset."""
        # get canonical cache path
        path = PYKEEN_DATASETS.joinpath(self.name, "cache", self._digest())

        # try to use cached dataset
        if path.is_dir() and not force:
            logger.info(f"Loading cached preprocessed dataset from {path.as_uri()}")
            return Dataset.from_directory_binary(path)

        # load dataset without cache
        dataset_instance = self._load()
        dataset_instance.metadata["name"] = self.name

        # store cache
        logger.info(f"Caching preprocessed dataset to {path.as_uri()}")
        dataset_instance.to_directory_binary(path=path)

        return dataset_instance

    @abstractmethod
    def _load(self) -> Dataset:
        """Load the dataset without cache."""


@dataclasses.dataclass
class PathDatasetLoader(DatasetLoader):
    """A loader of pre-split datasets."""

    training_path: pathlib.Path = dataclasses.field(init=False)
    testing_path: pathlib.Path = dataclasses.field(init=False)
    validation_path: Optional[pathlib.Path] = None
    load_triples_kwargs: OptionalKwargs = None

    def _load(self) -> Dataset:  # noqa: D102
        training = TriplesFactory.from_path(
            path=self.training_path,
            create_inverse_triples=self.create_inverse_triples,
            load_triples_kwargs=self.load_triples_kwargs,
        )
        return Dataset(
            training=training,
            testing=TriplesFactory.from_path(
                path=self.testing_path,
                entity_to_id=training.entity_to_id,  # share entity index with training
                relation_to_id=training.relation_to_id,  # share relation index with training
                # do not explicitly create inverse triples for testing; this is handled by the evaluation code
                create_inverse_triples=False,
                load_triples_kwargs=self.load_triples_kwargs,
            ),
            validation=TriplesFactory.from_path(
                path=self.validation_path,
                entity_to_id=training.entity_to_id,  # share entity index with training
                relation_to_id=training.relation_to_id,  # share relation index with training
                # do not explicitly create inverse triples for testing; this is handled by the evaluation code
                create_inverse_triples=False,
                load_triples_kwargs=self.load_triples_kwargs,
            )
            if self.validation_path
            else None,
        )
