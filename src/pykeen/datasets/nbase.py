"""Utility classes for constructing datasets."""


import base64
import dataclasses
import hashlib
import logging
import pathlib
from abc import abstractmethod
from typing import Any, Mapping, MutableMapping, Optional, Tuple, Union

import pystow
import torch
from class_resolver import OptionalKwargs, normalize_string

from .nations import NATIONS_TRAIN_PATH
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
    metadata: MutableMapping[str, Any] = dataclasses.field(default_factory=dict)

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
    metadata: MutableMapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def name(self) -> str:
        """The canonical dataset name."""
        return normalize_string(
            self.metadata.get("name") or self.__class__.__name__, suffix=DatasetLoader.__class__.__name__
        )

    @property
    def cache_root(self) -> pystow.Module:
        """The data loader's cache root."""
        return pystow.Module(PYKEEN_DATASETS).submodule(self.name)

    def load(self, force: bool = False) -> Dataset:
        """Load the dataset."""
        # get canonical cache path
        path = self.cache_root.join("cache", _digest_kwargs(self.__dict__), ensure_exists=False)

        # try to use cached dataset
        if path.is_dir() and not force:
            logger.info(f"Loading cached preprocessed dataset from {path.as_uri()}")
            return Dataset.from_directory_binary(path)

        # load dataset without cache
        dataset_instance = self._load()
        # normalize name
        dataset_instance.metadata["name"] = normalize_string(dataset_instance.metadata["name"])

        # store cache
        logger.info(f"Caching preprocessed dataset to {path.as_uri()}")
        dataset_instance.to_directory_binary(path=path)

        return dataset_instance

    @abstractmethod
    def _load(self) -> Dataset:
        """Load the dataset without cache."""


@dataclasses.dataclass
class PreSplitDatasetLoader(DatasetLoader):
    """A loader of pre-split datasets."""

    training: pathlib.PurePath = None
    testing: pathlib.PurePath = None
    validation: Optional[pathlib.PurePath] = None

    def _load(self) -> Dataset:  # noqa: D102
        training = self._load_tf(relative_path=self.training)
        return Dataset(
            training=training,
            testing=self._load_tf(
                relative_path=self.testing,
                # share entity & relation index with training
                entity_to_id=training.entity_to_id,
                relation_to_id=training.relation_to_id,
            ),
            validation=self._load_tf(
                relative_path=self.validation,
                # share entity & relation index with training
                entity_to_id=training.entity_to_id,
                relation_to_id=training.relation_to_id,
            )
            if self.validation
            else None,
            metadata=self.metadata,
        )

    @abstractmethod
    def _load_tf(
        self,
        relative_path: pathlib.PurePath,
        entity_to_id: Optional[Mapping[str, int]] = None,
        relation_to_id: Optional[Mapping[str, int]] = None,
    ) -> TriplesFactory:
        raise NotImplementedError


@dataclasses.dataclass
class PathDatasetLoader(PreSplitDatasetLoader):
    """A loader of pre-split datasets."""

    root: pathlib.Path = None
    load_triples_kwargs: OptionalKwargs = None

    def _load_tf(
        self,
        relative_path: pathlib.PurePath,
        entity_to_id: Optional[Mapping[str, int]] = None,
        relation_to_id: Optional[Mapping[str, int]] = None,
    ) -> TriplesFactory:
        assert self.root is not None
        return TriplesFactory.from_path(
            path=self.root.joinpath(relative_path),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            # do not explicitly create inverse triples for testing; this is handled by the evaluation code
            create_inverse_triples=self.create_inverse_triples and entity_to_id is None and relation_to_id is None,
            load_triples_kwargs=self.load_triples_kwargs,
        )


@dataclasses.dataclass
class PackedZipRemoteDatasetLoader(PreSplitDatasetLoader):
    """Load a remote dataset contained inside a zipfile."""

    head_column: int = 0
    relation_column: int = 1
    tail_column: int = 2
    sep = "\t"
    header = None

    url: str = None
    file_name: Optional[str] = None

    def _load_tf(
        self,
        relative_path: pathlib.PurePath,
        entity_to_id: Optional[Mapping[str, int]] = None,
        relation_to_id: Optional[Mapping[str, int]] = None,
    ) -> TriplesFactory:
        return TriplesFactory.from_labeled_triples(
            triples=self.cache_root.ensure_zip_df(
                url=self.url,
                inner_path=relative_path.as_posix(),
                name=self.file_name,
                read_csv_kwargs=dict(
                    usecols=[self.head_column, self.relation_column, self.tail_column],
                    header=self.header,
                    sep=self.sep,
                ),
            ).values,
            create_inverse_triples=self.create_inverse_triples,
            metadata={"path": relative_path},
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )


nations_loader = PathDatasetLoader(
    root=NATIONS_TRAIN_PATH.parent,
    training=pathlib.PurePath("train.txt"),
    testing=pathlib.PurePath("test.txt"),
    validation=pathlib.PurePath("valid.txt"),
    metadata=dict(
        name="Nations",
        statistics=dict(
            entities=14,
            relations=55,
            training=1592,
            testing=201,
            validation=199,
            triples=1992,
        ),
        citation=dict(
            author="Zhenfeng Lei",
            year=2017,
            github="ZhenfengLei/KGDatasets",
        ),
    ),
)
fb15k237_loader = PackedZipRemoteDatasetLoader(
    metadata=dict(
        name="FB15k-237",
        statistics=dict(
            entities=14505,
            relations=237,
            training=272115,
            testing=20438,
            validation=17526,
            triples=310079,
        ),
        citation=dict(
            author="Toutanova",
            year=2015,
            link="https://www.aclweb.org/anthology/W15-4007/",
        ),
    ),
    training=pathlib.PurePath("Release", "train.txt"),
    testing=pathlib.PurePath("Release", "test.txt"),
    validation=pathlib.PurePath("Release", "valid.txt"),
    url="https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip",
)
