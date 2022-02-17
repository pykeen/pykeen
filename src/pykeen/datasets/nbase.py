"""Utility classes for constructing datasets."""


import dataclasses
from abc import abstractmethod
from typing import Optional
import pathlib

from class_resolver import OptionalKwargs

from ..triples import CoreTriplesFactory, TriplesFactory


@dataclasses.dataclass
class Dataset:
    """A dataset is a collection of training and testing triples, and optionally validation triples."""

    training: CoreTriplesFactory
    testing: CoreTriplesFactory
    validation: Optional[CoreTriplesFactory] = None
    metadata: OptionalKwargs = None


@dataclasses.dataclass
class DatasetLoader:
    """A loader for datasets."""

    create_inverse_triples: bool = False

    @abstractmethod
    def load(self) -> Dataset:
        """Load the dataset."""
        # TODO: add caching here
        raise NotImplementedError


@dataclasses.dataclass
class PathDatasetLoader(DatasetLoader):
    """A loader of pre-split datasets."""

    # TODO: non-default after default
    training_path: pathlib.Path
    testing_path: pathlib.Path
    validation_path: Optional[pathlib.Path] = None
    load_triples_kwargs: OptionalKwargs = None

    def load(self) -> Dataset:  # noqa: D102
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
