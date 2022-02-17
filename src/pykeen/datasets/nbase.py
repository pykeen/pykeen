"""Utility classes for constructing datasets."""


import dataclasses
from abc import abstractmethod
from typing import Optional

from class_resolver import OptionalKwargs

from ..triples import CoreTriplesFactory


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
        raise NotImplementedError
