# -*- coding: utf-8 -*-

"""Sample datasets for use with POEM, borrowed from https://github.com/ZhenfengLei/KGDatasets."""

from dataclasses import dataclass, field
from typing import Type

from ..instance_creation_factories import TriplesFactory

__all__ = [
    'DataSet',
]


@dataclass
class DataSet:
    """Contains a lazy reference to a training, testing, and validation data set."""

    training_cls: Type[TriplesFactory]
    testing_cls: Type[TriplesFactory]
    validation_cls: Type[TriplesFactory]

    training: TriplesFactory = field(init=False)
    testing: TriplesFactory = field(init=False)
    validation: TriplesFactory = field(init=False)

    _loaded: bool = False

    def load(self) -> 'DataSet':
        """Load the data sets."""
        if not self._loaded:
            self.training = self.training_cls()
            self.testing = self.testing_cls()
            self.validation = self.validation_cls()
            self._loaded = True
        return self
