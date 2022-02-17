"""Utility classes for constructing datasets."""


import dataclasses
from typing import Optional

from class_resolver import OptionalKwargs

from pykeen.triples.triples_factory import CoreTriplesFactory


@dataclasses.dataclass
class Dataset:
    """A dataset is a collection of training and testing triples, and optionally validation triples."""

    training: CoreTriplesFactory
    testing: CoreTriplesFactory
    validation: Optional[CoreTriplesFactory] = None
    metadata: OptionalKwargs = None
