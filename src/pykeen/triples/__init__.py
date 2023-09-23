# -*- coding: utf-8 -*-

"""Classes for creating and storing training data from triples."""

from .instances import Instances, LCWAInstances, SLCWAInstances
from .triples_factory import AnyTriples, CoreTriplesFactory, KGInfo, TriplesFactory, get_mapped_triples
from .triples_numeric_literals_factory import TriplesNumericLiteralsFactory

__all__ = [
    "Instances",
    "LCWAInstances",
    "SLCWAInstances",
    "KGInfo",
    "CoreTriplesFactory",
    "TriplesFactory",
    "TriplesNumericLiteralsFactory",
    "get_mapped_triples",
    "AnyTriples",
]
