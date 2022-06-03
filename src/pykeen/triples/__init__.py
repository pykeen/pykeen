# -*- coding: utf-8 -*-

"""Classes for creating and storing training data from triples."""

from .instances import Instances, LCWAInstances, SLCWAInstances
from .triples_factory import CoreTriplesFactory, KGInfo, TriplesFactory
from .triples_numeric_literals_factory import TriplesNumericLiteralsFactory

__all__ = [
    "Instances",
    "LCWAInstances",
    "SLCWAInstances",
    "KGInfo",
    "CoreTriplesFactory",
    "TriplesFactory",
    "TriplesNumericLiteralsFactory",
]
