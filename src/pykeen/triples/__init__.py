# -*- coding: utf-8 -*-

"""Classes for creating and storing training data from triples."""

from .instances import Instances, LCWAInstances, SLCWAInstances, LCWAQuadrupleInstances, QuadrupleInstances, SLCWAQuadrupleInstances
from .quadruples_factory import QuadruplesFactory
from .triples_factory import AnyTriples, CoreTriplesFactory, KGInfo, TriplesFactory, get_mapped_triples
from .triples_numeric_literals_factory import TriplesNumericLiteralsFactory

__all__ = [
    "AnyTriples",
    "Instances",
    "CoreTriplesFactory",
    "LCWAInstances",
    "LCWAQuadrupleInstances",
    "SLCWAInstances",
    "SLCWAQuadrupleInstances",
    "KGInfo",
    "QuadrupleInstances",
    "QuadruplesFactory",
    "TriplesFactory",
    "TriplesNumericLiteralsFactory",
    "get_mapped_triples",
]
