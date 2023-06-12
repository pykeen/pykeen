# -*- coding: utf-8 -*-

"""Classes for creating and storing training data from triples."""

from .instances import Instances, LCWAInstances, LCWAQuadrupleInstances, SLCWAInstances, SLCWAQuadrupleInstances, QuadrupleInstances
from .triples_factory import AnyTriples, CoreTriplesFactory, KGInfo, QuadruplesFactory, TriplesFactory,  get_mapped_triples
from .triples_numeric_literals_factory import TriplesNumericLiteralsFactory

__all__ = [
    "Instances",
    "LCWAInstances",
    "LCWAQuadrupleInstances",
    "SLCWAInstances",
    "SLCWAQuadrupleInstances",
    "KGInfo",
    "CoreTriplesFactory",
    "QuadrupleInstances",
    "QuadruplesFactory",
    "TriplesFactory",
    "TriplesNumericLiteralsFactory",
    "get_mapped_triples", 
    "AnyTriples",
]
