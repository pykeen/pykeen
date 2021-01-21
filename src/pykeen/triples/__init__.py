# -*- coding: utf-8 -*-

"""Classes for creating and storing training data from triples."""

from .instances import (
    Instances, LCWAInstances, MultimodalInstances, MultimodalLCWAInstances, MultimodalSLCWAInstances, SLCWAInstances,
)
from .triples_factory import CoreTriplesFactory, TriplesFactory
from .triples_numeric_literals_factory import TriplesNumericLiteralsFactory

__all__ = [
    'Instances',
    'LCWAInstances',
    'MultimodalInstances',
    'MultimodalSLCWAInstances',
    'MultimodalLCWAInstances',
    'SLCWAInstances',
    'CoreTriplesFactory',
    'TriplesFactory',
    'TriplesNumericLiteralsFactory',
]
