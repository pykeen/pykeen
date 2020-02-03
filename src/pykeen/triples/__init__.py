# -*- coding: utf-8 -*-

"""Classes for creating and storing training data from triples."""

from .instances import (
    Instances, LCWAInstances, MultimodalInstances, MultimodalLCWAInstances, MultimodalOWAInstances, OWAInstances,
)
from .triples_factory import TriplesFactory
from .triples_numeric_literals_factory import TriplesNumericLiteralsFactory

__all__ = [
    'Instances',
    'LCWAInstances',
    'MultimodalInstances',
    'MultimodalOWAInstances',
    'MultimodalLCWAInstances',
    'OWAInstances',
    'TriplesFactory',
    'TriplesNumericLiteralsFactory',
]
