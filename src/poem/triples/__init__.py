# -*- coding: utf-8 -*-

"""Classes for creating and storing training data from triples."""

from .instances import (
    CWAInstances, Instances, MultimodalCWAInstances, MultimodalInstances, MultimodalOWAInstances, OWAInstances,
)
from .triples_factory import TriplesFactory
from .triples_numeric_literals_factory import TriplesNumericLiteralsFactory

__all__ = [
    'Instances',
    'OWAInstances',
    'CWAInstances',
    'MultimodalInstances',
    'MultimodalOWAInstances',
    'MultimodalCWAInstances',
    'TriplesFactory',
    'TriplesNumericLiteralsFactory',
]
