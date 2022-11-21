# -*- coding: utf-8 -*-

"""Classes for creating and storing training data from triples."""

from .instances import Instances, LCWAInstances, SLCWAInstances
from .statement_factory import StatementFactory
from .triples_factory import (
    AnyTriples,
    CoreTriplesFactory,
    KGInfo,
    RelationInverter,
    TriplesFactory,
    get_mapped_triples,
    relation_inverter,
)
from .triples_numeric_literals_factory import TriplesNumericLiteralsFactory

__all__ = [
    "Instances",
    "LCWAInstances",
    "SLCWAInstances",
    "KGInfo",
    "CoreTriplesFactory",
    "RelationInverter",
    "relation_inverter",
    "TriplesFactory",
    "TriplesNumericLiteralsFactory",
    "get_mapped_triples",
    "AnyTriples",
    "StatementFactory",
]
