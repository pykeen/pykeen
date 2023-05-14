# -*- coding: utf-8 -*-

"""Classes for creating and storing training data from quadruples for temporal KGs."""

from .instances import LCWAQuadrupleInstances, QuadrupleInstances, SLCWAQuadrupleInstances
from .quadruples_factory import QuadruplesFactory

__all__ = [
    "QuadrupleInstances",
    "SLCWAQuadrupleInstances",
    "LCWAQuadrupleInstances",
    "QuadruplesFactory",
]
