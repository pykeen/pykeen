# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

from dataclasses import dataclass
import numpy as np

@dataclass
class Instances():
    """."""
    triples: np.array

@dataclass
class OWAInstances(Instances):
    """."""

@dataclass
class CWAInstances(Instances):
    """."""
    labels: np.array

class TriplesFactory():
    """."""

    def __int__(self, config):
        """."""
        self.config = config

    def create_instances(self) -> Instances:
        """"""