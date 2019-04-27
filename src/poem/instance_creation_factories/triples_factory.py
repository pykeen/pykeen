# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

from dataclasses import dataclass

@dataclass
class Instances():
    """."""

@dataclass
class OWAInstances(Instances):
    """."""

@dataclass
class CWAInstances(Instances):
    """."""

class TriplesFactory():
    """."""

    def __int__(self, config):
        """."""
        self.config = config

    def create_instances(self) -> Instances:
        """"""