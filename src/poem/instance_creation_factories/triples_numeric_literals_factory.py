# -*- coding: utf-8 -*-

"""Implementation of factory that create instances containing of triples and numeric literals."""

from poem.instance_creation_factories.triples_factory import TriplesFactory, Instances


class TriplesNumericLiteralsFactory(TriplesFactory):
    """."""

    def __int__(self, config):
        """."""
        self.config = config

    def create_instances(self) -> Instances:
        """"""
        self.create_instances()
