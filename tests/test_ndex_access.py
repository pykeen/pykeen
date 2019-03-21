# -*- coding: utf-8 -*-

"""Test NDEx access"""

import unittest

from pykeen.utilities.pipeline import load_data


class TestNDExAccess(unittest.TestCase):
    """Test whether NDEx can be accessed."""

    def test_load_graph(self):
        # Load Drugbank
        triples = load_data("ndex:eb1cf70a-1162-11e6-b550-06603eb7f303")

        self.assertIsNotNone(triples)

        # BioPlex 2.0
        triples = load_data("ndex:98ba6a19-586e-11e7-8f50-0ac135e8bacf")

        self.assertIsNotNone(triples)
