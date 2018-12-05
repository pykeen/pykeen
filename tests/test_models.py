# -*- coding: utf-8 -*-

"""Test that all models can be instantiated."""

import unittest

from pykeen.kge_models import TransE


class TestModelInstantiation(unittest.TestCase):
    """Test that all models can be instantiated."""

    def test_instantiate_trans_e(self):
        """Test that TransE can be instantiated."""
        trans_e = TransE(config={
        })
        self.assertIsNotNone(trans_e)
        # TODO
