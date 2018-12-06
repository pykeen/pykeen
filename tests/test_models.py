# -*- coding: utf-8 -*-

"""Test that all models can be instantiated."""

import unittest

from pykeen.constants import *
from pykeen.kge_models import TransE


class TestModelInstantiation(unittest.TestCase):
    """Test that all models can be instantiated."""

    def test_instantiate_trans_e(self):
        """Test that TransE can be instantiated."""
        trans_e = TransE(config={
            NUM_ENTITIES: 5,
            NUM_RELATIONS: 5,
            EMBEDDING_DIM: 5,
            NORM_FOR_NORMALIZATION_OF_ENTITIES: 2,
            SCORING_FUNCTION_NORM: 1,
            MARGIN_LOSS: 4,
        })
        # TODO @mehdi type checking for parameters
        # TODO @mehdi @cthoyt refactor reused code for TransXXX models to base class
        self.assertIsNotNone(trans_e)
