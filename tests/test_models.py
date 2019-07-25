# -*- coding: utf-8 -*-

"""Test that models can be executed."""
import os
import unittest

import torch

from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.models.unimodal import *
from tests.constants import RESOURCES_DIRECTORY


class TestModels(unittest.TestCase):
    """Test that models can be executed."""

    path_to_training_data = os.path.join(RESOURCES_DIRECTORY, 'test.txt')
    factory = TriplesFactory(path=path_to_training_data)

    def test_um(self):
        """Tests that Unstructured Model can be executed."""
        um = UnstructuredModel(triples_factory=self.factory)
        self.assertIsNotNone(um)

    def test_se(self):
        """Tests that Structured Embedding can be executed."""
        se = StructuredEmbedding(triples_factory=self.factory)
        self.assertIsNotNone(se)

    def test_trans_e(self):
        """Tests that TransE can be executed."""
        trans_e = TransE(triples_factory=self.factory)
        self.assertIsNotNone(trans_e)

    def test_trans_h(self):
        """Tests that TransH can be executed."""
        trans_h = TransH(triples_factory=self.factory)
        self.assertIsNotNone(trans_h)

    def test_trans_r(self):
        """Tests that TransR can be executed."""
        trans_r = TransR(triples_factory=self.factory)
        self.assertIsNotNone(trans_r)

    def test_trans_d(self):
        """Tests that TransD can be executed."""
        trans_d = TransD(triples_factory=self.factory)
        self.assertIsNotNone(trans_d)

    def test_rescal(self):
        """Tests that RESCAL can be executed."""
        rescale = RESCAL(triples_factory=self.factory)
        self.assertIsNotNone(rescale)

    def test_distmult(self):
        """Tests that DISTMULT can be executed."""
        distmult = DistMult(triples_factory=self.factory)
        self.assertIsNotNone(distmult)

    def test_complex(self):
        """Tests that COMPLEX can be executed."""
        complex = ComplEx(triples_factory=self.factory)
        self.assertIsNotNone(complex)

    def test_rotate(self):
        """Tests that Rotate can be executed."""
        model = RotatE(triples_factory=self.factory)
        self.assertIsNotNone(model)

        # Dummy forward passes
        # TODO: Use triple factory
        batch_size = 16
        triples = torch.zeros(batch_size, 3, dtype=torch.long)

        # TODO: Refactor common tests for all models, e.g. shape checking
        # Test forward_owa
        scores = model.forward_owa(triples)
        # Check shape
        assert scores.shape == (batch_size, 1)
        # Scores are negative distance -> non-positive
        assert scores.max() <= 0.

        # Test forward_cwa
        scores = model.forward_cwa(triples[:, :2])
        # Check shape
        assert scores.shape == (batch_size, model.num_entities)
        # Scores are negative distance -> non-positive
        assert scores.max() <= 0.

        # Test forward_inverse_cwa
        scores = model.forward_inverse_cwa(triples[:, 1:])
        # Check shape
        assert scores.shape == (batch_size, model.num_entities)
        # Scores are negative distance -> non-positive
        assert scores.max() <= 0.

    def test_hole(self):
        """Tests that HolE can be executed."""
        hole = HolE(triples_factory=self.factory)
        self.assertIsNotNone(hole)

    # TODO
    def test_conv_kb(self):
        """Tests that ConvKB can be executed."""
        pass
