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

    # TODO
    def test_rotate(self):
        """Tests that Rotate can be executed."""
        pass

    def test_hole(self):
        """Tests that HolE can be executed."""
        hole = HolE(triples_factory=self.factory)
        self.assertIsNotNone(hole)

    # TODO
    def test_conv_kb(self):
        """Tests that ConvKB can be executed."""
        pass

    def test_ntn(self):
        """Tests that NTN can be executed."""
        model = NTN(triples_factory=self.factory, embedding_dim=16)
        self.assertIsNotNone(model)

        batch_size = 8

        batch = torch.zeros(batch_size, 3, dtype=torch.long)
        scores = model.forward_owa(batch)
        assert scores.shape == (batch_size, 1)

        scores = model.forward_cwa(batch)
        assert scores.shape == (batch_size, model.num_entities)

        scores = model.forward_inverse_cwa(batch)
        assert scores.shape == (batch_size, model.num_entities)
