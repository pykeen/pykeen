# -*- coding: utf-8 -*-

"""Test that models can be executed."""
import unittest
from typing import ClassVar, Type

import torch

from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.models import BaseModule
from poem.models.unimodal import *
from tests.constants import TEST_DATA


class ModelTestCase(unittest.TestCase):
    """A test case for quickly defining common tests for KGE models."""

    model_cls: ClassVar[Type[BaseModule]]

    def setUp(self) -> None:
        self.batch_size = 16
        self.embedding_dim = 8
        self.factory = TriplesFactory(path=TEST_DATA)
        self.model = self.model_cls(self.factory, embedding_dim=self.embedding_dim)

    def check_scores(self, batch, scores):
        pass

    def test_forward_owa(self):
        batch = torch.zeros(self.batch_size, 3, dtype=torch.long)
        scores = self.model.forward_owa(batch)
        assert scores.shape == (self.batch_size, 1)
        self.check_scores(batch, scores)

    def test_forward_cwa(self):
        batch = torch.zeros(self.batch_size, 2, dtype=torch.long)
        try:
            scores = self.model.forward_cwa(batch)
        except NotImplementedError:
            self.fail(msg='Forward CWA not yet implemented')
        assert scores.shape == (self.batch_size, self.model.num_entities)
        self.check_scores(batch, scores)

    def test_forward_inverse_cwa(self):
        batch = torch.zeros(self.batch_size, 2, dtype=torch.long)
        try:
            scores = self.model.forward_inverse_cwa(batch)
        except NotImplementedError:
            self.fail(msg='Forward Inverse CWA not yet implemented')
        assert scores.shape == (self.batch_size, self.model.num_entities)
        self.check_scores(batch, scores)


class TestCaseComplex(ModelTestCase):
    model_cls = ComplEx


class TestCaseConvKB(ModelTestCase):
    model_cls = ConvKB


class TestCaseDistMult(ModelTestCase):
    model = DistMult


class TestCaseHolE(ModelTestCase):
    model_cls = HolE


class TestCaseRESCAL(ModelTestCase):
    model_cls = RESCAL


class TestCaseRotatE(ModelTestCase):
    model_cls = RotatE


class TestCaseSE(ModelTestCase):
    model_cls = StructuredEmbedding


class TestCaseTransD(ModelTestCase):
    model_cls = TransD

    def check_scores(self, batch, scores):
        super(TestCaseTransD, self).check_scores(batch=batch, scores=scores)

        # Distance-based model
        assert (scores <= 0.0).all()


class TestCaseTransE(ModelTestCase):
    model_cls = TransE


class TestCaseTransR(ModelTestCase):
    model_cls = TransR


class TestCaseUM(ModelTestCase):
    model_cls = UnstructuredModel

# class TestModels(unittest.TestCase):
#     """Test that models can be executed."""
#
#     path_to_training_data = os.path.join(RESOURCES_DIRECTORY, 'test.txt')
#     factory = TriplesFactory(path=path_to_training_data)
#
#     def _common_test(self, constructor):
#         model = constructor(triples_factory=self.factory)
#
#     def test_um(self):
#         """Tests that Unstructured Model can be executed."""
#         um = UnstructuredModel(triples_factory=self.factory)
#         self.assertIsNotNone(um)
#
#     def test_se(self):
#         """Tests that Structured Embedding can be executed."""
#         model = StructuredEmbedding(triples_factory=self.factory, embedding_dim=8)
#         self.assertIsNotNone(model)
#
#         # Dummy forward passes
#         # TODO: Use triple factory
#         batch_size = 16
#         triples = torch.zeros(batch_size, 3, dtype=torch.long)
#
#         # TODO: Refactor common tests for all models, e.g. shape checking
#         # Test forward_owa
#         scores = model.forward_owa(triples)
#         # Check shape
#         assert scores.shape == (batch_size, 1)
#
#         # Test forward_cwa
#         scores = model.forward_cwa(triples[:, :2])
#         # Check shape
#         assert scores.shape == (batch_size, model.num_entities)
#
#         # Test forward_inverse_cwa
#         scores = model.forward_inverse_cwa(triples[:, 1:])
#         # Check shape
#         assert scores.shape == (batch_size, model.num_entities)
#
#     def test_trans_e(self):
#         """Tests that TransE can be executed."""
#         trans_e = TransE(triples_factory=self.factory)
#         self.assertIsNotNone(trans_e)
#
#     def test_trans_h(self):
#         """Tests that TransH can be executed."""
#         trans_h = TransH(triples_factory=self.factory)
#         self.assertIsNotNone(trans_h)
#
#     def test_trans_r(self):
#         """Tests that TransR can be executed."""
#         trans_r = TransR(triples_factory=self.factory)
#         self.assertIsNotNone(trans_r)
#
#     def test_trans_d(self):
#         """Tests that TransD can be executed."""
#         trans_d = TransD(triples_factory=self.factory)
#         self.assertIsNotNone(trans_d)
#
#     def test_rescal(self):
#         """Tests that RESCAL can be executed."""
#         model = RESCAL(triples_factory=self.factory)
#         self.assertIsNotNone(model)
#
#         # Dummy forward passes
#         # TODO: Use triple factory
#         batch_size = 16
#         triples = torch.zeros(batch_size, 3, dtype=torch.long)
#
#         # TODO: Refactor common tests for all models, e.g. shape checking
#         # Test forward_owa
#         scores = model.forward_owa(triples)
#         # Check shape
#         assert scores.shape == (batch_size, 1)
#
#         # Test forward_cwa
#         scores = model.forward_cwa(triples[:, :2])
#         # Check shape
#         assert scores.shape == (batch_size, model.num_entities)
#
#         # Test forward_inverse_cwa
#         scores = model.forward_inverse_cwa(triples[:, 1:])
#         # Check shape
#         assert scores.shape == (batch_size, model.num_entities)
#
#     def test_distmult(self):
#         """Tests that DISTMULT can be executed."""
#         model = DistMult(triples_factory=self.factory)
#         self.assertIsNotNone(model)
#
#         # Dummy forward passes
#         # TODO: Use triple factory
#         batch_size = 16
#         triples = torch.zeros(batch_size, 3, dtype=torch.long)
#
#         # TODO: Refactor common tests for all models, e.g. shape checking
#         # Test forward_owa
#         scores = model.forward_owa(triples)
#         # Check shape
#         assert scores.shape == (batch_size, 1)
#
#         # Test forward_cwa
#         scores = model.forward_cwa(triples[:, :2])
#         # Check shape
#         assert scores.shape == (batch_size, model.num_entities)
#
#         # Test forward_inverse_cwa
#         scores = model.forward_inverse_cwa(triples[:, 1:])
#         # Check shape
#         assert scores.shape == (batch_size, model.num_entities)
#
#     def test_complex(self):
#         """Tests that COMPLEX can be executed."""
#         model = ComplEx(triples_factory=self.factory)
#         self.assertIsNotNone(model)
#
#         # Dummy forward passes
#         # TODO: Use triple factory
#         batch_size = 16
#         triples = torch.zeros(batch_size, 3, dtype=torch.long)
#
#         # TODO: Refactor common tests for all models, e.g. shape checking
#         # Test forward_owa
#         scores = model.forward_owa(triples)
#         # Check shape
#         assert scores.shape == (batch_size, 1)
#
#         # Test forward_cwa
#         scores = model.forward_cwa(triples[:, :2])
#         # Check shape
#         assert scores.shape == (batch_size, model.num_entities)
#
#         # Test forward_inverse_cwa
#         scores = model.forward_inverse_cwa(triples[:, 1:])
#         # Check shape
#         assert scores.shape == (batch_size, model.num_entities)
#
#     def test_rotate(self):
#         """Tests that Rotate can be executed."""
#         model = RotatE(triples_factory=self.factory)
#         self.assertIsNotNone(model)
#
#         # Dummy forward passes
#         # TODO: Use triple factory
#         batch_size = 16
#         triples = torch.zeros(batch_size, 3, dtype=torch.long)
#
#         # TODO: Refactor common tests for all models, e.g. shape checking
#         # Test forward_owa
#         scores = model.forward_owa(triples)
#         # Check shape
#         assert scores.shape == (batch_size, 1)
#         # Scores are negative distance -> non-positive
#         assert scores.max() <= 0.
#
#         # Test forward_cwa
#         scores = model.forward_cwa(triples[:, :2])
#         # Check shape
#         assert scores.shape == (batch_size, model.num_entities)
#         # Scores are negative distance -> non-positive
#         assert scores.max() <= 0.
#
#         # Test forward_inverse_cwa
#         scores = model.forward_inverse_cwa(triples[:, 1:])
#         # Check shape
#         assert scores.shape == (batch_size, model.num_entities)
#         # Scores are negative distance -> non-positive
#         assert scores.max() <= 0.
#
#     def test_hole(self):
#         """Tests that HolE can be executed."""
#         model = HolE(triples_factory=self.factory, embedding_dim=8)
#         self.assertIsNotNone(model)
#
#         # Dummy forward passes
#         # TODO: Use triple factory
#         batch_size = 16
#         triples = torch.zeros(batch_size, 3, dtype=torch.long)
#
#         # TODO: Refactor common tests for all models, e.g. shape checking
#         # Test forward_owa
#         scores = model.forward_owa(triples)
#         # Check shape
#         assert scores.shape == (batch_size, 1)
#
#         # Test forward_cwa
#         scores = model.forward_cwa(triples[:, :2])
#         # Check shape
#         assert scores.shape == (batch_size, model.num_entities)
#
#         # Test forward_inverse_cwa
#         scores = model.forward_inverse_cwa(triples[:, 1:])
#         # Check shape
#         assert scores.shape == (batch_size, model.num_entities)
#
#     # TODO
#     def test_conv_kb(self):
#         """Tests that ConvKB can be executed."""
#         model = ConvKB(triples_factory=self.factory, embedding_dim=4, num_filters=8)
#         self.assertIsNotNone(model)
#
#         # Dummy forward passes
#         # TODO: Use triple factory
#         batch_size = 16
#         triples = torch.zeros(batch_size, 3, dtype=torch.long)
#
#         # TODO: Refactor common tests for all models, e.g. shape checking
#         # Test forward_owa
#         scores = model.forward_owa(triples)
#         # Check shape
#         assert scores.shape == (batch_size, 1)
#
#         # Test forward_cwa
#         scores = model.forward_cwa(triples[:, :2])
#         # Check shape
#         assert scores.shape == (batch_size, model.num_entities)
#
#         # Test forward_inverse_cwa
#         scores = model.forward_inverse_cwa(triples[:, 1:])
#         # Check shape
#         assert scores.shape == (batch_size, model.num_entities)
