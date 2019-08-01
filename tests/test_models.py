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
    model_cls = DistMult


class TestCaseHolE(ModelTestCase):
    model_cls = HolE


class TestCaseNTN(ModelTestCase):
    model_cls = NTN


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

    def check_scores(self, batch, scores):
        super(TestCaseTransE, self).check_scores(batch=batch, scores=scores)

        # Distance-based model
        assert (scores <= 0.0).all()


class TestCaseTransH(ModelTestCase):
    model_cls = TransH

    def check_scores(self, batch, scores):
        super(TestCaseTransH, self).check_scores(batch=batch, scores=scores)

        # Distance-based model
        assert (scores <= 0.0).all()


class TestCaseTransR(ModelTestCase):
    model_cls = TransR

    def check_scores(self, batch, scores):
        super(TestCaseTransR, self).check_scores(batch=batch, scores=scores)

        # Distance-based model
        assert (scores <= 0.0).all()


class TestCaseUM(ModelTestCase):
    model_cls = UnstructuredModel

    def check_scores(self, batch, scores):
        super(TestCaseUM, self).check_scores(batch=batch, scores=scores)

        # Distance-based model
        assert (scores <= 0.0).all()
