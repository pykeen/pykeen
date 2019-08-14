# -*- coding: utf-8 -*-

"""Test that models are set in the right mode when they're training."""

import unittest

import torch

from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.models import TransE
from tests.constants import TEST_DATA


class TestBaseModel(unittest.TestCase):
    """Test models are set in the right mode at the right time."""

    def setUp(self) -> None:
        """Set up the test case with a triples factory and TransE as an example model."""
        self.batch_size = 16
        self.embedding_dim = 8
        self.factory = TriplesFactory(path=TEST_DATA)
        self.model = TransE(self.factory, embedding_dim=self.embedding_dim)

    def _check_scores(self, batch, scores) -> None:
        """Check the scores produced by a forward function."""
        # check for finite values by default
        assert torch.all(torch.isfinite(scores)).item()

    def test_predict_scores_all_subjects(self) -> None:
        """Test ``BaseModule.predict_scores_all_subjects``."""
        batch = torch.zeros(self.batch_size, 2, dtype=torch.long)

        # Set into training mode to check if it is correctly set to evaluation mode.
        self.model.train()

        scores = self.model.predict_scores_all_subjects(batch)
        assert scores.shape == (self.batch_size, self.model.num_entities)
        self._check_scores(batch, scores)

        assert not self.model.training

    def test_predict_scores_all_objects(self) -> None:
        """Test ``BaseModule.predict_scores_all_objects``."""
        batch = torch.zeros(self.batch_size, 2, dtype=torch.long)

        # Set into training mode to check if it is correctly set to evaluation mode.
        self.model.train()

        scores = self.model.predict_scores_all_objects(batch)
        assert scores.shape == (self.batch_size, self.model.num_entities)
        self._check_scores(batch, scores)

        assert not self.model.training

    def test_predict_scores(self) -> None:
        """Test ``BaseModule.predict_scores``."""
        batch = torch.zeros(self.batch_size, 3, dtype=torch.long)

        # Set into training mode to check if it is correctly set to evaluation mode.
        self.model.train()

        scores = self.model.predict_scores(batch)
        assert scores.shape == (self.batch_size, 1)
        self._check_scores(batch, scores)

        assert not self.model.training
