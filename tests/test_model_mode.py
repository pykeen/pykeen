# -*- coding: utf-8 -*-

"""Test that models are set in the right mode when they're training."""

import unittest
from dataclasses import dataclass

import torch

from pykeen.datasets import Nations
from pykeen.models import FixedModel, Model, TransE
from pykeen.triples import TriplesFactory
from pykeen.utils import resolve_device


class TestBaseModel(unittest.TestCase):
    """Test models are set in the right mode at the right time."""

    batch_size: int
    embedding_dim: int
    factory: TriplesFactory
    model: Model

    def setUp(self) -> None:
        """Set up the test case with a triples factory and TransE as an example model."""
        self.batch_size = 16
        self.embedding_dim = 8
        self.factory = Nations().training
        self.device = resolve_device("cpu")
        self.model = TransE(triples_factory=self.factory, embedding_dim=self.embedding_dim).to(self.device)

    def _check_scores(self, scores) -> None:
        """Check the scores produced by a forward function."""
        # check for finite values by default
        assert torch.all(torch.isfinite(scores)).item()

    def test_predict_scores_all_heads(self) -> None:
        """Test ``BaseModule.predict_scores_all_heads``."""
        batch = torch.zeros(self.batch_size, 2, dtype=torch.long, device=self.model.device)

        # Set into training mode to check if it is correctly set to evaluation mode.
        self.model.train()

        scores = self.model.predict_h(batch)
        assert scores.shape == (self.batch_size, self.model.num_entities)
        self._check_scores(scores)

        assert not self.model.training

    def test_predict_scores_all_tails(self) -> None:
        """Test ``BaseModule.predict_scores_all_tails``."""
        batch = torch.zeros(self.batch_size, 2, dtype=torch.long, device=self.model.device)

        # Set into training mode to check if it is correctly set to evaluation mode.
        self.model.train()

        scores = self.model.predict_t(batch)
        assert scores.shape == (self.batch_size, self.model.num_entities)
        self._check_scores(scores)

        assert not self.model.training

    def test_predict_scores(self) -> None:
        """Test ``BaseModule.predict_scores``."""
        batch = torch.zeros(self.batch_size, 3, dtype=torch.long, device=self.model.device)

        # Set into training mode to check if it is correctly set to evaluation mode.
        self.model.train()

        scores = self.model.predict_hrt(batch)
        assert scores.shape == (self.batch_size, 1)
        self._check_scores(scores)

        assert not self.model.training


class TestBaseModelScoringFunctions(unittest.TestCase):
    """Tests for testing the correctness of the base model fall back scoring functions."""

    def setUp(self):
        """Prepare for testing the scoring functions."""
        self.generator = torch.random.manual_seed(seed=42)
        self.triples_factory = MinimalTriplesFactory
        self.device = resolve_device()
        self.model = FixedModel(triples_factory=self.triples_factory).to(self.device)

    def test_alignment_of_score_t_fall_back(self) -> None:
        """Test if ``BaseModule.score_t`` aligns with ``BaseModule.score_hrt``."""
        hr_batch = torch.tensor(
            [
                [0, 0],
                [1, 0],
            ],
            dtype=torch.long,
            device=self.device,
        )
        hrt_batch = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [1, 0, 0],
                [1, 0, 1],
            ],
            dtype=torch.long,
            device=self.device,
        )
        scores_t_function = self.model.score_t(hr_batch=hr_batch).flatten()
        scores_hrt_function = self.model.score_hrt(hrt_batch=hrt_batch)
        assert all(scores_t_function == scores_hrt_function)

    def test_alignment_of_score_h_fall_back(self) -> None:
        """Test if ``BaseModule.score_h`` aligns with ``BaseModule.score_hrt``."""
        rt_batch = torch.tensor(
            [
                [0, 0],
                [1, 0],
            ],
            dtype=torch.long,
            device=self.device,
        )
        hrt_batch = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=torch.long,
            device=self.device,
        )
        scores_h_function = self.model.score_h(rt_batch=rt_batch).flatten()
        scores_hrt_function = self.model.score_hrt(hrt_batch=hrt_batch)
        assert all(scores_h_function == scores_hrt_function)

    def test_alignment_of_score_r_fall_back(self) -> None:
        """Test if ``BaseModule.score_r`` aligns with ``BaseModule.score_hrt``."""
        ht_batch = torch.tensor(
            [
                [0, 0],
                [1, 0],
            ],
            dtype=torch.long,
            device=self.device,
        )
        hrt_batch = torch.tensor(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
            ],
            dtype=torch.long,
            device=self.device,
        )
        scores_r_function = self.model.score_r(ht_batch=ht_batch).flatten()
        scores_hrt_function = self.model.score_hrt(hrt_batch=hrt_batch)
        assert all(scores_r_function == scores_hrt_function)


@dataclass
class MinimalTriplesFactory:
    """A triples factory with minial attributes to allow the model to initiate."""

    num_entities = 2
    num_relations = 2
    create_inverse_triples: bool = False
