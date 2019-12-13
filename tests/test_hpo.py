# -*- coding: utf-8 -*-

"""Test hyperparameter optimization."""

import unittest

from poem.hpo import make_study


class TestHyperparameterOptimization(unittest.TestCase):
    """Test hyperparameter optimization."""

    def test_run(self):
        """Test simply making a study."""
        study = make_study(
            model='TransE',
            dataset='nations',
            n_trials=2,
        )
        df = study.trials_dataframe()
        # Check a model param is optimized
        self.assertIn(('params', 'embedding_dim'), df.columns)
        # Check a loss param is optimized
        self.assertIn(('params', 'margin'), df.columns)

    def test_specified_model_hyperparameter(self):
        """Test making a study that has a specified model hyperparameter."""
        target_embedding_dim = 50
        study = make_study(
            model='TransE',
            model_kwargs=dict(embedding_dim=target_embedding_dim),
            dataset='nations',
            n_trials=2,
        )
        df = study.trials_dataframe()
        # Check a model param is NOT optimized
        self.assertNotIn(('params', 'embedding_dim'), df.columns)
        # Check a loss param is optimized
        self.assertIn(('params', 'margin'), df.columns)

    def test_specified_loss_hyperparameter(self):
        """Test making a study that has a specified loss hyperparameter."""
        study = make_study(
            model='TransE',
            dataset='nations',
            loss_kwargs=dict(margin=1.0),
            n_trials=2,
        )
        df = study.trials_dataframe()
        # Check a model param is optimized
        self.assertIn(('params', 'embedding_dim'), df.columns)
        # Check a loss param is NOT optimized
        self.assertNotIn(('params', 'margin'), df.columns)

    def test_specified_loss_and_model_hyperparameter(self):
        """Test making a study that has a specified loss hyperparameter."""
        target_embedding_dim = 50
        study = make_study(
            model='TransE',
            model_kwargs=dict(embedding_dim=target_embedding_dim),
            dataset='nations',
            loss='MarginRankingLoss',
            loss_kwargs=dict(margin=1.0),
            n_trials=2,
        )
        df = study.trials_dataframe()
        # Check a model param is NOT optimized
        self.assertNotIn(('params', 'embedding_dim'), df.columns)
        # Check a loss param is NOT optimized
        self.assertNotIn(('params', 'margin'), df.columns)

    def test_specified_range(self):
        """Test making a study that has a specified hyperparameter."""
        study = make_study(
            model='TransE',
            model_kwargs_ranges=dict(
                embedding_dim=dict(type=int, low=60, high=80, q=10),
            ),
            loss_kwargs_ranges=dict(
                margin=dict(type=int, low=1, high=2),
            ),
            dataset='nations',
            n_trials=2,
        )
        df = study.trials_dataframe()
        self.assertIn(('params', 'embedding_dim'), df.columns)
        self.assertTrue(df[('params', 'embedding_dim')].isin({60., 70., 80.}).all())

        self.assertIn(('params', 'margin'), df.columns)
        self.assertTrue(df[('params', 'margin')].isin({1, 2}).all())
