# -*- coding: utf-8 -*-

"""Test hyper-parameter optimization."""

import tempfile
import unittest

import optuna
import pytest

from pykeen.datasets.nations import NATIONS_TRAIN_PATH
from pykeen.hpo import hpo_pipeline
from pykeen.hpo.hpo import suggest_kwargs
from pykeen.triples import TriplesFactory


class TestInvalidConfigurations(unittest.TestCase):
    """Tests of invalid HPO configurations."""

    def test_earl_stopping_with_optimize_epochs(self):
        """Assert that the pipeline raises a value error."""
        with self.assertRaises(ValueError):
            hpo_pipeline(
                dataset='kinships',
                model='transe',
                stopper='early',
                training_kwargs_ranges=dict(epochs=...),
            )


@pytest.mark.slow
class TestHyperparameterOptimization(unittest.TestCase):
    """Test hyper-parameter optimization."""

    def test_run(self):
        """Test simply making a study."""
        hpo_pipeline_result = hpo_pipeline(
            dataset='nations',
            model='TransE',
            training_kwargs=dict(num_epochs=5, use_tqdm=False),
            n_trials=2,
        )
        df = hpo_pipeline_result.study.trials_dataframe(multi_index=True)
        # Check a model param is optimized
        self.assertIn(('params', 'model.embedding_dim'), df.columns)
        # Check a loss param is optimized
        self.assertIn(('params', 'loss.margin'), df.columns)
        self.assertNotIn(('params', 'training.num_epochs'), df.columns)

    def test_specified_model_hyperparameter(self):
        """Test making a study that has a specified model hyper-parameter."""
        target_embedding_dim = 50
        hpo_pipeline_result = hpo_pipeline(
            dataset='nations',
            model='TransE',
            model_kwargs=dict(embedding_dim=target_embedding_dim),
            training_kwargs=dict(num_epochs=5, use_tqdm=False),
            n_trials=2,
        )
        df = hpo_pipeline_result.study.trials_dataframe(multi_index=True)
        # Check a model param is NOT optimized
        self.assertNotIn(('params', 'model.embedding_dim'), df.columns)
        # Check a loss param is optimized
        self.assertIn(('params', 'loss.margin'), df.columns)

    def test_specified_loss_hyperparameter(self):
        """Test making a study that has a specified loss hyper-parameter."""
        hpo_pipeline_result = hpo_pipeline(
            dataset='nations',
            model='TransE',
            loss_kwargs=dict(margin=1.0),
            training_kwargs=dict(num_epochs=5, use_tqdm=False),
            n_trials=2,
        )
        df = hpo_pipeline_result.study.trials_dataframe(multi_index=True)
        # Check a model param is optimized
        self.assertIn(('params', 'model.embedding_dim'), df.columns)
        # Check a loss param is NOT optimized
        self.assertNotIn(('params', 'loss.margin'), df.columns)

    def test_specified_loss_and_model_hyperparameter(self):
        """Test making a study that has a specified loss hyper-parameter."""
        target_embedding_dim = 50
        hpo_pipeline_result = hpo_pipeline(
            dataset='nations',
            model='TransE',
            model_kwargs=dict(embedding_dim=target_embedding_dim),
            loss='MarginRankingLoss',
            loss_kwargs=dict(margin=1.0),
            training_kwargs=dict(num_epochs=5, use_tqdm=False),
            n_trials=2,
        )
        df = hpo_pipeline_result.study.trials_dataframe(multi_index=True)
        # Check a model param is NOT optimized
        self.assertNotIn(('params', 'model.embedding_dim'), df.columns)
        # Check a loss param is NOT optimized
        self.assertNotIn(('params', 'loss.margin'), df.columns)

    def test_specified_range(self):
        """Test making a study that has a specified hyper-parameter."""
        hpo_pipeline_result = hpo_pipeline(
            dataset='nations',
            model='TransE',
            model_kwargs_ranges=dict(
                embedding_dim=dict(type=int, low=60, high=80, q=10),
            ),
            loss_kwargs_ranges=dict(
                margin=dict(type=int, low=1, high=2),
            ),
            training_kwargs=dict(num_epochs=5, use_tqdm=False),
            n_trials=2,
        )
        df = hpo_pipeline_result.study.trials_dataframe(multi_index=True)
        self.assertIn(('params', 'model.embedding_dim'), df.columns)
        self.assertTrue(df[('params', 'model.embedding_dim')].isin({60., 70., 80.}).all())

        self.assertIn(('params', 'loss.margin'), df.columns)
        self.assertTrue(df[('params', 'loss.margin')].isin({1, 2}).all())

    def test_sampling_values_from_2_power_x(self):
        """Test making a study that has a range defined by f(x) = 2^x."""

        def objective(trial):
            suggest_kwargs(prefix='model', trial=trial, kwargs_ranges=model_kwargs_ranges)
            return 1.

        model_kwargs_ranges = dict(
            embedding_dim=dict(type=int, low=0, high=4, scale='power_two'),
        )

        study = optuna.create_study()
        study.optimize(objective, n_trials=2)

        df = study.trials_dataframe(multi_index=True)
        self.assertIn(('params', 'model.embedding_dim'), df.columns)
        self.assertTrue(df[('params', 'model.embedding_dim')].isin({1, 2, 4, 8, 16}).all())

        model_kwargs_ranges = dict(
            embedding_dim=dict(type=int, low=0, high=4, scale='power_two'),
        )

        with self.assertRaises(Exception) as context:
            study = optuna.create_study()
            study.optimize(objective, n_trials=2)
            self.assertIn('Upper bound 4 is not greater than lower bound 4.', context.exception)

    def test_custom_tf(self):
        """Test using a custom triples factories with HPO.

        .. seealso:: https://github.com/pykeen/pykeen/issues/230
        """
        tf = TriplesFactory.from_path(path=NATIONS_TRAIN_PATH)
        training, testing, validation = tf.split([.8, .1, .1], random_state=0)

        hpo_pipeline_result = hpo_pipeline(
            training=training,
            testing=testing,
            validation=validation,
            model='TransE',
            n_trials=2,
            training_kwargs=dict(num_epochs=2),
        )

        with tempfile.TemporaryDirectory() as directory:
            hpo_pipeline_result.save_to_directory(directory)
