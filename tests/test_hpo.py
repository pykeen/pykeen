# -*- coding: utf-8 -*-

"""Test hyper-parameter optimization."""
import inspect
import tempfile
import unittest
from typing import Collection, Type
from unittest.mock import MagicMock, patch

import optuna
import pytest
from class_resolver import get_subclasses
from optuna.trial import TrialState
from torch.optim import Adam

from pykeen.datasets.nations import (
    NATIONS_TEST_PATH,
    NATIONS_TRAIN_PATH,
    NATIONS_VALIDATE_PATH,
    Nations,
    NationsLiteral,
)
from pykeen.evaluation import RankBasedEvaluator
from pykeen.hpo import hpo_pipeline
from pykeen.hpo.hpo import ExtraKeysError, Objective, suggest_kwargs
from pykeen.losses import Loss, MarginRankingLoss
from pykeen.models import (
    ERModel,
    FixedModel,
    InductiveERModel,
    LiteralModel,
    MarginalDistributionBaseline,
    Model,
    SoftInverseTripleBaseline,
)
from pykeen.regularizers import Regularizer
from pykeen.sampling import NegativeSampler
from pykeen.stoppers.stopper import NopStopper
from pykeen.trackers import ResultTracker, tracker_resolver
from pykeen.trackers.base import PythonResultTracker
from pykeen.training import LCWATrainingLoop, TrainingLoop
from pykeen.triples import TriplesFactory


class TestInvalidConfigurations(unittest.TestCase):
    """Tests of invalid HPO configurations."""

    def test_earl_stopping_with_optimize_epochs(self):
        """Assert that the pipeline raises a value error."""
        with self.assertRaises(ValueError):
            hpo_pipeline(
                dataset="kinships",
                model="transe",
                stopper="early",
                training_kwargs_ranges=dict(epochs=...),
            )


class TestHPOObjective(unittest.TestCase):
    """Test HPO objective."""

    def _test_re_raise(self, MockTrial, exception: Type[Exception]):  # noqa: N803
        """Test whether the given exception is raised when evaluating with the mocked trial."""
        objective = Objective(
            dataset=Nations,
            model=FixedModel,
            loss=MarginRankingLoss,
            optimizer=Adam,
            training_loop=LCWATrainingLoop,
            stopper=NopStopper,
            evaluator=RankBasedEvaluator,
            result_tracker=PythonResultTracker,
            metric="...",
        )
        with self.assertRaises(expected_exception=exception):
            objective(trial=MockTrial())

    @patch("pykeen.pipeline.pipeline", side_effect=MemoryError)
    @patch("optuna.Trial")
    def test_re_raise_memory_error(self, _mock_pipeline, MockTrial):  # noqa: N803
        """Check that memory errors are re-raised (to be catched by study.optimize)."""
        self._test_re_raise(MockTrial=MockTrial, exception=MemoryError)

    @patch("pykeen.pipeline.pipeline", side_effect=RuntimeError)
    @patch("optuna.Trial")
    def test_re_raise_runtime_error(self, _mock_pipeline, MockTrial):  # noqa: N803
        """Check that runtime errors are re-raised (to be catched by study.optimize)."""
        self._test_re_raise(MockTrial=MockTrial, exception=RuntimeError)


@pytest.mark.slow
class TestHyperparameterOptimization(unittest.TestCase):
    """Test hyper-parameter optimization."""

    def test_run(self):
        """Test simply making a study."""
        hpo_pipeline_result = hpo_pipeline(
            dataset="nations",
            model="TransE",
            training_kwargs=dict(num_epochs=5, use_tqdm=False),
            n_trials=2,
        )
        df = hpo_pipeline_result.study.trials_dataframe(multi_index=True)
        # Check a model param is optimized
        self.assertIn(("params", "model.embedding_dim"), df.columns)
        # Check a loss param is optimized
        self.assertIn(("params", "loss.margin"), df.columns)
        self.assertNotIn(("params", "training.num_epochs"), df.columns)

    def test_fail_invalid_kwarg_ranges(self):
        """Test that an exception is thrown if an incorrect argument is passed."""
        with self.assertRaises(ExtraKeysError) as e:
            hpo_pipeline(
                dataset="Nations",
                model="TransE",
                n_trials=1,
                training_loop="sLCWA",
                training_kwargs=dict(num_epochs=5, use_tqdm=False),
                negative_sampler_kwargs_ranges=dict(
                    garbage_key=dict(type=int, low=1, high=100),
                ),
            )
            self.assertEqual(["garbage_key"], e.exception.args[0])

    def test_specified_model_hyperparameter(self):
        """Test making a study that has a specified model hyper-parameter."""
        target_embedding_dim = 50
        hpo_pipeline_result = hpo_pipeline(
            dataset="nations",
            model="TransE",
            model_kwargs=dict(embedding_dim=target_embedding_dim),
            training_kwargs=dict(num_epochs=5, use_tqdm=False),
            n_trials=2,
        )
        df = hpo_pipeline_result.study.trials_dataframe(multi_index=True)
        # Check a model param is NOT optimized
        self.assertNotIn(("params", "model.embedding_dim"), df.columns)
        # Check a loss param is optimized
        self.assertIn(("params", "loss.margin"), df.columns)

    def test_specified_loss_hyperparameter(self):
        """Test making a study that has a specified loss hyper-parameter."""
        hpo_pipeline_result = hpo_pipeline(
            dataset="nations",
            model="TransE",
            loss_kwargs=dict(margin=1.0),
            training_kwargs=dict(num_epochs=5, use_tqdm=False),
            n_trials=2,
        )
        df = hpo_pipeline_result.study.trials_dataframe(multi_index=True)
        # Check a model param is optimized
        self.assertIn(("params", "model.embedding_dim"), df.columns)
        # Check a loss param is NOT optimized
        self.assertNotIn(("params", "loss.margin"), df.columns)

    def test_specified_loss_and_model_hyperparameter(self):
        """Test making a study that has a specified loss hyper-parameter."""
        target_embedding_dim = 50
        hpo_pipeline_result = hpo_pipeline(
            dataset="nations",
            model="TransE",
            model_kwargs=dict(embedding_dim=target_embedding_dim),
            loss="MarginRankingLoss",
            loss_kwargs=dict(margin=1.0),
            training_kwargs=dict(num_epochs=5, use_tqdm=False),
            n_trials=2,
        )
        df = hpo_pipeline_result.study.trials_dataframe(multi_index=True)
        # Check a model param is NOT optimized
        self.assertNotIn(("params", "model.embedding_dim"), df.columns)
        # Check a loss param is NOT optimized
        self.assertNotIn(("params", "loss.margin"), df.columns)

    def test_specified_range(self):
        """Test making a study that has a specified hyper-parameter."""
        hpo_pipeline_result = hpo_pipeline(
            dataset="nations",
            model="TransE",
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
        self.assertIn(("params", "model.embedding_dim"), df.columns)
        self.assertTrue(df[("params", "model.embedding_dim")].isin({60.0, 70.0, 80.0}).all())

        self.assertIn(("params", "loss.margin"), df.columns)
        self.assertTrue(df[("params", "loss.margin")].isin({1, 2}).all())

    def test_sampling_values_from_2_power_x(self):
        """Test making a study that has a range defined by f(x) = 2^x."""
        model_kwargs_ranges = dict(
            embedding_dim=dict(type=int, low=0, high=4, scale="power_two"),
        )
        objective = _test_suggest(model_kwargs_ranges)
        study = optuna.create_study()
        study.optimize(objective, n_trials=2)

        df = study.trials_dataframe(multi_index=True)
        self.assertIn(("params", "model.embedding_dim"), df.columns)
        self.assertTrue(df[("params", "model.embedding_dim")].isin({1, 2, 4, 8, 16}).all())

        objective = _test_suggest(model_kwargs_ranges)
        with self.assertRaises(Exception) as context:
            study = optuna.create_study()
            study.optimize(objective, n_trials=2)
            self.assertIn("Upper bound 4 is not greater than lower bound 4.", context.exception)

    def test_sampling_values_from_power_x(self):
        """Test making a study that has a range defined by f(x) = base^x."""
        kwargs_ranges = dict(
            embedding_dim=dict(type=int, low=0, high=2, scale="power", base=10),
        )
        objective = _test_suggest(kwargs_ranges)
        study = optuna.create_study()
        study.optimize(objective, n_trials=2)

        df = study.trials_dataframe(multi_index=True)
        self.assertIn(("params", "model.embedding_dim"), df.columns)
        values = df[("params", "model.embedding_dim")]
        self.assertTrue(values.isin({1, 10, 100}).all(), msg=f"Got values: {values}")

    def test_failing_trials(self):
        """Test whether failing trials are correctly reported."""

        class MockResultTracker(MagicMock, ResultTracker):
            """A mock result tracker."""

        tracker_resolver.register(MockResultTracker)

        mock_result_tracker = MockResultTracker()
        mock_result_tracker.end_run = MagicMock()
        result = hpo_pipeline(
            dataset="nations",
            model="distmult",
            model_kwargs_ranges=dict(
                embedding_dim=dict(
                    type=int,
                    low=-10,
                    high=-1,  # will fail
                ),
            ),
            n_trials=1,
            result_tracker=mock_result_tracker,
        )
        # verify failure
        assert all(t.state == TrialState.FAIL for t in result.study.trials)
        assert all(ca[1]["success"] is False for ca in mock_result_tracker.end_run.call_args_list)


def _test_suggest(kwargs_ranges):
    def objective(trial):
        suggest_kwargs(prefix="model", trial=trial, kwargs_ranges=kwargs_ranges)
        return 1.0

    return objective


@pytest.mark.slow
class TestHPODatasets(unittest.TestCase):
    """Test different ways of loading data in HPO."""

    def test_custom_dataset_instance(self):
        """Test passing a pre-instantiated dataset to HPO."""
        hpo_pipeline_result = self._help_test_hpo(
            study_name="HPO with custom dataset instance",
            dataset=Nations(),  # mock a "custom" dataset by using one already available
        )
        # Since custom data was passed, we can't store any of this
        self.assertNotIn("dataset", hpo_pipeline_result.study.user_attrs)
        self.assertNotIn("training", hpo_pipeline_result.study.user_attrs)
        self.assertNotIn("testing", hpo_pipeline_result.study.user_attrs)
        self.assertNotIn("validation", hpo_pipeline_result.study.user_attrs)

    def test_custom_dataset_cls(self):
        """Test passing a dataset class to HPO."""
        hpo_pipeline_result = self._help_test_hpo(
            study_name="HPO with custom dataset class",
            dataset=Nations,
        )
        # currently, any custom data doesn't get stored.
        self.assertNotIn("dataset", hpo_pipeline_result.study.user_attrs)
        # self.assertEqual(Nations.get_normalized_name(), hpo_pipeline_result.study.user_attrs['dataset'])
        self.assertNotIn("training", hpo_pipeline_result.study.user_attrs)
        self.assertNotIn("testing", hpo_pipeline_result.study.user_attrs)
        self.assertNotIn("validation", hpo_pipeline_result.study.user_attrs)

    def test_custom_dataset_path(self):
        """Test passing a dataset class to HPO."""
        hpo_pipeline_result = self._help_test_hpo(
            study_name="HPO with custom dataset path",
            dataset=NATIONS_TRAIN_PATH,
        )
        self.assertIn("dataset", hpo_pipeline_result.study.user_attrs)
        self.assertEqual(str(NATIONS_TRAIN_PATH), hpo_pipeline_result.study.user_attrs["dataset"])
        self.assertNotIn("training", hpo_pipeline_result.study.user_attrs)
        self.assertNotIn("testing", hpo_pipeline_result.study.user_attrs)
        self.assertNotIn("validation", hpo_pipeline_result.study.user_attrs)

    def test_custom_tf_object(self):
        """Test using a custom triples factories with HPO.

        .. seealso:: https://github.com/pykeen/pykeen/issues/230
        """
        tf = TriplesFactory.from_path(path=NATIONS_TRAIN_PATH)
        training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=0)

        hpo_pipeline_result = self._help_test_hpo(
            study_name="HPO with custom triples factories",
            training=training,
            testing=testing,
            validation=validation,
        )
        self.assertNotIn("dataset", hpo_pipeline_result.study.user_attrs)
        # Since there's no source path information, these shouldn't be
        # added, even if it might be possible to infer path information
        # from the triples factories
        self.assertNotIn("training", hpo_pipeline_result.study.user_attrs)
        self.assertNotIn("testing", hpo_pipeline_result.study.user_attrs)
        self.assertNotIn("validation", hpo_pipeline_result.study.user_attrs)

    def test_custom_paths(self):
        """Test using a custom triples paths with HPO."""
        hpo_pipeline_result = self._help_test_hpo(
            study_name="HPO with custom triples paths",
            training=NATIONS_TRAIN_PATH,  # mock "custom" paths
            testing=NATIONS_TEST_PATH,
            validation=NATIONS_VALIDATE_PATH,
        )
        self.assertNotIn("dataset", hpo_pipeline_result.study.user_attrs)
        # Since paths were passed for training, testing, and validation,
        # they should be stored as study-level attributes
        self.assertIn("training", hpo_pipeline_result.study.user_attrs)
        self.assertEqual(str(NATIONS_TRAIN_PATH), hpo_pipeline_result.study.user_attrs["training"])
        self.assertIn("testing", hpo_pipeline_result.study.user_attrs)
        self.assertEqual(str(NATIONS_TEST_PATH), hpo_pipeline_result.study.user_attrs["testing"])
        self.assertIn("validation", hpo_pipeline_result.study.user_attrs)
        self.assertEqual(str(NATIONS_VALIDATE_PATH), hpo_pipeline_result.study.user_attrs["validation"])

    def _help_test_hpo(self, **kwargs):
        hpo_pipeline_result = hpo_pipeline(
            **kwargs,
            model="TransE",
            n_trials=1,
            training_kwargs=dict(num_epochs=1, use_tqdm=False),
            evaluation_kwargs=dict(use_tqdm=False),
        )
        with tempfile.TemporaryDirectory() as directory:
            hpo_pipeline_result.save_to_directory(directory)
        return hpo_pipeline_result


@pytest.mark.slow
class TestHyperparameterOptimizationLiterals(unittest.TestCase):
    """Test hyper-parameter optimization."""

    def test_run(self):
        """Test simply making a study."""
        hpo_pipeline_result = hpo_pipeline(
            dataset=NationsLiteral,
            model="DistMultLiteral",
            training_kwargs=dict(num_epochs=5, use_tqdm=False),
            n_trials=2,
        )
        df = hpo_pipeline_result.study.trials_dataframe(multi_index=True)
        # Check a model param is optimized
        self.assertIn(("params", "model.embedding_dim"), df.columns)
        # Check a loss param is optimized
        self.assertIn(("params", "loss.margin"), df.columns)
        self.assertNotIn(("params", "training.num_epochs"), df.columns)


@pytest.mark.parametrize(
    "base_cls,ignore",
    [
        (Loss, []),
        (Regularizer, []),
        (Model, [InductiveERModel, LiteralModel, ERModel, SoftInverseTripleBaseline, MarginalDistributionBaseline]),
        (NegativeSampler, []),
        (TrainingLoop, []),
    ],
)
def test_hpo_defaults(base_cls: Type, ignore: Collection[Type]):
    """Test HPO defaults for components that are used in the HPO pipeline."""
    assert set(ignore) == {
        cls
        for cls in get_subclasses(base_cls)
        if not (inspect.isabstract(cls) or isinstance(getattr(cls, "hpo_default", None), dict))
    }
