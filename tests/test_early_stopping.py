# -*- coding: utf-8 -*-

"""Tests of early stopping."""

import unittest
from typing import List
from unittest.mock import Mock

import numpy
import pytest
import torch
import unittest_templates
from torch.optim import Adam

from pykeen.datasets import Nations
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import FixedModel, Model, TransE
from pykeen.stoppers.early_stopping import EarlyStopper, EarlyStoppingLogic, is_improvement
from pykeen.training import SLCWATrainingLoop
from pykeen.typing import RANK_REALISTIC, SIDE_BOTH
from tests.mocks import MockEvaluator


class TestRandom(unittest.TestCase):
    """Random tests for early stopper."""

    def test_is_improvement(self):
        """Test is_improvement()."""
        for best_value, current_value, larger_is_better, relative_delta, is_better in [
            # equal value; larger is better
            (1.0, 1.0, True, 0.0, False),
            # equal value; smaller is better
            (1.0, 1.0, False, 0.0, False),
            # larger is better; improvement
            (1.0, 1.1, True, 0.0, True),
            # larger is better; improvement; but not significant
            (1.0, 1.1, True, 0.1, False),
        ]:
            with self.subTest(
                best_value=best_value,
                current_value=current_value,
                larger_is_better=larger_is_better,
                relative_delta=relative_delta,
                is_better=is_better,
            ):
                self.assertEqual(
                    is_better,
                    is_improvement(
                        best_value=best_value,
                        current_value=current_value,
                        larger_is_better=larger_is_better,
                        relative_delta=relative_delta,
                    ),
                )


class LogCallWrapper:
    """An object which wraps functions and checks whether they have been called."""

    def __init__(self):
        self.called = set()

    def wrap(self, func):
        """Wrap the function."""
        id_func = id(func)

        def wrapped(*args, **kwargs):
            self.called.add(id_func)
            return func(*args, **kwargs)

        return wrapped

    def was_called(self, func) -> bool:
        """Report whether the previously wrapped function has been called."""
        return id(func) in self.called


class TestEarlyStopper(unittest.TestCase):
    """Tests for early stopping."""

    #: The window size used by the early stopper
    patience: int = 2
    #: The mock losses the mock evaluator will return
    mock_losses: List[float] = [10.0, 9.0, 8.0, 9.0, 8.0, 8.0]
    #: The (zeroed) index  - 1 at which stopping will occur
    stop_constant: int = 4
    #: The minimum improvement
    delta: float = 0.0
    #: The best results
    best_results: List[float] = [10.0, 9.0, 8.0, 8.0, 8.0]

    def setUp(self):
        """Prepare for testing the early stopper."""
        # Set automatic_memory_optimization to false for tests
        self.mock_evaluator = MockEvaluator(
            key=("hits_at_10", SIDE_BOTH, RANK_REALISTIC),
            values=self.mock_losses,
            automatic_memory_optimization=False,
        )
        nations = Nations()
        self.model = FixedModel(triples_factory=nations.training)
        self.stopper = EarlyStopper(
            model=self.model,
            evaluator=self.mock_evaluator,
            training_triples_factory=nations.training,
            evaluation_triples_factory=nations.validation,
            patience=self.patience,
            relative_delta=self.delta,
            larger_is_better=False,
        )

    def test_initialization(self):
        """Test warm-up phase."""
        for epoch in range(self.patience):
            should_stop = self.stopper.should_stop(epoch=epoch)
            assert not should_stop

    def test_result_processing(self):
        """Test that the mock evaluation of the early stopper always gives the right loss."""
        for epoch in range(len(self.mock_losses)):
            # Step early stopper
            should_stop = self.stopper.should_stop(epoch=epoch)

            if not should_stop:
                # check storing of results
                assert self.stopper.results == self.mock_losses[: epoch + 1]
                assert self.stopper.best_metric == self.best_results[epoch]

    def test_should_stop(self):
        """Test that the stopper knows when to stop."""
        for epoch in range(self.stop_constant):
            self.assertFalse(self.stopper.should_stop(epoch=epoch))
        self.assertTrue(self.stopper.should_stop(epoch=self.stop_constant))

    def test_result_logging(self):
        """Test whether result logger is called properly."""
        self.stopper.result_tracker = mock_tracker = Mock()
        self.stopper.should_stop(epoch=0)
        log_metrics = mock_tracker.log_metrics
        self.assertIsInstance(log_metrics, Mock)
        log_metrics.assert_called_once()
        _, call_args = log_metrics.call_args_list[0]
        self.assertIn("step", call_args)
        self.assertEqual(0, call_args["step"])
        self.assertIn("prefix", call_args)
        self.assertEqual("validation", call_args["prefix"])

    def test_serialization(self):
        """Test for serialization."""
        summary = self.stopper.get_summary_dict()
        new_stopper = EarlyStopper(
            # not needed for test
            model=...,
            evaluator=...,
            training_triples_factory=...,
            evaluation_triples_factory=...,
        )
        new_stopper._write_from_summary_dict(**summary)
        for key in summary.keys():
            assert getattr(self.stopper, key) == getattr(new_stopper, key)


class TestEarlyStoppingLogic(unittest_templates.GenericTestCase[EarlyStoppingLogic]):
    """Tests for early stopping logic."""

    cls = EarlyStoppingLogic
    kwargs = dict(
        patience=2,
        relative_delta=0.1,
        larger_is_better=False,
    )

    def test_report_result(self):
        """Test report_result API."""
        metric = 1.0e-03
        epoch = 3
        stop = self.instance.report_result(metric=metric, epoch=epoch)
        assert isinstance(stop, bool)

        # assert that reporting another metric for this epoch raises an error
        with self.assertRaises(ValueError):
            self.instance.report_result(metric=..., epoch=epoch)

    def test_early_stopping(self):
        """Test early stopping."""
        for epoch, value in enumerate([10.0, 9.0, 8.0, 7.99, 7.98, 7.97]):
            stop = self.instance.report_result(metric=value, epoch=epoch)
            self.assertEqual(stop, epoch >= 4)


class TestEarlyStopperDelta(TestEarlyStopper):
    """Test early stopping with a tiny delta."""

    mock_losses: List[float] = [10.0, 9.0, 8.0, 7.99, 7.98, 7.97]
    stop_constant: int = 4
    delta: float = 0.1
    best_results: List[float] = [10.0, 10.0, 8.0, 8.0, 8.0]


class TestEarlyStopperRealWorld(unittest.TestCase):
    """Test early stopping on a real-world use case of training TransE with Adam."""

    #: The window size used by the early stopper
    patience: int = 2
    #: The (zeroed) index  - 1 at which stopping will occur
    stop_constant: int = 4
    #: The minimum improvement
    relative_delta: float = 0.1
    #: The random seed to use for reproducibility
    seed: int = 42
    #: The maximum number of epochs to train. Should be large enough to allow for early stopping.
    max_num_epochs: int = 1000
    #: The epoch at which the stop should happen. Depends on the choice of random seed.
    stop_epoch: int = 30
    #: The batch size to use.
    batch_size: int = 128

    def setUp(self) -> None:
        """Set up the real world early stopping test."""
        # Fix seed for reproducibility
        torch.manual_seed(seed=self.seed)
        numpy.random.seed(seed=self.seed)

    @pytest.mark.slow
    def test_early_stopping(self):
        """Tests early stopping."""
        # Set automatic_memory_optimization to false during testing
        nations = Nations()
        model: Model = TransE(triples_factory=nations.training)
        evaluator = RankBasedEvaluator(automatic_memory_optimization=False)
        stopper = EarlyStopper(
            model=model,
            evaluator=evaluator,
            training_triples_factory=nations.training,
            evaluation_triples_factory=nations.validation,
            patience=self.patience,
            relative_delta=self.relative_delta,
            metric="mean_rank",
        )
        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=nations.training,
            optimizer=Adam(params=model.get_grad_params()),
            automatic_memory_optimization=False,
        )
        losses = training_loop.train(
            triples_factory=nations.training,
            num_epochs=self.max_num_epochs,
            batch_size=self.batch_size,
            stopper=stopper,
            use_tqdm=False,
        )
        self.assertEqual(stopper.number_results, len(losses) // stopper.frequency)
        self.assertEqual(stopper.best_epoch, self.stop_epoch - self.patience * stopper.frequency)
        self.assertEqual(self.stop_epoch, len(losses), msg="Did not stop early like it should have")
