# -*- coding: utf-8 -*-

"""Tests of early stopping."""

import unittest
from typing import List

import numpy
import pytest
import torch
import unittest_templates
from torch.optim import Adam

from pykeen.datasets import Nations
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import Model, TransE
from pykeen.stoppers.early_stopping import EarlyStopper, EarlyStoppingLogic, is_improvement
from pykeen.training import SLCWATrainingLoop
from tests import cases


@pytest.mark.parametrize(
    "best,current,larger_is_better,relative_delta,is_better",
    [
        # equal value; larger is better
        (1.0, 1.0, True, 0.0, False),
        # equal value; smaller is better
        (1.0, 1.0, False, 0.0, False),
        # larger is better; improvement
        (1.0, 1.1, True, 0.0, True),
        # larger is better; improvement; but not significant
        (1.0, 1.1, True, 0.1, False),
        # negative number
        (-1, -1, True, 0.1, False),
    ],
)
def test_is_improvement(best: float, current: float, larger_is_better: bool, relative_delta: float, is_better: bool):
    """Test is_improvement()."""
    assert (
        is_improvement(
            best_value=best, current_value=current, larger_is_better=larger_is_better, relative_delta=relative_delta
        )
        is is_better
    )


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


class TestEarlyStopper(cases.EarlyStopperTestCase):
    """Tests for early stopping."""

    patience: int = 2
    mock_losses: List[float] = [10.0, 9.0, 8.0, 9.0, 8.0, 8.0]
    stop_constant: int = 4
    delta: float = 0.0
    best_results: List[float] = [10.0, 9.0, 8.0, 8.0, 8.0]


class TestEarlyStopperDelta(cases.EarlyStopperTestCase):
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
