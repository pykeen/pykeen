# -*- coding: utf-8 -*-

"""Test that regularizers can be executed."""

import logging
import unittest
from typing import Any, ClassVar, Dict, Optional, Type

import torch

from poem.datasets import NationsTrainingTriplesFactory
from poem.models import RESCAL
from poem.regularizers import CombinedRegularizer, LpRegularizer, NoRegularizer, PowerSumRegularizer, Regularizer
from poem.triples import TriplesFactory
from poem.typing import MappedTriples
from poem.utils import resolve_device


class _RegularizerTestCase:
    """A test case for quickly defining common tests for regularizers."""

    #: The batch size
    batch_size: int
    #: The triples factory
    triples_factory: TriplesFactory
    #: Class of regularizer to test
    regularizer_cls: ClassVar[Type[Regularizer]]
    #: The constructor parameters to pass to the regularizer
    regularizer_kwargs: Optional[Dict[str, Any]] = None
    #: The regularizer instance, initialized in setUp
    regularizer: Regularizer
    #: A positive batch
    positive_batch: MappedTriples
    #: The device
    device: torch.device

    def setUp(self) -> None:
        """Set up the test case with a triples factory and model."""
        self.batch_size = 16
        self.triples_factory = NationsTrainingTriplesFactory()
        self.device = resolve_device()
        self.regularizer = self.regularizer_cls(
            device=self.device,
            **(self.regularizer_kwargs or {}),
        )
        self.positive_batch = self.triples_factory.mapped_triples[:self.batch_size, :].to(device=self.device)

    def test_model(self) -> None:
        """Test whether the regularizer can be passed to a model."""
        # Use RESCAL as it regularizes multiple tensors of different shape.
        model = RESCAL(
            triples_factory=self.triples_factory,
            regularizer=self.regularizer,
        ).to(self.device)

        # Check if regularizer is stored correctly.
        self.assertEqual(model.regularizer, self.regularizer)

        # Forward pass (should update regularizer)
        model.forward_owa(batch=self.positive_batch)

        # Call post_parameter_update (should reset regularizer)
        model.post_parameter_update()

        # Check if regularization term is reset
        self.assertEqual(0., model.regularizer.term)

    def test_reset(self) -> None:
        """Test method `reset`."""
        # Call method
        self.regularizer.reset()

        self.assertEqual(0., self.regularizer.regularization_term)

    def test_update(self) -> None:
        """Test method `update`."""
        # Generate random tensors
        a = torch.rand(self.batch_size, 10, device=self.device)
        b = torch.rand(self.batch_size, 20, device=self.device)

        # Call update
        self.regularizer.update(a, b)

        # check shape
        self.assertEqual((1,), self.regularizer.term.shape)

        # compute expected term
        exp_penalties = torch.stack([self._expected_penalty(x) for x in (a, b)])
        if self.regularizer.normalize:
            exp_penalties /= torch.tensor([x.numel() for x in (a, b)], device=self.device)
        expected_term = torch.sum(exp_penalties).view(1) * self.regularizer.weight
        assert expected_term.shape == (1,)

        self.assertAlmostEqual(self.regularizer.term.item(), expected_term.item())

    def test_forward(self) -> None:
        """Test the regularizer's `forward` method."""
        # Generate random tensor
        x = torch.rand(self.batch_size, 10)

        # calculate penalty
        penalty = self.regularizer.forward(x=x)

        # check shape
        assert penalty.numel() == 1

        # check value
        expected_penalty = self._expected_penalty(x=x)
        if expected_penalty is None:
            logging.warning(f'{self.__class__.__name__} did not override `_expected_penalty`.')
        else:
            assert (expected_penalty == penalty).all()

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Compute expected penalty for given tensor."""
        return None


class NoRegularizerTest(_RegularizerTestCase, unittest.TestCase):
    """Test the empty regularizer."""

    regularizer_cls = NoRegularizer

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return torch.zeros(1, device=x.device, dtype=x.dtype)


class _LpRegularizerTest(_RegularizerTestCase):
    """Common test for L_p regularizers."""

    regularizer_cls = LpRegularizer

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return torch.norm(x, p=self.regularizer.p)


class L1RegularizerTest(_LpRegularizerTest, unittest.TestCase):
    """Test an L_1 normed regularizer."""

    regularizer_kwargs = {'p': 1}


class NormedL2RegularizerTest(_LpRegularizerTest, unittest.TestCase):
    """Test an L_2 normed regularizer."""

    regularizer_kwargs = {'normalize': True, 'p': 2}


class CombinedRegularizerTest(_RegularizerTestCase, unittest.TestCase):
    """Test the combined regularizer."""

    regularizer_cls = CombinedRegularizer
    regularizer_kwargs = {
        'regularizers': [
            LpRegularizer(weight=0.1, p=1, device=resolve_device()),
            LpRegularizer(weight=0.7, p=2, device=resolve_device()),
        ]
    }

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        regularizers = self.regularizer_kwargs['regularizers']
        return sum(r.weight * r.forward(x) for r in regularizers) / sum(r.weight for r in regularizers)


class PowerSumRegularizerTest(_RegularizerTestCase, unittest.TestCase):
    """Test the power sum regularizer."""

    regularizer_cls = PowerSumRegularizer

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return x.pow(self.regularizer.p).sum()
