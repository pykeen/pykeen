# -*- coding: utf-8 -*-

"""Test that regularizers can be executed."""

import logging
import unittest
from typing import Any, ClassVar, Dict, Optional, Type

import torch
from torch.nn import functional

from pykeen.datasets import Nations
from pykeen.models import ConvKB, RESCAL, TransH
from pykeen.regularizers import (
    CombinedRegularizer, LpRegularizer, NoRegularizer, PowerSumRegularizer, Regularizer,
    TransHRegularizer,
)
from pykeen.triples import TriplesFactory
from pykeen.typing import MappedTriples
from pykeen.utils import resolve_device


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
        self.generator = torch.random.manual_seed(seed=42)
        self.batch_size = 16
        self.triples_factory = Nations().training
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
        model.score_hrt(hrt_batch=self.positive_batch)

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
        a = torch.rand(self.batch_size, 10, device=self.device, generator=self.generator)
        b = torch.rand(self.batch_size, 20, device=self.device, generator=self.generator)

        # Call update
        self.regularizer.update(a, b)

        # check shape
        self.assertEqual((1,), self.regularizer.term.shape)

        # compute expected term
        exp_penalties = torch.stack([self._expected_penalty(x) for x in (a, b)])
        expected_term = torch.sum(exp_penalties).view(1) * self.regularizer.weight
        assert expected_term.shape == (1,)

        self.assertAlmostEqual(self.regularizer.term.item(), expected_term.item())

    def test_forward(self) -> None:
        """Test the regularizer's `forward` method."""
        # Generate random tensor
        x = torch.rand(self.batch_size, 10, generator=self.generator)

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
        kwargs = self.regularizer_kwargs
        if kwargs is None:
            kwargs = {}
        p = kwargs.get('p', self.regularizer.p)
        value = x.norm(p=p, dim=-1).mean()
        if kwargs.get('normalize', False):
            dim = torch.as_tensor(x.shape[-1], dtype=torch.float, device=x.device)
            if p == 2:
                value = value / dim.sqrt()
            elif p == 1:
                value = value / dim
            else:
                raise NotImplementedError
        return value


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
        ],
    }

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        regularizers = self.regularizer_kwargs['regularizers']
        return sum(r.weight * r.forward(x) for r in regularizers) / sum(r.weight for r in regularizers)


class PowerSumRegularizerTest(_RegularizerTestCase, unittest.TestCase):
    """Test the power sum regularizer."""

    regularizer_cls = PowerSumRegularizer

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        kwargs = self.regularizer_kwargs
        if kwargs is None:
            kwargs = {}
        p = kwargs.get('p', self.regularizer.p)
        value = x.pow(p).sum(dim=-1).mean()
        if kwargs.get('normalize', False):
            value = value / x.shape[-1]
        return value


class TransHRegularizerTest(unittest.TestCase):
    """Test the TransH regularizer."""

    generator: torch.Generator
    device: torch.device
    regularizer_kwargs: Dict
    num_entities: int
    num_relations: int
    entities_weight: torch.Tensor
    relations_weight: torch.Tensor
    normal_vector_weight: torch.Tensor

    def setUp(self) -> None:
        """Set up the test case."""
        self.generator = torch.random.manual_seed(seed=42)
        self.device = resolve_device()
        self.regularizer_kwargs = {'weight': .5, 'epsilon': 1e-5}
        self.regularizer = TransHRegularizer(
            device=self.device,
            **(self.regularizer_kwargs or {}),
        )
        self.num_entities = 10
        self.num_relations = 5
        self.entities_weight = torch.rand(self.num_entities, 10, device=self.device, generator=self.generator)
        self.relations_weight = torch.rand(self.num_relations, 20, device=self.device, generator=self.generator)
        self.normal_vector_weight = torch.rand(self.num_relations, 20, device=self.device, generator=self.generator)

    def test_update(self):
        """Test update function of TransHRegularizer."""
        # Tests that exception will be thrown when more than or less than three tensors are passed
        with self.assertRaises(KeyError) as context:
            self.regularizer.update(
                self.entities_weight,
                self.normal_vector_weight,
                self.relations_weight,
                torch.rand(self.num_entities, 10, device=self.device, generator=self.generator),
            )
            self.assertTrue('Expects exactly three tensors' in context.exception)

            self.regularizer.update(
                self.entities_weight,
                self.normal_vector_weight,
            )
            self.assertTrue('Expects exactly three tensors' in context.exception)

        # Test that regularization term is computed correctly
        self.regularizer.update(self.entities_weight, self.normal_vector_weight, self.relations_weight)
        expected_term = self._expected_penalty()
        weight = self.regularizer_kwargs.get('weight')
        self.assertAlmostEqual(self.regularizer.term.item(), weight * expected_term.item())

    def _expected_penalty(self) -> torch.FloatTensor:  # noqa: D102
        # Entity soft constraint
        regularization_term = torch.sum(functional.relu(torch.norm(self.entities_weight, dim=-1) ** 2 - 1.0))
        epsilon = self.regularizer_kwargs.get('epsilon')  #

        # Orthogonality soft constraint
        d_r_n = functional.normalize(self.relations_weight, dim=-1)
        regularization_term += torch.sum(
            functional.relu(torch.sum((self.normal_vector_weight * d_r_n) ** 2, dim=-1) - epsilon),
        )

        return regularization_term


class TestOnlyUpdateOnce(unittest.TestCase):
    """Tests for when the regularizer should only update once."""

    generator: torch.Generator
    device: torch.device

    def setUp(self) -> None:
        """Set up the test case."""
        self.generator = torch.random.manual_seed(seed=42)
        self.device = resolve_device()

    def test_lp(self):
        """Test when the Lp regularizer only updates once, like for ConvKB."""
        self.assertIn('apply_only_once', ConvKB.regularizer_default_kwargs)
        self.assertTrue(ConvKB.regularizer_default_kwargs['apply_only_once'])
        regularizer = LpRegularizer(
            device=self.device,
            **ConvKB.regularizer_default_kwargs,
        )
        self._help_test_regularizer(regularizer)

    def test_transh_regularizer(self):
        """Test the TransH regularizer only updates once."""
        self.assertNotIn('apply_only_once', TransH.regularizer_default_kwargs)
        regularizer = TransHRegularizer(
            device=self.device,
            **TransH.regularizer_default_kwargs,
        )
        self._help_test_regularizer(regularizer)

    def _help_test_regularizer(self, regularizer: Regularizer, n_tensors: int = 3):
        self.assertFalse(regularizer.updated)
        self.assertEqual(0.0, regularizer.regularization_term.item())

        # After first update, should change the term
        first_tensors = [
            torch.rand(10, 10, device=self.device, generator=self.generator)
            for _ in range(n_tensors)
        ]
        regularizer.update(*first_tensors)
        self.assertTrue(regularizer.updated)
        self.assertNotEqual(0.0, regularizer.regularization_term.item())
        term = regularizer.regularization_term.clone()

        # After second update, no change should happen
        second_tensors = [
            torch.rand(10, 10, device=self.device, generator=self.generator)
            for _ in range(n_tensors)
        ]
        regularizer.update(*second_tensors)
        self.assertTrue(regularizer.updated)
        self.assertEqual(term, regularizer.regularization_term)

        regularizer.reset()
        self.assertFalse(regularizer.updated)
        self.assertEqual(0.0, regularizer.regularization_term.item())
