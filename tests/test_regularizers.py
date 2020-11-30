# -*- coding: utf-8 -*-

"""Test that regularizers can be executed."""

import logging
import unittest
from typing import Any, ClassVar, Dict, Optional, Type
from unittest.mock import MagicMock

import torch
from torch import nn
from torch.nn import functional

from pykeen.datasets import Nations
from pykeen.models import ConvKB, ERModel, RESCAL
from pykeen.nn import EmbeddingSpecification
from pykeen.regularizers import (
    CombinedRegularizer, LpRegularizer, NoRegularizer, PowerSumRegularizer, Regularizer,
    TransHRegularizer, collect_regularization_terms,
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
            **(self.regularizer_kwargs or {}),
        ).to(self.device)
        self.positive_batch = self.triples_factory.mapped_triples[:self.batch_size, :].to(device=self.device)

    def test_model(self) -> None:
        """Test whether the regularizer can be passed to a model."""
        # Use RESCAL as it regularizes multiple tensors of different shape.
        model = RESCAL(
            triples_factory=self.triples_factory,
        ).to(self.device)

        # check for regularizer
        assert sum(1 for m in model.modules() if isinstance(m, Regularizer)) > 0

        # Forward pass (should update regularizer)
        model.score_hrt(hrt_batch=self.positive_batch)

        # check that regularization term is accessible
        term = collect_regularization_terms(model)
        assert torch.is_tensor(term)
        assert term.requires_grad

        # second time should be 0.
        term = collect_regularization_terms(model)
        assert term == 0.

    def test_update(self) -> None:
        """Test method `update`."""
        # Generate random tensors
        a = torch.rand(self.batch_size, 10, device=self.device, generator=self.generator)
        b = torch.rand(self.batch_size, 20, device=self.device, generator=self.generator)

        # Call update
        assert self.regularizer.update(a, b)

        # check shape
        assert 1 == self.regularizer.regularization_term.numel()

        # compute expected term
        exp_penalties = torch.stack([self._expected_penalty(x) for x in (a, b)])
        expected_term = torch.sum(exp_penalties).view(1) * self.regularizer.weight
        assert expected_term.shape == (1,)

        observed = self.regularizer.pop_regularization_term()
        self.assertAlmostEqual(observed.item(), expected_term.item())

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

    def test_pop_regularization_term(self):
        """Test pop_regularization_term."""
        regularization_term = self.regularizer.pop_regularization_term()

        # check type
        assert isinstance(regularization_term, float) or torch.is_tensor(regularization_term)

        # float only if there is not real regularization term
        if isinstance(regularization_term, float):
            assert regularization_term == 0.0

        # check that the regularizer has been clear
        assert isinstance(self.regularizer.regularization_term, float)
        assert self.regularizer.regularization_term == 0.0
        assert self.regularizer.updated is False

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
            LpRegularizer(weight=0.1, p=1),
            LpRegularizer(weight=0.7, p=2),
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
    entities_weight: nn.Parameter
    relations_weight: nn.Parameter
    normal_vector_weight: nn.Parameter

    def setUp(self) -> None:
        """Set up the test case."""
        self.generator = torch.random.manual_seed(seed=42)
        self.device = resolve_device()
        self.num_entities = 10
        self.num_relations = 5
        self.entities_weight = self._rand_param(10)
        self.relations_weight = self._rand_param(20)
        self.normal_vector_weight = self._rand_param(20)
        self.weight = .5
        self.epsilon = 1e-5
        self.regularizer_kwargs = dict()
        self.regularizer = TransHRegularizer(
            weight=self.weight, epsilon=self.epsilon,
            entity_embeddings=self.entities_weight,
            normal_vector_embeddings=self.normal_vector_weight,
            relation_embeddings=self.relations_weight,
        )

    def _rand_param(self, n) -> nn.Parameter:
        return nn.Parameter(torch.rand(self.num_entities, n, device=self.device, generator=self.generator))

    def test_update(self):
        """Test update function of TransHRegularizer."""
        # Test that regularization term is computed correctly
        expected_term = self._expected_penalty()
        observed_term = self.regularizer.pop_regularization_term()
        assert torch.allclose(observed_term, self.weight * expected_term)

    def _expected_penalty(self) -> torch.FloatTensor:  # noqa: D102
        # Entity soft constraint
        regularization_term = torch.sum(functional.relu(torch.norm(self.entities_weight, dim=-1) ** 2 - 1.0))

        # Orthogonality soft constraint
        d_r_n = functional.normalize(self.relations_weight, dim=-1)
        regularization_term += torch.sum(
            functional.relu(torch.sum((self.normal_vector_weight * d_r_n) ** 2, dim=-1) - self.epsilon),
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
            **ConvKB.regularizer_default_kwargs,
        )
        self._help_test_regularizer(regularizer)

    def _help_test_regularizer(self, regularizer: Regularizer, n_tensors: int = 3):
        self.assertFalse(regularizer.updated)
        assert 0.0 == regularizer.regularization_term

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

        regularizer.pop_regularization_term()
        self.assertFalse(regularizer.updated)
        assert 0.0 == regularizer.regularization_term


class TestRandom(unittest.TestCase):
    """Test random regularization utilities."""

    def test_collect_regularization_terms(self):
        """Test whether collect_regularization_terms finds and resets all regularization terms."""
        regularizers = [
            LpRegularizer(),
            PowerSumRegularizer(),
            LpRegularizer(p=1, normalize=True, apply_only_once=True),
            PowerSumRegularizer(normalize=True),
        ]
        model = ERModel(
            triples_factory=MagicMock(num_entities=3, num_relations=2),
            interaction=MagicMock(relation_shape=("d",), entity_shape=("d",)),
            entity_representations=EmbeddingSpecification(
                regularizer=regularizers[0],
                embedding_dim=2,
            ),
            relation_representations=EmbeddingSpecification(
                regularizer=regularizers[1],
                embedding_dim=2,
            ),
        )

        # add weighted modules
        linear = nn.Linear(3, 2)
        model.sub_module = nn.ModuleList([
            nn.Sequential(
                linear,
                nn.Linear(2, 3),
            ),
            nn.BatchNorm1d(2),
            linear,  # one module occuring twice
        ])

        # add weight regularizer
        model.append_weight_regularizer(
            parameter="sub_module.0.0.bias",
            regularizer=regularizers[2],
        )
        model.append_weight_regularizer(
            parameter="entity_representations.0._embeddings.weight",
            regularizer=regularizers[3],
        )

        # retrieve all regularization terms
        collect_regularization_terms(model)

        # check that all terms are reset
        found_regularizers = set()
        for module in model.modules():
            if isinstance(module, Regularizer):
                term = module.regularization_term
                assert isinstance(term, float)
                assert term == 0.0
                found_regularizers.add(id(module))

        # check that all regularizers were found
        self.assertEqual(found_regularizers, set(map(id, regularizers)))
