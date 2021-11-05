# -*- coding: utf-8 -*-

"""Test that regularizers can be executed."""

import unittest

import pytest
import torch
import unittest_templates
from torch.nn import functional

from pykeen.models import ConvKB, TransH
from pykeen.regularizers import (
    CombinedRegularizer,
    LpRegularizer,
    NoRegularizer,
    PowerSumRegularizer,
    Regularizer,
    TransHRegularizer,
)
from pykeen.utils import get_expected_norm, resolve_device
from tests import cases
from tests.utils import rand


class NoRegularizerTest(cases.RegularizerTestCase):
    """Test the empty regularizer."""

    cls = NoRegularizer

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return torch.zeros(1, device=x.device, dtype=x.dtype)


class L1RegularizerTest(cases.LpRegularizerTest):
    """Test an L_1 normed regularizer."""

    kwargs = {"p": 1}


class NormedL2RegularizerTest(cases.LpRegularizerTest):
    """Test an L_2 normed regularizer."""

    kwargs = {"normalize": True, "p": 2}

    @pytest.mark.slow
    def test_expected_norm(self):
        """Numerically check expected norm."""
        n = 100
        for p in (1, 2, 3):
            for d in (2, 8, 64):
                e_norm = get_expected_norm(p=p, d=d)
                norm = torch.randn(n, d).norm(p=p, dim=-1).numpy()
                norm_mean = norm.mean()
                norm_std = norm.std()
                # check if within 0.5 std of observed
                assert (abs(norm_mean - e_norm) / norm_std) < 0.5

        # test error is raised
        with pytest.raises(NotImplementedError):
            get_expected_norm(p=float("inf"), d=d)


class CombinedRegularizerTest(cases.RegularizerTestCase):
    """Test the combined regularizer."""

    cls = CombinedRegularizer
    kwargs = {
        "regularizers": [
            LpRegularizer(weight=0.1, p=1),
            LpRegularizer(weight=0.7, p=2),
        ],
    }

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        regularizers = self.kwargs["regularizers"]
        return sum(r.weight * r.forward(x) for r in regularizers) / sum(r.weight for r in regularizers)


class PowerSumRegularizerTest(cases.RegularizerTestCase):
    """Test the power sum regularizer."""

    cls = PowerSumRegularizer

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        kwargs = self.kwargs
        if kwargs is None:
            kwargs = {}
        p = kwargs.get("p", self.instance.p)
        value = x.pow(p).sum(dim=-1).mean()
        if kwargs.get("normalize", False):
            value = value / x.shape[-1]
        return value


class TransHRegularizerTest(unittest.TestCase):
    """Test the TransH regularizer."""

    cls = TransHRegularizer
    generator: torch.Generator
    device: torch.device

    instance: Regularizer
    num_entities: int
    num_relations: int
    entities_weight: torch.Tensor
    relations_weight: torch.Tensor
    normal_vector_weight: torch.Tensor

    def setUp(self) -> None:
        """Set up the test case."""
        self.generator = torch.random.manual_seed(seed=42)
        self.device = resolve_device()
        self.kwargs = {"weight": 0.5, "epsilon": 1e-5}
        self.instance = self.cls(
            **(self.kwargs or {}),
        ).to(self.device)
        self.num_entities = 10
        self.num_relations = 5
        self.entities_weight = rand(self.num_entities, 10, generator=self.generator, device=self.device)
        self.relations_weight = rand(self.num_relations, 20, generator=self.generator, device=self.device)
        self.normal_vector_weight = rand(self.num_relations, 20, generator=self.generator, device=self.device)

    def test_update(self):
        """Test update function of TransHRegularizer."""
        # Tests that exception will be thrown when more than or less than three tensors are passed
        with self.assertRaises(KeyError) as context:
            self.instance.update(
                self.entities_weight,
                self.normal_vector_weight,
                self.relations_weight,
                rand(self.num_entities, 10, generator=self.generator, device=self.device),
            )
            self.assertTrue("Expects exactly three tensors" in context.exception)

            self.instance.update(
                self.entities_weight,
                self.normal_vector_weight,
            )
            self.assertTrue("Expects exactly three tensors" in context.exception)

        # Test that regularization term is computed correctly
        self.instance.update(self.entities_weight, self.normal_vector_weight, self.relations_weight)
        expected_term = self._expected_penalty()
        weight = self.kwargs.get("weight")
        self.assertAlmostEqual(self.instance.term.item(), weight * expected_term.item())

    def _expected_penalty(self) -> torch.FloatTensor:  # noqa: D102
        # Entity soft constraint
        regularization_term = torch.sum(functional.relu(torch.norm(self.entities_weight, dim=-1) ** 2 - 1.0))
        epsilon = self.kwargs.get("epsilon")  #

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
        self.assertIn("apply_only_once", ConvKB.regularizer_default_kwargs)
        self.assertTrue(ConvKB.regularizer_default_kwargs["apply_only_once"])
        regularizer = LpRegularizer(
            **ConvKB.regularizer_default_kwargs,
        )
        self._help_test_regularizer(regularizer)

    def test_transh_regularizer(self):
        """Test the TransH regularizer only updates once."""
        self.assertNotIn("apply_only_once", TransH.regularizer_default_kwargs)
        regularizer = TransHRegularizer(
            **TransH.regularizer_default_kwargs,
        )
        self._help_test_regularizer(regularizer)

    def _help_test_regularizer(self, regularizer: Regularizer, n_tensors: int = 3):
        # ensure regularizer is on correct device
        regularizer = regularizer.to(self.device)

        self.assertFalse(regularizer.updated)
        self.assertEqual(0.0, regularizer.regularization_term.item())

        # After first update, should change the term
        first_tensors = [rand(10, 10, generator=self.generator, device=self.device) for _ in range(n_tensors)]
        regularizer.update(*first_tensors)
        self.assertTrue(regularizer.updated)
        self.assertNotEqual(0.0, regularizer.regularization_term.item())
        term = regularizer.regularization_term.clone()

        # After second update, no change should happen
        second_tensors = [rand(10, 10, generator=self.generator, device=self.device) for _ in range(n_tensors)]
        regularizer.update(*second_tensors)
        self.assertTrue(regularizer.updated)
        self.assertEqual(term, regularizer.regularization_term)

        regularizer.reset()
        self.assertFalse(regularizer.updated)
        self.assertEqual(0.0, regularizer.regularization_term.item())


class TestRegularizerTests(unittest_templates.MetaTestCase[Regularizer]):
    """Test all regularizers are tested."""

    base_cls = Regularizer
    base_test = cases.RegularizerTestCase
    skip_cls = {TransHRegularizer}
