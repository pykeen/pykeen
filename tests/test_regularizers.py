# -*- coding: utf-8 -*-

"""Test that regularizers can be executed."""

import unittest
from typing import Sequence

import pytest
import torch
import unittest_templates
from torch.nn import functional

import pykeen.regularizers
from pykeen.utils import get_expected_norm
from tests import cases
from tests.utils import rand


class NoRegularizerTest(cases.RegularizerTestCase):
    """Test the empty regularizer."""

    cls = pykeen.regularizers.NoRegularizer

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return torch.zeros(1, device=x.device, dtype=x.dtype)

    # docstr-coverage: inherited
    def test_apply_only_once(self):  # noqa: D102
        raise unittest.SkipTest()


class L1RegularizerTest(cases.LpRegularizerTest):
    """Test an L_1 normed regularizer."""

    kwargs = dict(p=1)


class NormedL2RegularizerTest(cases.LpRegularizerTest):
    """Test an L_2 normed regularizer."""

    kwargs = dict(p=2, normalize=True)

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

    cls = pykeen.regularizers.CombinedRegularizer
    kwargs = dict(
        regularizers=[
            pykeen.regularizers.LpRegularizer(weight=0.1, p=1),
            pykeen.regularizers.LpRegularizer(weight=0.7, p=2),
        ]
    )

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        assert isinstance(self.instance, pykeen.regularizers.CombinedRegularizer)
        regularizers = self.instance.regularizers
        return sum(r.weight * r(x) for r in regularizers) / sum(r.weight for r in regularizers)


class PowerSumRegularizerTest(cases.RegularizerTestCase):
    """Test the power sum regularizer."""

    cls = pykeen.regularizers.PowerSumRegularizer
    kwargs = dict(
        apply_only_once=True,
    )

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        kwargs = self.instance_kwargs
        p = kwargs.get("p", 2.0)
        value = x.pow(p).sum(dim=-1).mean()
        if kwargs.get("normalize", False):
            value = value / x.shape[-1]
        return value


class NormLimitRegularizerTest(cases.RegularizerTestCase):
    """Test the norm-limit regularizer."""

    cls = pykeen.regularizers.NormLimitRegularizer

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        kwargs = self.instance_kwargs
        p = kwargs.get("p", 2.0)
        power_norm = kwargs.get("power_norm", True)
        if power_norm:
            value = x.pow(p).sum(dim=-1)
        else:
            value = x.norm(p=p, dim=-1)
        max_norm = kwargs.get("max_norm", 1.0)
        return (value - max_norm).relu().sum()


class OrthogonalityRegularizerTest(cases.RegularizerTestCase):
    """Test the orthogonaliy regularizer."""

    cls = pykeen.regularizers.OrthogonalityRegularizer
    kwargs = dict(
        weight=0.5,
        epsilon=1.0e-05,
        # there is an extra test for this case
        apply_only_once=False,
    )

    # docstr-coverage: inherited
    def _generate_update_input(self, requires_grad: bool = False) -> Sequence[torch.FloatTensor]:  # noqa: D102
        # same size tensors
        return (
            rand(self.batch_size, 12, generator=self.generator, device=self.device).requires_grad_(requires_grad),
            rand(self.batch_size, 12, generator=self.generator, device=self.device).requires_grad_(requires_grad),
        )

    # docstr-coverage: inherited
    def _expected_updated_term(self, inputs: Sequence[torch.FloatTensor]) -> torch.FloatTensor:  # noqa: D102
        assert len(inputs) == 2
        return functional.cosine_similarity(*inputs).pow(2).subtract(self.instance_kwargs["epsilon"]).relu().sum()

    # docstr-coverage: inherited
    def test_forward(self) -> None:  # noqa: D102
        raise unittest.SkipTest(f"{self.cls.__name__} cannot be applied to a single tensor.")

    # docstr-coverage: inherited
    def test_model(self) -> None:  # noqa: D102
        raise unittest.SkipTest(f"{self.cls.__name__} is not supported by all models.")

    def test_update_error(self):
        """Test update function of TransHRegularizer."""
        # Tests that exception will be thrown when more than or less than two tensors are passed
        for num in (1, 3):
            with self.assertRaises(ValueError) as context:
                self.instance.update(
                    *(rand(self.batch_size, 10, generator=self.generator, device=self.device) for _ in range(num)),
                )
                self.assertTrue("Expects exactly two tensors" in context.exception)


class TestRegularizerTests(unittest_templates.MetaTestCase[pykeen.regularizers.Regularizer]):
    """Test all regularizers are tested."""

    base_cls = pykeen.regularizers.Regularizer
    base_test = cases.RegularizerTestCase
