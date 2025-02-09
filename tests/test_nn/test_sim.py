"""Tests for the :mod:`pykeen.nn.sim` submodule."""

import itertools
from typing import Literal

import torch
from unittest_templates import GenericTestCase

from pykeen.nn.sim import NegativeKullbackLeiblerDivergence
from pykeen.typing import GaussianDistribution
from pykeen.utils import calculate_broadcasted_elementwise_result_shape


def _torch_kl_similarity(
    h: GaussianDistribution,
    r: GaussianDistribution,
    t: GaussianDistribution,
) -> torch.FloatTensor:
    """Compute KL similarity using torch.distributions.

    :param h: shape: (batch_size, num_heads, 1, 1, d)
        The head entity Gaussian distribution.
    :param r: shape: (batch_size, 1, num_relations, 1, d)
        The relation Gaussian distribution.
    :param t: shape: (batch_size, 1, 1, num_tails, d)
        The tail entity Gaussian distribution.
    :return: torch.Tensor, shape: (s_1, ..., s_k)
        The KL-divergence.

    .. warning ::
        Do not use this method in production code.
    """
    e_mean = h.mean - t.mean
    e_var = h.diagonal_covariance + t.diagonal_covariance

    # allocate result
    batch_size, num_heads, num_relations, num_tails = calculate_broadcasted_elementwise_result_shape(
        e_mean.shape,
        r.mean.shape,
    )[:-1]
    result = h.mean.new_empty(batch_size, num_heads, num_relations, num_tails)
    for bi, hi, ri, ti in itertools.product(
        range(batch_size),
        range(num_heads),
        range(num_relations),
        range(num_tails),
    ):
        # prepare distributions
        e_loc = e_mean[bi, hi, 0, ti, :]
        r_loc = r.mean[bi, 0, ri, 0, :]
        e_cov = torch.diag(e_var[bi, hi, 0, ti, :])
        r_cov = torch.diag(r.diagonal_covariance[bi, 0, ri, 0, :])
        p = torch.distributions.MultivariateNormal(
            loc=e_loc,
            covariance_matrix=e_cov,
        )
        q = torch.distributions.MultivariateNormal(
            loc=r_loc,
            covariance_matrix=r_cov,
        )
        result[bi, hi, ri, ti] = torch.distributions.kl_divergence(p=p, q=q).view(-1)
    return -result


class KullbackLeiblerDivergenceKG2ESimilarityTests(GenericTestCase[NegativeKullbackLeiblerDivergence]):
    """Tests for the vectorized computation of KL divergences."""

    batch_size: int = 2
    num_heads: int = 3
    num_relations: int = 5
    num_tails: int = 7
    d: int = 11

    cls = NegativeKullbackLeiblerDivergence
    kwargs = dict(exact=True)

    def _get(self, name: Literal["h", "r", "t"]):
        if name == "h":
            index = 1
            num = self.num_heads
        elif name == "r":
            index = 2
            num = self.num_relations
        elif name == "t":
            index = 3
            num = self.num_tails
        else:
            raise ValueError(name)
        shape = [self.batch_size, 1, 1, 1, self.d]
        shape[index] = num
        mean = torch.rand(*shape)
        # ensure positivity for variance
        vari = torch.rand(*shape).exp()
        return GaussianDistribution(mean, vari)

    def test_against_torch_builtin(self):
        """Compare value against torch.distributions."""
        # compute using pykeen
        h, r, t = (self._get(name=name) for name in "hrt")
        sim = self.instance(h=h, r=r, t=t)
        sim2 = _torch_kl_similarity(h=h, r=r, t=t)
        self.assertTrue(torch.allclose(sim, sim2), msg=f"Difference: {(sim - sim2).abs()}")

    def test_self_similarity(self):
        """Check value of similarity to self."""
        # e: (batch_size, num_heads, num_tails, d)
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Properties
        # divergence = 0 => similarity = -divergence = 0
        # (h - t), r
        r = self._get(name="r")
        h = GaussianDistribution(mean=2 * r.mean, diagonal_covariance=0.5 * r.diagonal_covariance)
        t = GaussianDistribution(mean=r.mean, diagonal_covariance=0.5 * r.diagonal_covariance)
        sim = self.instance(h=h, r=r, t=t)
        self.assertTrue(torch.allclose(sim, torch.zeros_like(sim)), msg=f"Sim: {sim}")

    def test_value_range(self):
        """Check the value range."""
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Properties
        # divergence >= 0 => similarity = -divergence <= 0
        h, r, t = (self._get(name=name) for name in "hrt")
        sim = self.instance(h=h, r=r, t=t)
        self.assertTrue((sim <= 0).all())
