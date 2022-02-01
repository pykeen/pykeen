# -*- coding: utf-8 -*-

"""Tests for the :mod:`pykeen.nn.sim` submodule."""

import itertools
import unittest

import torch

from pykeen.nn.sim import _torch_kl_similarity, kullback_leibler_similarity
from pykeen.typing import GaussianDistribution
from pykeen.utils import convert_to_canonical_shape


class KullbackLeiblerTests(unittest.TestCase):
    """Tests for the vectorized computation of KL divergences."""

    batch_size: int = 2
    num_heads: int = 3
    num_relations: int = 5
    num_tails: int = 7
    d: int = 11

    def setUp(self) -> None:  # noqa: D102
        dims = dict(h=self.num_heads, r=self.num_relations, t=self.num_tails)
        (self.h_mean, self.r_mean, self.t_mean), (self.h_var, self.r_var, self.t_var) = [
            [
                # TODO this is the only place this function is used.
                #  Is there an alternative so we can remove it?
                convert_to_canonical_shape(
                    x=torch.rand(self.batch_size, num, self.d),
                    dim=dim,
                    num=num,
                    batch_size=self.batch_size,
                )
                for dim, num in dims.items()
            ]
            for _ in ("mean", "diagonal_covariance")
        ]
        # ensure positivity
        self.h_var, self.r_var, self.t_var = [x.exp() for x in (self.h_var, self.r_var, self.t_var)]

    def _get(self, name: str):
        if name == "h":
            mean, var = self.h_mean, self.h_var
        elif name == "r":
            mean, var = self.r_mean, self.r_var
        elif name == "t":
            mean, var = self.t_mean, self.t_var
        elif name == "e":
            mean, var = self.h_mean - self.t_mean, self.h_var + self.t_var
        else:
            raise ValueError
        return GaussianDistribution(mean=mean, diagonal_covariance=var)

    def _get_kl_similarity_torch(self):
        # compute using pytorch
        e_mean = self.h_mean - self.t_mean
        e_var = self.h_var + self.t_var
        r_mean, r_var = self.r_var, self.r_mean
        self.assertTrue((e_var > 0).all())
        sim2 = torch.empty(self.batch_size, self.num_heads, self.num_relations, self.num_tails)
        for bi, hi, ri, ti in itertools.product(
            range(self.batch_size),
            range(self.num_heads),
            range(self.num_relations),
            range(self.num_tails),
        ):
            # prepare distributions
            e_loc = e_mean[bi, hi, 0, ti, :]
            r_loc = r_mean[bi, 0, ri, 0, :]
            e_cov = torch.diag(e_var[bi, hi, 0, ti, :])
            r_cov = torch.diag(r_var[bi, 0, ri, 0, :])
            p = torch.distributions.MultivariateNormal(
                loc=e_loc,
                covariance_matrix=e_cov,
            )
            q = torch.distributions.MultivariateNormal(
                loc=r_loc,
                covariance_matrix=r_cov,
            )
            sim2[bi, hi, ri, ti] = -torch.distributions.kl_divergence(p=p, q=q).view(-1)
        return sim2

    def test_against_torch_builtin(self):
        """Compare value against torch.distributions."""
        # compute using pykeen
        h, r, t = [self._get(name=name) for name in "hrt"]
        sim = kullback_leibler_similarity(h=h, r=r, t=t, exact=True)
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
        sim = kullback_leibler_similarity(h=h, r=r, t=t, exact=True)
        self.assertTrue(torch.allclose(sim, torch.zeros_like(sim)), msg=f"Sim: {sim}")

    def test_value_range(self):
        """Check the value range."""
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Properties
        # divergence >= 0 => similarity = -divergence <= 0
        h, r, t = [self._get(name=name) for name in "hrt"]
        sim = kullback_leibler_similarity(h=h, r=r, t=t, exact=True)
        self.assertTrue((sim <= 0).all())
