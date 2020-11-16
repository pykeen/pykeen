# -*- coding: utf-8 -*-

"""Similarity functions."""

import math

import torch

from ..typing import GaussianDistribution

__all__ = [
    'expected_likelihood',
    'kullback_leibler_similarity',
    'KG2E_SIMILARITIES',
]


def expected_likelihood(
    e: GaussianDistribution,
    r: GaussianDistribution,
    epsilon: float = 1.0e-10,
    exact: bool = True,
) -> torch.FloatTensor:
    r"""Compute the similarity based on expected likelihood.

    .. math::

        D((\mu_e, \Sigma_e), (\mu_r, \Sigma_r)))
        = \frac{1}{2} \left(
            (\mu_e - \mu_r)^T(\Sigma_e + \Sigma_r)^{-1}(\mu_e - \mu_r)
            + \log \det (\Sigma_e + \Sigma_r) + d \log (2 \pi)
        \right)
        = \frac{1}{2} \left(
            \mu^T\Sigma^{-1}\mu
            + \log \det \Sigma + d \log (2 \pi)
        \right)

    :param e: shape: (batch_size, num_heads, num_tails, d)
        The entity Gaussian distribution.
    :param r: shape: (batch_size, num_relations, d)
        The relation Gaussian distribution.
    :param epsilon: float (default=1.0)
        Small constant used to avoid numerical issues when dividing.
    :param exact:
        Whether to return the exact similarity, or leave out constant offsets.

    :return: torch.Tensor, shape: (s_1, ..., s_k)
        The similarity.
    """
    # subtract, shape: (batch_size, num_heads, num_relations, num_tails, dim)
    r_shape = r.mean.shape
    r_shape = (r_shape[0], 1, r_shape[1], 1, r_shape[2])
    var = r.diagonal_covariance.view(*r_shape) + e.diagonal_covariance.unsqueeze(dim=2)
    mean = e.mean.unsqueeze(dim=2) - r.mean.view(*r_shape)

    #: a = \mu^T\Sigma^{-1}\mu
    safe_sigma = torch.clamp_min(var, min=epsilon)
    sigma_inv = torch.reciprocal(safe_sigma)
    sim = torch.sum(sigma_inv * mean ** 2, dim=-1)

    #: b = \log \det \Sigma
    sim = sim + safe_sigma.log().sum(dim=-1)
    if exact:
        sim = sim + sim.shape[-1] * math.log(2. * math.pi)
    return sim


def kullback_leibler_similarity(
    e: GaussianDistribution,
    r: GaussianDistribution,
    epsilon: float = 1.0e-10,
    exact: bool = True,
) -> torch.FloatTensor:
    r"""Compute the similarity based on KL divergence.

    This is done between two Gaussian distributions given by mean `mu_*` and diagonal covariance matrix `sigma_*`.

    .. math::

        D((\mu_e, \Sigma_e), (\mu_r, \Sigma_r)))
        = \frac{1}{2} \left(
            tr(\Sigma_r^{-1}\Sigma_e)
            + (\mu_r - \mu_e)^T\Sigma_r^{-1}(\mu_r - \mu_e)
            - \log \frac{det(\Sigma_e)}{det(\Sigma_r)} - k_e
        \right)

    Note: The sign of the function has been flipped as opposed to the description in the paper, as the
          Kullback Leibler divergence is large if the distributions are dissimilar.

    :param e: shape: (batch_size, num_heads, num_tails, d)
        The entity Gaussian distributions, as mean/diagonal covariance pairs.
    :param r: shape: (batch_size, num_relations, d)
        The relation Gaussian distributions, as mean/diagonal covariance pairs.
    :param epsilon: float (default=1.0)
        Small constant used to avoid numerical issues when dividing.
    :param exact:
        Whether to return the exact similarity, or leave out constant offsets.

    :return: torch.Tensor, shape: (s_1, ..., s_k)
        The similarity.
    """
    # invert covariance, shape: (batch_size, num_relations, d)
    safe_sigma_r = torch.clamp_min(r.diagonal_covariance, min=epsilon)
    sigma_r_inv = torch.reciprocal(safe_sigma_r)

    #: a = tr(\Sigma_r^{-1}\Sigma_e), (batch_size, num_heads, num_relations, num_tails)
    # [(b, h, t, d), (b, r, d) -> (b, 1, r, d) -> (b, 1, d, r)] -> (b, h, t, r) -> (b, h, r, t)
    sim = (e.diagonal_covariance @ sigma_r_inv.unsqueeze(dim=1).transpose(-2, -1)).transpose(-2, -1)

    #: b = (\mu_r - \mu_e)^T\Sigma_r^{-1}(\mu_r - \mu_e)
    r_shape = r.mean.shape
    # mu.shape: (b, h, r, t, d)
    mu = r.mean.view(r_shape[0], 1, r_shape[1], 1, r_shape[2]) - e.mean.unsqueeze(dim=2)
    sim = sim + (mu ** 2 @ sigma_r_inv.view(r_shape[0], 1, r_shape[1], r_shape[2], 1)).squeeze(dim=-1)

    #: c = \log \frac{det(\Sigma_e)}{det(\Sigma_r)}
    # = sum log (sigma_e)_i - sum log (sigma_r)_i
    # ce.shape: (b, h, t)
    ce = e.diagonal_covariance.clamp_min(min=epsilon).log().sum(dim=-1)
    # cr.shape: (b, r)
    cr = safe_sigma_r.log().sum(dim=-1)
    sim = sim + ce.unsqueeze(dim=2) - cr.view(r_shape[0], 1, r_shape[1], 1)

    if exact:
        sim = sim - e.mean.shape[-1]
        sim = 0.5 * sim

    return sim


KG2E_SIMILARITIES = {
    'KL': kullback_leibler_similarity,
    'EL': expected_likelihood,
}
