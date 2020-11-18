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

from ..utils import tensor_sum


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
    r"""Compute the negative KL divergence.

    This is done between two Gaussian distributions given by mean `mu_*` and diagonal covariance matrix `sigma_*`.
    
    .. math::
    
        D((\mu_0, \Sigma_0), (\mu_1, \Sigma_1)) = 0.5 * (
          tr(\Sigma_1^-1 \Sigma_0)
          + (\mu_1 - \mu_0) * \Sigma_1^-1 (\mu_1 - \mu_0)
          - k
          + ln (det(\Sigma_1) / det(\Sigma_0))
        )

    .. note ::
        This methods assumes diagonal covariance matrices :math:`\Sigma`.

    .. seealso ::
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence

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

    assert (e.diagonal_covariance > 0).all() and (r.diagonal_covariance > 0).all()

    # broadcast shapes to (batch_size, num_heads, num_relations, num_tails, dim)
    e_shape = e.mean.shape  # (batch_size, num_heads, num_tails, dim)
    e_mean = e.mean.view(e_shape[0], e_shape[1], 1, e_shape[2], e_shape[3])
    e_var = e.diagonal_covariance.view(e_shape[0], e_shape[1], 1, e_shape[2], e_shape[3])

    r_shape = r.mean.shape  # (batch_size, num_relations, dim)
    r_mean = r.mean.view(r_shape[0], 1, r_shape[1], 1, r_shape[2])
    r_var: torch.FloatTensor = r.diagonal_covariance.view(r_shape[0], 1, r_shape[1], 1, r_shape[2])

    terms = []

    # 1. Component
    # tr(sigma_1^-1 sigma_0) = sum (sigma_1^-1 sigma_0)[i, i]
    # since sigma_0, sigma_1 are diagonal matrices:
    # = sum (sigma_1^-1[i] sigma_0[i]) = sum (sigma_0[i] / sigma_1[i])
    r_var_safe = r_var.clamp_min(min=epsilon)
    terms.append(
        (e_var / r_var_safe).sum(dim=-1)
    )

    # 2. Component
    # (mu_1 - mu_0) * Sigma_1^-1 (mu_1 - mu_0)
    # with mu = (mu_1 - mu_0)
    # = mu * Sigma_1^-1 mu
    # since Sigma_1 is diagonal
    # = mu**2 / sigma_1
    mu = r_mean - e_mean
    terms.append(
        (mu.pow(2) / r_var_safe).sum(dim=-1)
    )

    # 3. Component
    if exact:
        terms.append(
            -e_shape[-1]
        )

    # 4. Component
    # ln (det(\Sigma_1) / det(\Sigma_0))
    # = ln det Sigma_1 - ln det Sigma_0
    # since Sigma is diagonal, we have det Sigma = prod Sigma[ii]
    # = ln prod Sigma_1[ii] - ln prod Sigma_0[ii]
    # = sum ln Sigma_1[ii] - sum ln Sigma_0[ii]
    e_var_safe = e_var.clamp_min(min=epsilon)
    terms.extend((
        r_var_safe.log().sum(dim=-1),
        -e_var_safe.log().sum(dim=-1)
    ))

    result = tensor_sum(*terms)
    if exact:
        result = 0.5 * result

    return -result


KG2E_SIMILARITIES = {
    'KL': kullback_leibler_similarity,
    'EL': expected_likelihood,
}
