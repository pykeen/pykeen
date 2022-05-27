# -*- coding: utf-8 -*-

"""Similarity functions."""

import itertools
import math

import torch

from .compute_kernel import batched_dot
from ..typing import GaussianDistribution
from ..utils import at_least_eps, calculate_broadcasted_elementwise_result_shape, tensor_sum

__all__ = [
    "expected_likelihood",
    "kullback_leibler_similarity",
    "KG2E_SIMILARITIES",
]


# TODO test
def expected_likelihood(
    h: GaussianDistribution,
    r: GaussianDistribution,
    t: GaussianDistribution,
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

    with :math:`\mu_e = \mu_h - \mu_t` and :math:`\Sigma_e = \Sigma_h + \Sigma_t`.

    :param h: shape: (batch_size, num_heads, 1, 1, d)
        The head entity Gaussian distribution.
    :param r: shape: (batch_size, 1, num_relations, 1, d)
        The relation Gaussian distribution.
    :param t: shape: (batch_size, 1, 1, num_tails, d)
        The tail entity Gaussian distribution.
    :param exact:
        Whether to return the exact similarity, or leave out constant offsets.

    :return: torch.Tensor, shape: (batch_size, num_heads, num_relations, num_tails)
        The similarity.
    """
    # subtract, shape: (batch_size, num_heads, num_relations, num_tails, dim)
    var = tensor_sum(*(d.diagonal_covariance for d in (h, r, t)))
    mean = tensor_sum(h.mean, -t.mean, -r.mean)

    #: a = \mu^T\Sigma^{-1}\mu
    safe_sigma = at_least_eps(var)
    sim = batched_dot(
        a=safe_sigma.reciprocal(),
        b=(mean**2),
    )

    #: b = \log \det \Sigma
    sim = sim + safe_sigma.log().sum(dim=-1)
    if exact:
        sim = sim + sim.shape[-1] * math.log(2.0 * math.pi)
    return sim


def kullback_leibler_similarity(
    h: GaussianDistribution,
    r: GaussianDistribution,
    t: GaussianDistribution,
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

    with :math:`\mu_e = \mu_h - \mu_t` and :math:`\Sigma_e = \Sigma_h + \Sigma_t`.

    .. note ::
        This methods assumes diagonal covariance matrices :math:`\Sigma`.

    .. seealso ::
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence

    :param h: shape: (batch_size, num_heads, 1, 1, d)
        The head entity Gaussian distribution.
    :param r: shape: (batch_size, 1, num_relations, 1, d)
        The relation Gaussian distribution.
    :param t: shape: (batch_size, 1, 1, num_tails, d)
        The tail entity Gaussian distribution.
    :param exact:
        Whether to return the exact similarity, or leave out constant offsets.

    :return: torch.Tensor, shape: (s_1, ..., s_k)
        The similarity.
    """
    assert all((d.diagonal_covariance > 0).all() for d in (h, r, t))
    return -_vectorized_kl_divergence(
        h=h,
        r=r,
        t=t,
        exact=exact,
    )


def _vectorized_kl_divergence(
    h: GaussianDistribution,
    r: GaussianDistribution,
    t: GaussianDistribution,
    exact: bool = True,
) -> torch.FloatTensor:
    r"""Vectorized implementation of KL-divergence.

    Computes the divergence between :math:`\mathcal{N}(\mu_e, \Sigma_e)` and :math:`\mathcal{N}(\mu_r, \Sigma_r)`
    given by

    .. math ::
        \mu_e = \mu_h - \mu_t

        \Sigma_e = \Sigma_h + \Sigma_t

    where all covariance matrices are diagonal. Hence we can simplify

    .. math ::
        D(\mathcal{N}(\mu_e, \Sigma_e), \mathcal{N}(\mu_r, \Sigma_r))
        =
        0.5 * (
          \trace(\Sigma_r^-1 \Sigma_e)
          + (\mu_r - \mu_e) * \Sigma_r^-1 (\mu_r - \mu_e)
          - k
          + \ln (\det(\Sigma_r) / \det(\Sigma_e))
        )
        = 0.5 * (
          \sum_i \Sigma_e[i] / Sigma_r[i]
          + \sum_i \mu[i]^2 / \Sigma_r[i]
          + \sum_i \ln Sigma_r[i]
          - \sum_i \ln Sigma_e[i]
          - k
        )

    where :math:`\mu = \mu_r - \mu_e = \mu_r - \mu_h + \mu_t`

    :param h: shape: (batch_size, num_heads, 1, 1, d)
        The head entity Gaussian distribution.
    :param r: shape: (batch_size, 1, num_relations, 1, d)
        The relation Gaussian distribution.
    :param t: shape: (batch_size, 1, 1, num_tails, d)
        The tail entity Gaussian distribution.
    :param exact:
        Whether to return the exact similarity, or leave out constant offsets.

    :return: torch.Tensor, shape: (s_1, ..., s_k)
        The KL-divergence.
    """
    e_var = h.diagonal_covariance + t.diagonal_covariance
    r_var_safe = at_least_eps(r.diagonal_covariance)
    terms = []
    # 1. Component
    # \sum_i \Sigma_e[i] / Sigma_r[i]
    r_var_safe_reciprocal = r_var_safe.reciprocal()
    terms.append(batched_dot(e_var, r_var_safe_reciprocal))
    # 2. Component
    # (mu_1 - mu_0) * Sigma_1^-1 (mu_1 - mu_0)
    # with mu = (mu_1 - mu_0)
    # = mu * Sigma_1^-1 mu
    # since Sigma_1 is diagonal
    # = mu**2 / sigma_1
    mu = tensor_sum(r.mean, -h.mean, t.mean)
    terms.append(batched_dot(mu.pow(2), r_var_safe_reciprocal))
    # 3. Component
    if exact:
        terms.append(-torch.as_tensor(data=[h.mean.shape[-1]], device=mu.device).squeeze())
    # 4. Component
    # ln (det(\Sigma_1) / det(\Sigma_0))
    # = ln det Sigma_1 - ln det Sigma_0
    # since Sigma is diagonal, we have det Sigma = prod Sigma[ii]
    # = ln prod Sigma_1[ii] - ln prod Sigma_0[ii]
    # = sum ln Sigma_1[ii] - sum ln Sigma_0[ii]
    e_var_safe = at_least_eps(e_var)
    terms.extend(
        (
            r_var_safe.log().sum(dim=-1),
            -e_var_safe.log().sum(dim=-1),
        )
    )
    result = tensor_sum(*terms)
    if exact:
        result = 0.5 * result
    return result


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


KG2E_SIMILARITIES = {
    "KL": kullback_leibler_similarity,
    "EL": expected_likelihood,
}
