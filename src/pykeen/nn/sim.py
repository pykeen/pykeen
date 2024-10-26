"""Similarity functions."""

import abc
import math

import torch
from class_resolver import ClassResolver
from torch import nn

from ..typing import FloatTensor, GaussianDistribution
from ..utils import at_least_eps, batched_dot, tensor_sum

__all__ = [
    "KG2ESimilarity",
    "kg2e_similarity_resolver",
    "ExpectedLikelihood",
    "NegativeKullbackLeiblerDivergence",
]


class KG2ESimilarity(nn.Module, abc.ABC):
    """The similarity between the difference of head and tail distribution and the relation distribution.

    Only implemented for multi-variate Gaussian distributions with diagonal covariance matrix.
    """

    def __init__(self, exact: bool = True):
        """
        Initialize the similarity module.

        :param exact:
            Whether to return the exact similarity, or leave out constant offsets for slightly improved speed.
        """
        super().__init__()
        self.exact = exact

    @abc.abstractmethod
    def forward(self, h: GaussianDistribution, r: GaussianDistribution, t: GaussianDistribution) -> FloatTensor:
        """
        Calculate the similarity.

        # noqa: DAR401

        :param h: shape: (`*batch_dims`, `d`)
            The head entity Gaussian distribution.
        :param r: shape: (`*batch_dims`, `d`)
            The relation Gaussian distribution.
        :param t: shape: (`*batch_dims`, `d`)
            The tail entity Gaussian distribution.

        :return: torch.Tensor, shape: (`*batch_dims`)  # noqa: DAR202
            The similarity.
        """
        raise NotImplementedError


class ExpectedLikelihood(KG2ESimilarity):
    r"""Compute the similarity based on expected likelihood.

    Denoting :math:`\mu = \mu_e - \mu_r` and :math:`\Sigma = \Sigma_e + \Sigma_t`, it is given by

    .. math::

        sim(\mathcal{N}(\mu_e, \Sigma_e),~\mathcal{N}(\mu_r, \Sigma_r)))
        = \frac{1}{2} \left(
            \mu^T\Sigma^{-1}\mu
            + \log \det \Sigma + d \log (2 \pi)
        \right)
    """

    # docstr-coverage: inherited
    def forward(self, h: GaussianDistribution, r: GaussianDistribution, t: GaussianDistribution) -> FloatTensor:
        var = tensor_sum(*(d.diagonal_covariance for d in (h, r, t)))
        mean = tensor_sum(h.mean, -t.mean, -r.mean)

        #: a = \mu^T\Sigma^{-1}\mu
        safe_sigma = at_least_eps(var)
        sim = batched_dot(a=safe_sigma.reciprocal(), b=(mean**2))

        #: b = \log \det \Sigma
        sim = sim + safe_sigma.log().sum(dim=-1)
        if not self.exact:
            return sim
        return sim + sim.shape[-1] * math.log(2.0 * math.pi)


class NegativeKullbackLeiblerDivergence(KG2ESimilarity):
    r"""Compute the negative KL divergence.

    Denoting :math:`\mu = \mu_e - \mu_r`, the similarity is given by

    .. math::

        sim(\mathcal{N}(\mu_e, \Sigma_e),~\mathcal{N}(\mu_r, \Sigma_r)) = -\frac{1}{2} \left(
            tr\left(\Sigma_r^{-1} \Sigma_e\right)
            + \mu^T \Sigma_r^{-1} \mu
            - k
            + \ln \left(\det(\Sigma_r) / \det(\Sigma_e)\right)
        \right)

    Since all covariance matrices are diagonal, we can further simplify:

    .. math::
        tr\left(\Sigma_r^{-1} \Sigma_e\right)
        &=&
        \sum_i \Sigma_e[i] / \Sigma_r[i]
        \\
        \mu^T \Sigma_r^{-1} \mu
        &=&
        \sum_i \mu[i]^2 / \Sigma_r[i]
        \\
        \ln \left(\det(\Sigma_r) / \det(\Sigma_e)\right)
        &=&
        \sum_i \ln \Sigma_r[i] - \sum_i \ln \Sigma_e[i]

    .. seealso ::
        `Wikipedia: Multivariate_normal_distribution > Kullback-Leibler Divergence
        <https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence>`_
    """

    # docstr-coverage: inherited
    def forward(self, h: GaussianDistribution, r: GaussianDistribution, t: GaussianDistribution) -> FloatTensor:
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
        if self.exact:
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
        if self.exact:
            result = 0.5 * result
        return -result


#: A resolver for similarities for :class:`pykeen.nn.modules.KG2EInteraction`
kg2e_similarity_resolver: ClassResolver[KG2ESimilarity] = ClassResolver.from_subclasses(
    base=KG2ESimilarity,
    synonyms={"kl": NegativeKullbackLeiblerDivergence, "el": ExpectedLikelihood},
    default=NegativeKullbackLeiblerDivergence,
)
