# -*- coding: utf-8 -*-

"""Implementation of KG2E."""

import math
from typing import Optional

import torch
import torch.autograd

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...utils import clamp_norm, get_embedding, get_embedding_in_canonical_shape

__all__ = [
    'KG2E',
]

_LOG_2_PI = math.log(2. * math.pi)


class KG2E(EntityRelationEmbeddingModel):
    r"""An implementation of KG2E from [he2015]_.

    KG2E aims to explicitly model (un)certainties in entities and relations (e.g. influenced by the number of triples
    observed for these entities and relations). Therefore, entities and relations are represented by probability
    distributions, in particular by multi-variate Gaussian distributions $\mathcal{N}_i(\mu_i,\Sigma_i)$
    where the mean $\mu_i \in \mathbb{R}^d$ denotes the position in the vector space and the diagonal variance
    $\Sigma_i \in \mathbb{R}^{d \times d}$ models the uncertainty.
    Inspired by the :class:`pykeen.models.TransE` model, relations are modeled as transformations from head to tail
    entities: $\mathcal{H} - \mathcal{T} \approx \mathcal{R}$ where
    $\mathcal{H} \sim \mathcal{N}_h(\mu_h,\Sigma_h)$, $\mathcal{H} \sim \mathcal{N}_t(\mu_t,\Sigma_t)$,
    $\mathcal{R} \sim \mathcal{P}_r = \mathcal{N}_r(\mu_r,\Sigma_r)$, and
    $\mathcal{H} - \mathcal{T} \sim \mathcal{P}_e = \mathcal{N}_{h-t}(\mu_h - \mu_t,\Sigma_h + \Sigma_t)$
    (since head and tail entities are considered to be independent with regards to the relations).
    The interaction model measures the similarity between $\mathcal{P}_e$ and $\mathcal{P}_r$ by
    means of the Kullback-Liebler Divergence (:meth:`KG2E.kullback_leibler_similarity`).

    .. math::
            f(h,r,t) = \mathcal{D_{KL}}(\mathcal{P}_e, \mathcal{P}_r)

    Besides the asymmetric KL divergence, the authors propose a symmetric variant which uses the expected
    likelihood (:meth:`KG2E.expected_likelihood`)

    .. math::
            f(h,r,t) = \mathcal{D_{EL}}(\mathcal{P}_e, \mathcal{P}_r)
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
        c_min=dict(type=float, low=0.01, high=0.1, scale='log'),
        c_max=dict(type=float, low=1.0, high=10.0),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        loss: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        dist_similarity: Optional[str] = None,
        c_min: float = 0.05,
        c_max: float = 5.,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        r"""Initialize KG2E.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 350]$.
        :param dist_similarity: Either 'KL' for kullback-liebler or 'EL' for expected liklihood. Defaults to KL.
        :param c_min:
        :param c_max:
        """
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        # Similarity function used for distributions
        if dist_similarity is None or dist_similarity.upper() == 'KL':
            self.similarity = self.kullback_leibler_similarity
        elif dist_similarity.upper() == 'EL':
            self.similarity = self.expected_likelihood
        else:
            raise ValueError(f'Unknown distribution similarity: "{dist_similarity}".')

        # element-wise covariance bounds
        self.c_min = c_min
        self.c_max = c_max

        # Additional covariance embeddings
        self.entity_covariances = get_embedding(
            num_embeddings=triples_factory.num_entities,
            embedding_dim=embedding_dim,
            device=self.device,
        )
        self.relation_covariances = get_embedding(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=embedding_dim,
            device=self.device,
        )

        # Finalize initialization
        self.reset_parameters_()

    def _reset_parameters_(self):  # noqa: D102
        # Constraints are applied through post_parameter_update
        for emb in [
            self.entity_embeddings,
            self.entity_covariances,
            self.relation_embeddings,
            self.relation_covariances,
        ]:
            emb.reset_parameters()

    def post_parameter_update(self) -> None:  # noqa: D102
        # Make sure to call super first
        super().post_parameter_update()

        # Normalize entity embeddings
        self.entity_embeddings.weight.data = clamp_norm(x=self.entity_embeddings.weight.data, maxnorm=1., p=2, dim=-1)
        self.relation_embeddings.weight.data = clamp_norm(
            x=self.relation_embeddings.weight.data,
            maxnorm=1.,
            p=2,
            dim=-1,
        )

        # Ensure positive definite covariances matrices and appropriate size by clamping
        for cov in (
            self.entity_covariances,
            self.relation_covariances,
        ):
            cov_data = cov.weight.data
            torch.clamp(cov_data, min=self.c_min, max=self.c_max, out=cov_data)

    def _score(
        self,
        h_ind: Optional[torch.LongTensor] = None,
        r_ind: Optional[torch.LongTensor] = None,
        t_ind: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Compute scores for NTN.

        :param h_ind: shape: (batch_size,)
        :param r_ind: shape: (batch_size,)
        :param t_ind: shape: (batch_size,)

        :return: shape: (batch_size, num_entities)
        """
        # Get embeddings
        mu_h = get_embedding_in_canonical_shape(embedding=self.entity_embeddings, ind=h_ind)
        mu_r = get_embedding_in_canonical_shape(embedding=self.relation_embeddings, ind=r_ind)
        mu_t = get_embedding_in_canonical_shape(embedding=self.entity_embeddings, ind=t_ind)

        sigma_h = get_embedding_in_canonical_shape(embedding=self.entity_covariances, ind=h_ind)
        sigma_r = get_embedding_in_canonical_shape(embedding=self.relation_covariances, ind=r_ind)
        sigma_t = get_embedding_in_canonical_shape(embedding=self.entity_covariances, ind=t_ind)

        # Compute entity distribution
        mu_e = mu_h - mu_t
        sigma_e = sigma_h + sigma_t
        return self.similarity(mu_e=mu_e, mu_r=mu_r, sigma_e=sigma_e, sigma_r=sigma_r)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=hrt_batch[:, 0], r_ind=hrt_batch[:, 1], t_ind=hrt_batch[:, 2]).view(-1, 1)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=hr_batch[:, 0], r_ind=hr_batch[:, 1])

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(r_ind=rt_batch[:, 0], t_ind=rt_batch[:, 1])

    @staticmethod
    def expected_likelihood(
        mu_e: torch.FloatTensor,
        mu_r: torch.FloatTensor,
        sigma_e: torch.FloatTensor,
        sigma_r: torch.FloatTensor,
        epsilon: float = 1.0e-10,
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

        :param mu_e: torch.Tensor, shape: (s_1, ..., s_k, d)
            The mean of the first Gaussian.
        :param mu_r: torch.Tensor, shape: (s_1, ..., s_k, d)
            The mean of the second Gaussian.
        :param sigma_e: torch.Tensor, shape: (s_1, ..., s_k, d)
            The diagonal covariance matrix of the first Gaussian.
        :param sigma_r: torch.Tensor, shape: (s_1, ..., s_k, d)
            The diagonal covariance matrix of the second Gaussian.
        :param epsilon: float (default=1.0)
            Small constant used to avoid numerical issues when dividing.

        :return: torch.Tensor, shape: (s_1, ..., s_k)
            The similarity.
        """
        d = sigma_e.shape[-1]
        sigma = sigma_r + sigma_e
        mu = mu_e - mu_r

        #: a = \mu^T\Sigma^{-1}\mu
        safe_sigma = torch.clamp_min(sigma, min=epsilon)
        sigma_inv = torch.reciprocal(safe_sigma)
        a = torch.sum(sigma_inv * mu ** 2, dim=-1)

        #: b = \log \det \Sigma
        b = safe_sigma.log().sum(dim=-1)
        return a + b + d * _LOG_2_PI

    @staticmethod
    def kullback_leibler_similarity(
        mu_e: torch.FloatTensor,
        mu_r: torch.FloatTensor,
        sigma_e: torch.FloatTensor,
        sigma_r: torch.FloatTensor,
        epsilon: float = 1.0e-10,
    ) -> torch.FloatTensor:
        r"""Compute the similarity based on KL divergence.

        This is done between two Gaussian distributions given by mean mu_* and diagonal covariance matrix sigma_*.

        .. math::

            D((\mu_e, \Sigma_e), (\mu_r, \Sigma_r)))
            = \frac{1}{2} \left(
                tr(\Sigma_r^{-1}\Sigma_e)
                + (\mu_r - \mu_e)^T\Sigma_r^{-1}(\mu_r - \mu_e)
                - \log \frac{det(\Sigma_e)}{det(\Sigma_r)} - k_e
            \right)

        Note: The sign of the function has been flipped as opposed to the description in the paper, as the
              Kullback Leibler divergence is large if the distributions are dissimilar.

        :param mu_e: torch.Tensor, shape: (s_1, ..., s_k, d)
            The mean of the first Gaussian.
        :param mu_r: torch.Tensor, shape: (s_1, ..., s_k, d)
            The mean of the second Gaussian.
        :param sigma_e: torch.Tensor, shape: (s_1, ..., s_k, d)
            The diagonal covariance matrix of the first Gaussian.
        :param sigma_r: torch.Tensor, shape: (s_1, ..., s_k, d)
            The diagonal covariance matrix of the second Gaussian.
        :param epsilon: float (default=1.0)
            Small constant used to avoid numerical issues when dividing.

        :return: torch.Tensor, shape: (s_1, ..., s_k)
            The similarity.
        """
        d = mu_e.shape[-1]
        safe_sigma_r = torch.clamp_min(sigma_r, min=epsilon)
        sigma_r_inv = torch.reciprocal(safe_sigma_r)

        #: a = tr(\Sigma_r^{-1}\Sigma_e)
        a = torch.sum(sigma_e * sigma_r_inv, dim=-1)

        #: b = (\mu_r - \mu_e)^T\Sigma_r^{-1}(\mu_r - \mu_e)
        mu = mu_r - mu_e
        b = torch.sum(sigma_r_inv * mu ** 2, dim=-1)

        #: c = \log \frac{det(\Sigma_e)}{det(\Sigma_r)}
        # = sum log (sigma_e)_i - sum log (sigma_r)_i
        c = sigma_e.clamp_min(min=epsilon).log().sum(dim=-1) - safe_sigma_r.log().sum(dim=-1)
        return -0.5 * (a + b - c - d)
