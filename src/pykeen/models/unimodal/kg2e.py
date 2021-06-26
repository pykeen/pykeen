# -*- coding: utf-8 -*-

"""Implementation of KG2E."""

import math
from typing import Any, ClassVar, Mapping, Optional

import torch
import torch.autograd
from torch.nn.init import uniform_

from ..base import EntityRelationEmbeddingModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.emb import Embedding, EmbeddingSpecification
from ...typing import Constrainer, Hint, Initializer
from ...utils import clamp_norm

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
    ---
    citation:
        author: He
        year: 2015
        link: https://dl.acm.org/doi/10.1145/2806416.2806502
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        c_min=dict(type=float, low=0.01, high=0.1, scale='log'),
        c_max=dict(type=float, low=1.0, high=10.0),
    )

    #: The default settings for the entity constrainer
    constrainer_default_kwargs = dict(maxnorm=1., p=2, dim=-1)

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        dist_similarity: Optional[str] = None,
        c_min: float = 0.05,
        c_max: float = 5.,
        entity_initializer: Hint[Initializer] = uniform_,
        entity_constrainer: Hint[Constrainer] = clamp_norm,  # type: ignore
        entity_constrainer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = uniform_,
        relation_constrainer: Hint[Constrainer] = clamp_norm,  # type: ignore
        relation_constrainer_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        r"""Initialize KG2E.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 350]$.
        :param dist_similarity: Either 'KL' for Kullback-Leibler or 'EL' for expected likelihood. Defaults to KL.
        :param c_min: covariance clamp minimum bound
        :param c_max: covariance clamp maximum bound
        :param entity_initializer: Entity initializer function. Defaults to :func:`torch.nn.init.uniform_`
        :param entity_constrainer: Entity constrainer function. Defaults to :func:`pykeen.utils.clamp_norm`
        :param entity_constrainer_kwargs: Keyword arguments to be used when calling the entity constrainer
        :param relation_initializer: Relation initializer function. Defaults to :func:`torch.nn.init.uniform_`
        :param relation_constrainer: Relation constrainer function. Defaults to :func:`pykeen.utils.clamp_norm`
        :param relation_constrainer_kwargs: Keyword arguments to be used when calling the relation constrainer
        :param kwargs:
            Remaining keyword arguments to forward to :class:`pykeen.models.EntityRelationEmbeddingModel`

        :raises ValueError: if an illegal ``dist_similarity`` is given
        """
        super().__init__(
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
                constrainer_kwargs=entity_constrainer_kwargs or self.constrainer_default_kwargs,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
                constrainer_kwargs=relation_constrainer_kwargs or self.constrainer_default_kwargs,
            ),
            **kwargs,
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
        self.entity_covariances = Embedding.init_with_device(
            num_embeddings=self.num_entities,
            embedding_dim=embedding_dim,
            device=self.device,
            # Ensure positive definite covariances matrices and appropriate size by clamping
            constrainer=torch.clamp,
            constrainer_kwargs=dict(min=self.c_min, max=self.c_max),
        )
        self.relation_covariances = Embedding.init_with_device(
            num_embeddings=self.num_relations,
            embedding_dim=embedding_dim,
            device=self.device,
            # Ensure positive definite covariances matrices and appropriate size by clamping
            constrainer=torch.clamp,
            constrainer_kwargs=dict(min=self.c_min, max=self.c_max),
        )

    def _reset_parameters_(self):  # noqa: D102
        # Constraints are applied through post_parameter_update
        super()._reset_parameters_()
        for emb in [
            self.entity_covariances,
            self.relation_covariances,
        ]:
            emb.reset_parameters()

    def post_parameter_update(self) -> None:  # noqa: D102
        super().post_parameter_update()
        for cov in (
            self.entity_covariances,
            self.relation_covariances,
        ):
            cov.post_parameter_update()

    def _score(
        self,
        h_indices: Optional[torch.LongTensor] = None,
        r_indices: Optional[torch.LongTensor] = None,
        t_indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Compute scores for NTN.

        :param h_indices: shape: (batch_size,)
        :param r_indices: shape: (batch_size,)
        :param t_indices: shape: (batch_size,)

        :return: shape: (batch_size, num_entities)
        """
        # Get embeddings
        mu_h = self.entity_embeddings.get_in_canonical_shape(indices=h_indices)
        mu_r = self.relation_embeddings.get_in_canonical_shape(indices=r_indices)
        mu_t = self.entity_embeddings.get_in_canonical_shape(indices=t_indices)

        sigma_h = self.entity_covariances.get_in_canonical_shape(indices=h_indices)
        sigma_r = self.relation_covariances.get_in_canonical_shape(indices=r_indices)
        sigma_t = self.entity_covariances.get_in_canonical_shape(indices=t_indices)

        # Compute entity distribution
        mu_e = mu_h - mu_t
        sigma_e = sigma_h + sigma_t
        return self.similarity(mu_e=mu_e, mu_r=mu_r, sigma_e=sigma_e, sigma_r=sigma_r)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_indices=hrt_batch[:, 0], r_indices=hrt_batch[:, 1], t_indices=hrt_batch[:, 2]).view(-1, 1)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_indices=hr_batch[:, 0], r_indices=hr_batch[:, 1])

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(r_indices=rt_batch[:, 0], t_indices=rt_batch[:, 1])

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
