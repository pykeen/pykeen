# -*- coding: utf-8 -*-

"""Implementation of KG2E."""

from typing import Optional

import torch
import torch.autograd
from torch import nn

from ..base import BaseModule
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...utils import clamp_norm, get_embedding_in_canonical_shape

__all__ = [
    'KG2E',
]


def _expected_likelihood(
    mu_e: torch.FloatTensor,
    mu_r: torch.FloatTensor,
    sigma_e: torch.FloatTensor,
    sigma_r: torch.FloatTensor,
    epsilon: float = 1.0e-10,
) -> torch.FloatTensor:
    """
    Compute the similarity based on expected likelihood.

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
    sigma = sigma_r + sigma_e
    safe_sigma = torch.clamp_min(sigma, min=epsilon)
    sigma_inv = torch.reciprocal(safe_sigma)
    mu = mu_r - mu_e
    a = torch.sum(sigma_inv * mu ** 2, dim=-1)
    b = torch.log(torch.norm(safe_sigma, dim=-1))
    return a + b


def _kullback_leibler_similarity(
    mu_e: torch.FloatTensor,
    mu_r: torch.FloatTensor,
    sigma_e: torch.FloatTensor,
    sigma_r: torch.FloatTensor,
    epsilon: float = 1.0e-10,
) -> torch.FloatTensor:
    """Compute the similarity based on KL divergence.

    This is done between two Gaussian distributions given by mean mu_* and diagonal covariance matrix sigma_*.

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
    safe_sigma_r = torch.clamp_min(sigma_r, min=epsilon)
    sigma_r_inv = torch.reciprocal(safe_sigma_r)
    mu = mu_r - mu_e
    a = torch.sum(sigma_e * sigma_r_inv, dim=-1)
    b = torch.sum(sigma_r_inv * mu ** 2, dim=-1)
    c = -torch.log(sigma_e.norm(p=2, dim=-1) / sigma_r.norm(p=2, dim=-1).clamp_min(min=epsilon))
    return a + b + c


class KG2E(BaseModule):
    """An implementation of KG2E from [he2015]_.

    This model represents entities and relations as multi-dimensional Gaussian distributions.

    Each relation is represented as
        R ~ N(mu_r, Sigma_r)
    Each entity is represented as
        E ~ N(mu_e, Sigma_e)

    For scoring, we compare E = (H - T) with R using a similarity function on distributions (KL div,
    Expected Likelihood).
    """

    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
        c_min=dict(type=float, low=0.01, high=0.1, scale='log'),
        c_max=dict(type=float, low=1.0, high=10.0),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        entity_embeddings: Optional[nn.Embedding] = None,
        relation_embeddings: Optional[nn.Embedding] = None,
        entity_covariances: Optional[nn.Embedding] = None,
        relation_covariances: Optional[nn.Embedding] = None,
        criterion: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        dist_similarity: Optional[str] = None,
        c_min: float = 0.05,
        c_max: float = 5.,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        # Similarity function used for distributions
        if dist_similarity is None or dist_similarity == 'KL':
            dist_similarity = _kullback_leibler_similarity
        elif dist_similarity == 'EL':
            dist_similarity = _expected_likelihood
        else:
            raise KeyError(f'Unknown distribution similarity: {dist_similarity}')
        self.similarity = dist_similarity

        # element-wise covariance bounds
        self.c_min = c_min
        self.c_max = c_max

        # Additional embeddings
        self.relation_embeddings = relation_embeddings
        self.entity_covariances = entity_covariances
        self.relation_covariances = relation_covariances

        # Finalize initialization
        self._init_weights_on_device()

    def init_empty_weights_(self):  # noqa: D102
        # means are restricted to max norm of 1
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        # covariance constraints are applied at _apply_forward_constraints_if_necessary
        if self.entity_covariances is None:
            self.entity_covariances = nn.Embedding(self.num_entities, self.embedding_dim)
        if self.relation_covariances is None:
            self.relation_covariances = nn.Embedding(self.num_relations, self.embedding_dim)

        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.entity_covariances = None
        self.relation_embeddings = None
        self.relation_covariances = None
        return self

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
