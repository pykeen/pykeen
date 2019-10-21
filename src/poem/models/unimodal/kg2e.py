# -*- coding: utf-8 -*-

"""Implementation of KG2E."""

from typing import Optional

import torch
import torch.autograd
from torch import nn

from ..base import BaseModule
from ...triples import TriplesFactory
from ...typing import OptionalLoss

__all__ = ['KG2E']


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
    c = -torch.log(torch.norm(sigma_e, dim=-1) / torch.clamp_min(torch.norm(sigma_r, dim=-1), min=epsilon))
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

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        entity_embeddings: Optional[nn.Embedding] = None,
        relation_embeddings: Optional[nn.Embedding] = None,
        entity_covariances: Optional[nn.Embedding] = None,
        relation_covariances: Optional[nn.Embedding] = None,
        criterion: OptionalLoss = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        dist_similarity: Optional[str] = None,
        c_min: float = 0.05,
        c_max: float = 5.,
        init: bool = True,
    ) -> None:
        if criterion is None:
            criterion = nn.MarginRankingLoss(margin=1., reduction='mean')

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
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

        if init:
            self.init_empty_weights_()

    def init_empty_weights_(self):  # noqa: D102
        # means are restricted to max norm of 1
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim, max_norm=1)
        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim, max_norm=1)

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

    def _apply_forward_constraints_if_necessary(self) -> None:
        # Ensure positive definite covariances matrices and appropriate size by clamping
        if not self.forward_constraint_applied:
            for cov in (
                self.entity_covariances,
                self.relation_covariances,
            ):
                cov_data = cov.weight.data
                torch.clamp(cov_data, min=self.c_min, max=self.c_max, out=cov_data)
            self.forward_constraint_applied = True

    def forward_owa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Normalize embeddings
        self._apply_forward_constraints_if_necessary()

        # Get embeddings
        mu_h = self.entity_embeddings(batch[:, 0])
        mu_r = self.relation_embeddings(batch[:, 1])
        mu_t = self.entity_embeddings(batch[:, 2])
        sigma_h = self.entity_covariances(batch[:, 0])
        sigma_r = self.relation_covariances(batch[:, 1])
        sigma_t = self.entity_covariances(batch[:, 2])

        # Compute entity distribution
        mu_e = mu_h - mu_t
        sigma_e = sigma_h + sigma_t

        # Compute score
        scores = self.similarity(
            mu_e=mu_e,
            mu_r=mu_r,
            sigma_e=sigma_e,
            sigma_r=sigma_r,
        )
        scores = scores.view(-1, 1)

        return scores

    def forward_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Normalize embeddings
        self._apply_forward_constraints_if_necessary()

        # Get embeddings
        mu_h = self.entity_embeddings(batch[:, 0])
        mu_r = self.relation_embeddings(batch[:, 1])
        mu_t = self.entity_embeddings.weight
        sigma_h = self.entity_covariances(batch[:, 0])
        sigma_r = self.relation_covariances(batch[:, 1])
        sigma_t = self.entity_covariances.weight

        # Compute entity distribution
        mu_e = mu_h[:, None, :] - mu_t[None, :, :]
        sigma_e = sigma_h[:, None, :] + sigma_t[None, :, :]

        # Rank against all entities
        scores = self.similarity(
            mu_e=mu_e,
            mu_r=mu_r[:, None, :],
            sigma_e=sigma_e,
            sigma_r=sigma_r[:, None, :],
        )

        return scores

    def forward_inverse_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Normalize embeddings
        self._apply_forward_constraints_if_necessary()

        # Get embeddings
        mu_h = self.entity_embeddings.weight
        mu_r = self.relation_embeddings(batch[:, 0])
        mu_t = self.entity_embeddings(batch[:, 1])
        sigma_h = self.entity_covariances.weight
        sigma_r = self.relation_covariances(batch[:, 0])
        sigma_t = self.entity_covariances(batch[:, 1])

        # Compute entity distribution
        mu_e = mu_h[None, :, :] - mu_t[:, None, :]
        sigma_e = sigma_h[None, :, :] + sigma_t[:, None, :]

        # Rank against all entities
        scores = self.similarity(
            mu_e=mu_e,
            mu_r=mu_r[:, None, :],
            sigma_e=sigma_e,
            sigma_r=sigma_r[:, None, :],
        )

        return scores
