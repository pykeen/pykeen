# -*- coding: utf-8 -*-

"""Implementation of KG2E."""

import math
from typing import Optional

import torch
import torch.autograd

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss
from ...nn import Embedding, functional as pykeen_functional
from ...nn.functional import KG2E_SIMILARITIES
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint
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
        preferred_device: DeviceHint = None,
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
            entity_constrainer=clamp_norm,
            entity_constrainer_kwargs=dict(maxnorm=1., p=2, dim=-1),
            relation_constrainer=clamp_norm,
            relation_constrainer_kwargs=dict(maxnorm=1., p=2, dim=-1),
        )

        # Similarity function used for distributions
        dist_similarity = dist_similarity.upper()
        if dist_similarity not in KG2E_SIMILARITIES:
            raise ValueError(dist_similarity)
        self.similarity = dist_similarity

        # element-wise covariance bounds
        self.c_min = c_min
        self.c_max = c_max

        # Additional covariance embeddings
        self.entity_covariances = Embedding.init_with_device(
            num_embeddings=triples_factory.num_entities,
            embedding_dim=embedding_dim,
            device=self.device,
            # Ensure positive definite covariances matrices and appropriate size by clamping
            constrainer=torch.clamp,
            constrainer_kwargs=dict(min=self.c_min, max=self.c_max),
        )
        self.relation_covariances = Embedding.init_with_device(
            num_embeddings=triples_factory.num_relations,
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
        return pykeen_functional.kg2e_interaction(
            h_mean=mu_h,
            h_var=sigma_h,
            r_mean=mu_r,
            r_var=sigma_r,
            t_mean=mu_t,
            t_var=sigma_t,
        )

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(
            h_indices=hrt_batch[:, 0],
            r_indices=hrt_batch[:, 1],
            t_indices=hrt_batch[:, 2],
        ).view(hrt_batch.shape[0], 1)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_indices=hr_batch[:, 0], r_indices=hr_batch[:, 1]).view(hr_batch.shape[0], self.num_entities)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(r_indices=rt_batch[:, 0], t_indices=rt_batch[:, 1]).view(rt_batch.shape[0], self.num_entities)
