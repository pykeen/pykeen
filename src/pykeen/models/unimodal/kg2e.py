# -*- coding: utf-8 -*-

"""Implementation of KG2E."""

from typing import Any, ClassVar, Mapping, Optional

import torch
import torch.autograd
from torch.nn.init import uniform_

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.modules import KG2EInteraction
from ...typing import Constrainer, Hint, Initializer
from ...utils import clamp_norm

__all__ = [
    "KG2E",
]


class KG2E(ERModel):
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
        c_min=dict(type=float, low=0.01, high=0.1, scale="log"),
        c_max=dict(type=float, low=1.0, high=10.0),
    )

    #: The default settings for the entity constrainer
    constrainer_default_kwargs = dict(maxnorm=1.0, p=2, dim=-1)

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        dist_similarity: Optional[str] = None,
        c_min: float = 0.05,
        c_max: float = 5.0,
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
        :param kwargs: Remaining keyword arguments to forward to :class:`pykeen.models.ERModel`
        """
        super().__init__(
            interaction=KG2EInteraction,
            interaction_kwargs=dict(
                similarity=dist_similarity,
            ),
            entity_representations_kwargs=[
                # mean
                dict(
                    shape=embedding_dim,
                    initializer=entity_initializer,
                    constrainer=entity_constrainer,
                    constrainer_kwargs=entity_constrainer_kwargs or self.constrainer_default_kwargs,
                ),
                # diagonal covariance
                dict(
                    shape=embedding_dim,
                    # Ensure positive definite covariances matrices and appropriate size by clamping
                    constrainer=torch.clamp,
                    constrainer_kwargs=dict(min=c_min, max=c_max),
                ),
            ],
            relation_representations_kwargs=[
                # mean
                dict(
                    shape=embedding_dim,
                    initializer=relation_initializer,
                    constrainer=relation_constrainer,
                    constrainer_kwargs=relation_constrainer_kwargs or self.constrainer_default_kwargs,
                ),
                # diagonal covariance
                dict(
                    shape=embedding_dim,
                    # Ensure positive definite covariances matrices and appropriate size by clamping
                    constrainer=torch.clamp,
                    constrainer_kwargs=dict(min=c_min, max=c_max),
                ),
            ],
            **kwargs,
        )
