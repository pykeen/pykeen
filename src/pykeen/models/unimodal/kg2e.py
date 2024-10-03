"""Implementation of KG2E."""

from collections.abc import Mapping
from typing import Any, ClassVar, Optional

import torch
from class_resolver import HintOrType, OptionalKwargs, ResolverKey, update_docstring_with_resolver_keys
from torch.nn.init import uniform_

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.modules import KG2EInteraction
from ...nn.sim import KG2ESimilarity
from ...typing import Constrainer, Hint, Initializer
from ...utils import clamp_norm

__all__ = [
    "KG2E",
]


class KG2E(ERModel):
    r"""An implementation of KG2E from [he2015]_.

    KG2E aims to explicitly model (un)certainties in entities and relations (e.g. influenced by the number of triples
    observed for these entities and relations). Therefore, entities and relations are represented by probability
    distributions, in particular by multi-variate Gaussian distributions $\mathcal{N}(\mu, \Sigma)$
    where the mean $\mu \in \mathbb{R}^d$ denotes the position in the vector space and the *diagonal* variance
    $\Sigma = diag(\sigma_1, \ldots, \sigma_d) \in \mathbb{R}^{d \times d}$ models the uncertainty.

    Thus, we have two $d$-dimensional vectors each stored in an :class:`~pykeen.nn.representation.Embedding` matrix for
    entities and also relations. The representations are then passed to the :class:`~pykeen.nn.modules.KG2EInteraction`
    function to obtain scores.
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

    @update_docstring_with_resolver_keys(
        ResolverKey(name="dist_similarity", resolver="pykeen.nn.sim.kg2e_similarity_resolver")
    )
    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        dist_similarity: HintOrType[KG2ESimilarity] = None,
        dist_similarity_kwargs: OptionalKwargs = None,
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
        :param dist_similarity:
            The similarity measures for gaussian distributions. Defaults to
            :class:`~pykeen.nn.sim.NegativeKullbackLeiblerDivergence`.
        :param dist_similarity_kwargs:
            Additional keyword-based parameters used to instantiate the similarity.
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
                similarity_kwargs=dist_similarity_kwargs,
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
