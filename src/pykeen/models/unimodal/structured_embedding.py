# -*- coding: utf-8 -*-

"""Implementation of structured model (SE)."""

import functools
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from .. import EntityRelationEmbeddingModel
from ...losses import Loss
from ...nn.emb import EmbeddingSpecification
from ...nn.init import xavier_uniform_
from ...nn.modules import StructuredEmbeddingInteractionFunction
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint
from ...utils import compose

__all__ = [
    'StructuredEmbedding',
]


class StructuredEmbedding(EntityRelationEmbeddingModel):
    r"""An implementation of the Structured Embedding (SE) published by [bordes2011]_.

    SE applies role- and relation-specific projection matrices
    $\textbf{M}_{r}^{h}, \textbf{M}_{r}^{t} \in \mathbb{R}^{d \times d}$ to the head and tail
    entities' embeddings before computing their differences. Then, the $l_p$ norm is applied
    and the result is negated such that smaller differences are considered better.

    .. math::

        f(h, r, t) = - \|\textbf{M}_{r}^{h} \textbf{e}_h  - \textbf{M}_{r}^{t} \textbf{e}_t\|_p

    By employing different projections for the embeddings of the head and tail entities, SE explicitly differentiates
    the role of an entity as either the subject or object.
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        scoring_fct_norm: int = 1,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        r"""Initialize SE.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm: The $l_p$ norm. Usually 1 for SE.
        """
        # Embeddings
        init_bound = 6 / np.sqrt(embedding_dim)
        # Initialise relation embeddings to unit length
        relation_initializer = compose(
            functools.partial(nn.init.uniform_, a=-init_bound, b=+init_bound),
            functional.normalize,
        )
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            relation_dim=embedding_dim ** 2,  # head projection matrices
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            embedding_specification=EmbeddingSpecification(
                initializer=xavier_uniform_,
                constrainer=functional.normalize,
            ),
            relation_embedding_specification=EmbeddingSpecification(
                initializer=relation_initializer,
            ),
        )
        self.second_relation_embedding = EmbeddingSpecification(
            initializer=relation_initializer,
        ).make(
            num_embeddings=self.num_relations,
            embedding_dim=self.relation_dim,
            device=self.device,
        )
        self.interaction_function = StructuredEmbeddingInteractionFunction(
            p=scoring_fct_norm,
            power_norm=False,
        ),

    def forward(
        self,
        h_indices: Optional[torch.LongTensor] = None,
        r_indices: Optional[torch.LongTensor] = None,
        t_indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """Evaluate the given triples.

        :param h_indices: shape: (batch_size,)
            The indices for head entities. If None, score against all.
        :param r_indices: shape: (batch_size,)
            The indices for relations. If None, score against all.
        :param t_indices: shape: (batch_size,)
            The indices for tail entities. If None, score against all.

        :return: The scores, shape: (batch_size, num_entities)
        """
        h = self.entity_embeddings.get_in_canonical_shape(indices=h_indices)
        r_h = self.relation_embeddings.get_in_canonical_shape(indices=r_indices)
        r_t = self.second_relation_embedding.get_in_canonical_shape(indices=r_indices)
        t = self.entity_embeddings.get_in_canonical_shape(indices=t_indices)
        return self.interaction_function(h=h, r=(r_h, r_t), t=t)
