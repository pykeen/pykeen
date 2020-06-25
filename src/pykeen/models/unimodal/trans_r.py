# -*- coding: utf-8 -*-

"""Implementation of TransR."""

from typing import Optional

import torch
import torch.autograd
from torch import nn
from torch.nn import functional

from ..base import EntityRelationEmbeddingModel
from ..init import embedding_xavier_uniform_
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...utils import clamp_norm, get_embedding

__all__ = [
    'TransR',
]


class TransR(EntityRelationEmbeddingModel):
    """An implementation of TransR from [lin2015]_.

    This model extends TransE and TransH by considering different vector spaces for entities and relations.

    Constraints:
     * $||h||_2 <= 1$: Done
     * $||r||_2 <= 1$: Done
     * $||t||_2 <= 1$: Done
     * $||h*M_r||_2 <= 1$: Done
     * $||t*M_r||_2 <= 1$: Done

    .. seealso::

       - OpenKE `TensorFlow implementation of TransR
         <https://github.com/thunlp/OpenKE/blob/master/models/TransR.py>`_
       - OpenKE `PyTorch implementation of TransR
         <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransR.py>`_
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=20, high=300, q=50),
        relation_dim=dict(type=int, low=20, high=300, q=50),
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        relation_dim: int = 30,
        scoring_fct_norm: int = 1,
        loss: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            relation_dim=relation_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
        self.scoring_fct_norm = scoring_fct_norm

        # embeddings
        self.relation_projections = get_embedding(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=relation_dim * embedding_dim,
            device=self.device,
        )

        # Finalize initialization
        self.reset_parameters_()

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

    def _reset_parameters_(self):  # noqa: D102
        # TODO: Initialize from TransE
        embedding_xavier_uniform_(self.entity_embeddings)
        embedding_xavier_uniform_(self.relation_embeddings)
        # Initialise relation embeddings to unit length
        functional.normalize(self.relation_embeddings.weight.data, out=self.relation_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_projections.weight.view(
            self.num_relations, self.embedding_dim, self.relation_dim))

    @staticmethod
    def interaction_function(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        m_r: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function for given embeddings.

        The embeddings have to be in a broadcastable shape.

        :param h: shape: (batch_size, num_entities, d_e)
            Head embeddings.
        :param r: shape: (batch_size, num_entities, d_r)
            Relation embeddings.
        :param t: shape: (batch_size, num_entities, d_e)
            Tail embeddings.
        :param m_r: shape: (batch_size, num_entities, d_e, d_r)
            The relation specific linear transformations.

        :return: shape: (batch_size, num_entities)
            The scores.
        """
        # project to relation specific subspace, shape: (b, e, d_r)
        h_bot = h @ m_r
        t_bot = t @ m_r
        # ensure constraints
        h_bot = clamp_norm(h_bot, p=2, dim=-1, maxnorm=1.)
        t_bot = clamp_norm(t_bot, p=2, dim=-1, maxnorm=1.)

        # evaluate score function, shape: (b, e)
        return -torch.norm(h_bot + r - t_bot, dim=-1) ** 2

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hrt_batch[:, 0]).unsqueeze(dim=1)
        r = self.relation_embeddings(hrt_batch[:, 1]).unsqueeze(dim=1)
        t = self.entity_embeddings(hrt_batch[:, 2]).unsqueeze(dim=1)
        m_r = self.relation_projections(hrt_batch[:, 1]).view(-1, self.embedding_dim, self.relation_dim)

        return self.interaction_function(h=h, r=r, t=t, m_r=m_r).view(-1, 1)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hr_batch[:, 0]).unsqueeze(dim=1)
        r = self.relation_embeddings(hr_batch[:, 1]).unsqueeze(dim=1)
        t = self.entity_embeddings.weight.unsqueeze(dim=0)
        m_r = self.relation_projections(hr_batch[:, 1]).view(-1, self.embedding_dim, self.relation_dim)

        return self.interaction_function(h=h, r=r, t=t, m_r=m_r)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings.weight.unsqueeze(dim=0)
        r = self.relation_embeddings(rt_batch[:, 0]).unsqueeze(dim=1)
        t = self.entity_embeddings(rt_batch[:, 1]).unsqueeze(dim=1)
        m_r = self.relation_projections(rt_batch[:, 0]).view(-1, self.embedding_dim, self.relation_dim)

        return self.interaction_function(h=h, r=r, t=t, m_r=m_r)
