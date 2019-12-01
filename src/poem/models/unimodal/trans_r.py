# -*- coding: utf-8 -*-

"""Implementation of TransR."""

from typing import Optional

import torch
import torch.autograd
from torch import nn
from torch.nn import functional

from ..base import BaseModule
from ..init import embedding_xavier_uniform_
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...utils import clamp_norm

__all__ = [
    'TransR',
]


class TransR(BaseModule):
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

    hpo_default = dict(
        embedding_dim=dict(type=int, low=20, high=300, q=50),
        relation_dim=dict(type=int, low=20, high=300, q=50),
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        entity_embeddings: Optional[nn.Embedding] = None,
        relation_dim: int = 30,
        relation_embeddings: Optional[nn.Embedding] = None,
        relation_projections: Optional[nn.Embedding] = None,
        scoring_fct_norm: int = 1,
        criterion: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        init: bool = True,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
        self.relation_embedding_dim = relation_dim
        self.scoring_fct_norm = scoring_fct_norm
        self.relation_embeddings = relation_embeddings
        self.relation_projections = relation_projections

        # Finalize initialization
        self._init_weights_on_device()

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

    def init_empty_weights_(self):  # noqa: D102
        # TODO: Initialize from TransE
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
            embedding_xavier_uniform_(self.entity_embeddings)
        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_embedding_dim)
            embedding_xavier_uniform_(self.relation_embeddings)
            # Initialise relation embeddings to unit length
            functional.normalize(self.relation_embeddings.weight.data, out=self.relation_embeddings.weight.data)
        if self.relation_projections is None:
            self.relation_projections = nn.Embedding(
                self.num_relations,
                self.relation_embedding_dim * self.embedding_dim,
            )
        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.relation_embeddings = None
        return self

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
        h_bot = (h.unsqueeze(dim=-2) @ m_r).squeeze(dim=-2)
        t_bot = (t.unsqueeze(dim=-2) @ m_r).squeeze(dim=-2)

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
        m_r = self.relation_projections(hrt_batch[:, 1]).view(-1, 1, self.embedding_dim, self.relation_embedding_dim)

        return self.interaction_function(h=h, r=r, t=t, m_r=m_r).view(-1, 1)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hr_batch[:, 0]).unsqueeze(dim=1)
        r = self.relation_embeddings(hr_batch[:, 1]).unsqueeze(dim=1)
        t = self.entity_embeddings.weight.unsqueeze(dim=0)
        m_r = self.relation_projections(hr_batch[:, 1]).view(-1, 1, self.embedding_dim, self.relation_embedding_dim)

        return self.interaction_function(h=h, r=r, t=t, m_r=m_r)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings.weight.unsqueeze(dim=0)
        r = self.relation_embeddings(rt_batch[:, 0]).unsqueeze(dim=1)
        t = self.entity_embeddings(rt_batch[:, 1]).unsqueeze(dim=1)
        m_r = self.relation_projections(rt_batch[:, 0]).view(-1, 1, self.embedding_dim, self.relation_embedding_dim)

        return self.interaction_function(h=h, r=r, t=t, m_r=m_r)
