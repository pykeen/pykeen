# -*- coding: utf-8 -*-

"""Implementation of TransD."""

from typing import Optional

import torch
import torch.autograd
from torch import nn

from ..base import Model
from ..init import embedding_xavier_normal_
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...utils import clamp_norm, get_embedding_in_canonical_shape

__all__ = [
    'TransD',
]


def _project_entity(
    e: torch.FloatTensor,
    e_p: torch.FloatTensor,
    r: torch.FloatTensor,
    r_p: torch.FloatTensor,
) -> torch.FloatTensor:
    r"""Project entity relation-specific.

    .. math::

        e_{\bot} = M_{re} e
                 = (r_p e_p^T + I^{d_r \times d_e}) e
                 = r_p e_p^T e + I^{d_r \times d_e} e
                 = r_p (e_p^T e) + e'

    and additionally enforces

    .. math::

        \|e_{\bot}\|_2 \leq 1

    :param e: shape: (batch_size, num_entities, d_e)
        The entity embedding.
    :param e_p: shape: (batch_size, num_entities, d_e)
        The entity projection.
    :param r: shape: (batch_size, num_entities, d_r)
        The relation embedding.
    :param r_p: shape: (batch_size, num_entities, d_r)
        The relation projection.

    :return: shape: (batch_size, num_entities, d_r)

    """
    # The dimensions affected by e'
    change_dim = min(e.shape[-1], r.shape[-1])

    # Project entities
    # r_p (e_p.T e) + e'
    e_bot = r_p * torch.sum(e_p * e, dim=-1, keepdim=True)
    e_bot[:, :, :change_dim] += e[:, :, :change_dim]

    # Enforce constraints
    e_bot = clamp_norm(e_bot, p=2, dim=-1, maxnorm=1)

    return e_bot


class TransD(Model):
    """An implementation of TransD from [ji2015]_.

    This model extends TransR to use fewer parameters.

    .. seealso::

       - OpenKE `implementation of TransD <https://github.com/thunlp/OpenKE/blob/master/models/TransD.py>`_
    """

    hpo_default = dict(
        embedding_dim=dict(type=int, low=20, high=300, q=50),
        relation_dim=dict(type=int, low=20, high=300, q=50),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        entity_embeddings: Optional[nn.Embedding] = None,
        entity_projections: Optional[nn.Embedding] = None,
        relation_dim: int = 30,
        relation_embeddings: Optional[nn.Embedding] = None,
        relation_projections: Optional[nn.Embedding] = None,
        loss: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            entity_embeddings=entity_embeddings,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
        self.relation_embedding_dim = relation_dim
        self.relation_embeddings = relation_embeddings
        self.entity_projections = entity_projections
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
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
            embedding_xavier_normal_(self.entity_embeddings)
        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_embedding_dim)
            embedding_xavier_normal_(self.relation_embeddings)
        if self.entity_projections is None:
            self.entity_projections = nn.Embedding(self.num_entities, self.embedding_dim)
            embedding_xavier_normal_(self.entity_projections)
        if self.relation_projections is None:
            self.relation_projections = nn.Embedding(self.num_relations, self.relation_embedding_dim)
            embedding_xavier_normal_(self.relation_projections)

        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.entity_projections = None
        self.relation_embeddings = None
        self.relation_projections = None
        return self

    @staticmethod
    def interaction_function(
        h: torch.FloatTensor,
        h_p: torch.FloatTensor,
        r: torch.FloatTensor,
        r_p: torch.FloatTensor,
        t: torch.FloatTensor,
        t_p: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function for given embeddings.

        The embeddings have to be in a broadcastable shape.

        :param h: shape: (batch_size, num_entities, d_e)
            Head embeddings.
        :param h_p: shape: (batch_size, num_entities, d_e)
            Head projections.
        :param r: shape: (batch_size, num_entities, d_r)
            Relation embeddings.
        :param r_p: shape: (batch_size, num_entities, d_r)
            Relation projections.
        :param t: shape: (batch_size, num_entities, d_e)
            Tail embeddings.
        :param t_p: shape: (batch_size, num_entities, d_e)
            Tail projections.

        :return: shape: (batch_size, num_entities)
            The scores.
        """
        # Project entities
        h_bot = _project_entity(e=h, e_p=h_p, r=r, r_p=r_p)
        t_bot = _project_entity(e=t, e_p=t_p, r=r, r_p=r_p)

        # score = -||h_bot + r - t_bot||_2^2
        return -torch.norm(h_bot + r - t_bot, dim=-1, p=2) ** 2

    def _score(
        self,
        h_ind: Optional[torch.LongTensor] = None,
        r_ind: Optional[torch.LongTensor] = None,
        t_ind: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Evaluate the interaction function.

        :param h_ind: shape: (batch_size,)
            The indices for head entities. If None, score against all.
        :param r_ind: shape: (batch_size,)
            The indices for relations. If None, score against all.
        :param t_ind: shape: (batch_size,)
            The indices for tail entities. If None, score against all.

        :return: The scores, shape: (batch_size, num_entities)
        """
        # Head
        h = get_embedding_in_canonical_shape(embedding=self.entity_embeddings, ind=h_ind)
        h_p = get_embedding_in_canonical_shape(embedding=self.entity_projections, ind=h_ind)

        r = get_embedding_in_canonical_shape(embedding=self.relation_embeddings, ind=r_ind)
        r_p = get_embedding_in_canonical_shape(embedding=self.relation_projections, ind=r_ind)

        t = get_embedding_in_canonical_shape(embedding=self.entity_embeddings, ind=t_ind)
        t_p = get_embedding_in_canonical_shape(embedding=self.entity_projections, ind=t_ind)

        return self.interaction_function(h=h, h_p=h_p, r=r, r_p=r_p, t=t, t_p=t_p)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=hrt_batch[:, 0], r_ind=hrt_batch[:, 1], t_ind=hrt_batch[:, 2])

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=hr_batch[:, 0], r_ind=hr_batch[:, 1], t_ind=None)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=None, r_ind=rt_batch[:, 0], t_ind=rt_batch[:, 1])
