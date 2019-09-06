# -*- coding: utf-8 -*-

"""Implementation of NTN."""

from typing import Optional

import torch
from torch import nn

from ..base import BaseModule
from ...triples import TriplesFactory
from ...typing import OptionalLoss

__all__ = ['NTN']


class NTN(BaseModule):
    """An implementation of NTN from [socher2013]_.

    In NTN, a bilinear tensor layer relates the two entity vectors across multiple dimensions.

    Scoring function:
        u_R.T . f(h.T . W_R^[1:k] . t + V_r . [h; t] + b_R)

    where h.T . W_R^[1:k] . t denotes the bilinear tensor product.

    .. seealso::

       - Original Implementation (Matlab): `<https://github.com/khurram18/NeuralTensorNetworks>`_
       - TensorFlow: `<https://github.com/dddoss/tensorflow-socher-ntn>`_
       - Keras: `<https://github.com/dapurv5/keras-neural-tensor-layer (Keras)>`_
    """

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 100,
            num_slices: int = 4,
            entity_embeddings: Optional[nn.Embedding] = None,
            w_relation: Optional[nn.Embedding] = None,
            v_relation: Optional[nn.Embedding] = None,
            b_relation: Optional[nn.Embedding] = None,
            u_relation: Optional[nn.Embedding] = None,
            criterion: OptionalLoss = None,
            preferred_device: Optional[str] = None,
            random_seed: Optional[int] = None,
            non_linearity=nn.Tanh(),
            init: bool = True,
    ) -> None:
        """Initialize the model."""
        if criterion is None:
            criterion = nn.MarginRankingLoss(margin=1., reduction='mean')

        super(NTN, self).__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        self.num_slices = num_slices

        self.w_relation = w_relation
        self.v_relation = v_relation
        self.b_relation = b_relation
        self.u_relation = u_relation
        self.non_linearity = non_linearity

        if init:
            self.init_empty_weights_()

    def init_empty_weights_(self):  # noqa: D102
        # Initialize entity embeddings
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        # bi-linear tensor layer
        # W_R: (d, d, k); store as (k, d, d)
        if self.w_relation is None:
            self.w_relation = nn.Embedding(self.num_relations, self.embedding_dim ** 2 * self.num_slices)
        # V_R: (k, 2d)
        if self.v_relation is None:
            self.v_relation = nn.Embedding(self.num_relations, 2 * self.embedding_dim * self.num_slices)
        # b_R: (k,)
        if self.b_relation is None:
            self.b_relation = nn.Embedding(self.num_relations, self.num_slices)
        # u_R: (k,)
        if self.u_relation is None:
            self.u_relation = nn.Embedding(self.num_relations, self.num_slices)

        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.w_relation = None
        self.v_relation = None
        self.b_relation = None
        self.u_relation = None
        return self

    def forward_owa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get entity embeddings
        h = self.entity_embeddings(batch[:, 0])
        t = self.entity_embeddings(batch[:, 2])

        # Get relation embeddings
        r = batch[:, 1]
        w_r = self.w_relation(r).view(-1, self.num_slices, self.embedding_dim, self.embedding_dim)
        v_r = self.v_relation(r).view(-1, self.num_slices, 2 * self.embedding_dim)
        b_r = self.b_relation(r).view(-1, self.num_slices)
        u_r = self.b_relation(r).view(-1, self.num_slices)

        # Apply scoring function
        # h.T . W_R^[1:k] . t
        h_for_w = h.view(-1, 1, 1, self.embedding_dim)
        t_for_w = t.view(-1, 1, self.embedding_dim, 1)
        h_w_t = (h_for_w @ w_r @ t_for_w).view(-1, self.num_slices)

        # V_R . [h; t]
        ht_for_v = torch.cat([h, t]).view(-1, 2 * self.embedding_dim, 1)
        v_h_t = (v_r @ ht_for_v).view(-1, self.num_slices)

        hidden = self.non_linearity(h_w_t + v_h_t + b_r)

        return torch.sum(u_r * hidden, dim=-1, keepdim=True)

    def forward_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # General dimension usage: (b, n, s, ...)
        # b: batch_size
        # n: num_entities
        # s: slices

        # Get entity embeddings
        h = self.entity_embeddings(batch[:, 0])
        t = self.entity_embeddings.weight

        # Get relation embeddings
        r = batch[:, 1]
        w_r = self.w_relation(r).view(-1, 1, self.num_slices, self.embedding_dim, self.embedding_dim)
        v_r = self.v_relation(r).view(-1, 1, self.num_slices, 2 * self.embedding_dim)
        b_r = self.b_relation(r).view(-1, 1, self.num_slices)
        u_r = self.b_relation(r).view(-1, 1, self.num_slices)

        # Apply scoring function
        # h.T . W_R^[1:k] . t
        h_for_w = h.view(-1, 1, 1, 1, self.embedding_dim)
        t_for_w = t.view(1, -1, 1, self.embedding_dim, 1)
        h_w_t = (h_for_w @ w_r @ t_for_w).view(-1, self.num_entities, self.num_slices)

        # V_R . [h; t]
        h_for_v_r = h.view(-1, 1, self.embedding_dim, 1)
        t_for_v_r = t.view(1, -1, self.embedding_dim, 1)
        v_r_for_h = v_r[:, :, :, :self.embedding_dim]
        v_r_for_t = v_r[:, :, :, self.embedding_dim:]
        v_h = (v_r_for_h @ h_for_v_r).view(-1, 1, self.num_slices)
        v_t = (v_r_for_t @ t_for_v_r).view(-1, self.num_entities, self.num_slices)
        # v_h_t = v_h + v_t

        hidden = self.non_linearity((h_w_t + v_h + b_r) + v_t)

        return torch.sum(u_r * hidden, dim=-1)

    def forward_inverse_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # General dimension usage: (b, n, s, ...)
        # b: batch_size
        # n: num_entities
        # s: slices

        # Get entity embeddings
        h = self.entity_embeddings.weight
        t = self.entity_embeddings(batch[:, 1])

        # Get relation embeddings
        r = batch[:, 0]
        w_r = self.w_relation(r).view(-1, 1, self.num_slices, self.embedding_dim, self.embedding_dim)
        v_r = self.v_relation(r).view(-1, 1, self.num_slices, 2 * self.embedding_dim)
        b_r = self.b_relation(r).view(-1, 1, self.num_slices)
        u_r = self.b_relation(r).view(-1, 1, self.num_slices)

        # Apply scoring function
        # h.T . W_R^[1:k] . t
        h_for_w = h.view(1, -1, 1, 1, self.embedding_dim)
        t_for_w = t.view(-1, 1, 1, self.embedding_dim, 1)
        h_w_t = (h_for_w @ w_r @ t_for_w).view(-1, self.num_entities, self.num_slices)

        # V_R . [h; t]
        h_for_v_r = h.view(1, -1, self.embedding_dim, 1)
        t_for_v_r = t.view(-1, 1, self.embedding_dim, 1)
        v_r_for_h = v_r[:, :, :, :self.embedding_dim]
        v_r_for_t = v_r[:, :, :, self.embedding_dim:]
        v_h = (v_r_for_h @ h_for_v_r).view(-1, self.num_entities, self.num_slices)
        v_t = (v_r_for_t @ t_for_v_r).view(-1, 1, self.num_slices)
        # v_h_t = v_h + v_t

        hidden = self.non_linearity((h_w_t + v_h + b_r) + v_t)

        return torch.sum(u_r * hidden, dim=-1)
