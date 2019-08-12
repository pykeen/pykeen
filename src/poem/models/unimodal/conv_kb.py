# -*- coding: utf-8 -*-

"""Implementation of the ConvKB model."""

import logging
from typing import Optional

import torch
import torch.autograd
from torch import nn

from ..base import BaseModule
from ...instance_creation_factories import TriplesFactory
from ...typing import OptionalLoss

__all__ = [
    'ConvKB',
]

log = logging.getLogger(__name__)


class ConvKB(BaseModule):
    """An implementation of ConvKB from [nguyen2018]_.

    .. seealso::

       - Authors' `implementation of ConvKB <https://github.com/daiquocnguyen/ConvKBsE.py>`_
    """

    def __init__(
            self,
            triples_factory: TriplesFactory,
            entity_embeddings: Optional[nn.Embedding] = None,
            relation_embeddings: Optional[nn.Embedding] = None,
            hidden_dropout_rate: float = 0.5,
            embedding_dim: int = 200,
            criterion: OptionalLoss = None,
            preferred_device: Optional[str] = None,
            num_filters: int = 400,
            random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the model."""
        if criterion is None:
            criterion = nn.MarginRankingLoss(margin=1., reduction='mean')

        super().__init__(
            triples_factory=triples_factory,
            criterion=criterion,
            embedding_dim=embedding_dim,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        self.num_filters = num_filters

        # Embeddings
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

        # Initialize embeddings if necessary
        if None in [self.entity_embeddings, self.relation_embeddings]:
            self._init_embeddings()

        # TODO: Initialize filters to [0.1, 0.1, -0.1],
        #  c.f. https://github.com/daiquocnguyen/ConvKB/blob/master/model.py#L34-L36
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(1, 3), bias=True)

        self.relu = nn.ReLU()
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_rate)
        self.linear = nn.Linear(embedding_dim * num_filters, 1, bias=True)

        # Use Xavier initialization for weight; bias to zero
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def _init_embeddings(self):
        # TODO: Use TransE embeddings for initialization..
        self.entity_embeddings = nn.Embedding(
            num_embeddings=self.num_entities,
            embedding_dim=self.embedding_dim,
        )
        self.relation_embeddings = nn.Embedding(
            num_embeddings=self.num_relations,
            embedding_dim=self.embedding_dim,
        )

    def forward_owa(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for training with the OWA."""
        h = self.entity_embeddings(batch[:, 0])
        r = self.relation_embeddings(batch[:, 1])
        t = self.entity_embeddings(batch[:, 2])

        # Stack to convolution input
        conv_inp = torch.stack([h, r, t], dim=-1).view(-1, 1, self.embedding_dim, 3)

        # Convolution
        conv_out = self.conv(conv_inp).view(-1, self.embedding_dim * self.num_filters)
        hidden = self.relu(conv_out)

        # Apply dropout, cf. https://github.com/daiquocnguyen/ConvKB/blob/master/model.py#L54-L56
        hidden = self.hidden_dropout(hidden)

        # Linear layer for final scores
        scores = self.linear(hidden)

        return scores

    def forward_cwa(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass using right side (object) prediction for training with the CWA."""
        h = self.entity_embeddings(batch[:, 0])
        r = self.relation_embeddings(batch[:, 1])
        t = self.entity_embeddings.weight

        # Explicitly perform convolution to exploit broadcasting

        # Convolve head and relation
        # Shapes:
        #  hr: (batch_size, embedding_dim, 2)
        #  conv.weight: (num_filters, 1, 1, 3)
        #  conv.bias: (num_filters,)
        #  hr_conv_out: (batch_size, embedding_dim, num_filters)
        hr = torch.stack([h, r], dim=-1)
        hr_conv_out = (
            torch.sum(hr[:, :, None, :] * self.conv.weight[None, None, :, 0, 0, :2], dim=-1)
            + self.conv.bias[None, None, :]
        )

        # Convolve tail
        # Shapes:
        #  t: (num_entities, embedding_dim)
        #  conv_t_out: (num_entities, embedding_dim, num_filters)
        t_conv_out = t[:, :, None] * self.conv.weight[None, None, :, 0, 0, 2]

        # Combine
        conv_out = hr_conv_out[:, None, :, :] + t_conv_out[None, :, :, :]
        hidden = self.relu(conv_out)

        # Dropout
        hidden = hidden.view(-1, self.embedding_dim * self.num_filters)
        hidden = self.hidden_dropout(hidden)

        # Linear layer for final scores
        scores = self.linear(hidden)

        return scores.view(-1, self.num_entities)

    def forward_inverse_cwa(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass using left side (subject) prediction for training with the CWA."""
        h = self.entity_embeddings.weight
        r = self.relation_embeddings(batch[:, 0])
        t = self.entity_embeddings(batch[:, 1])

        # Explicitly perform convolution to exploit broadcasting

        # Convolve head and relation
        # Shapes:
        #  rt: (batch_size, embedding_dim, 2)
        #  conv.weight: (num_filters, 1, 1, 3)
        #  conv.bias: (num_filters,)
        #  rt_conv_out: (batch_size, embedding_dim, num_filters)
        rt = torch.stack([r, t], dim=-1)
        rt_conv_out = (
            torch.sum(rt[:, :, None, :] * self.conv.weight[None, None, :, 0, 0, 1:], dim=-1)
            + self.conv.bias[None, None, :]
        )

        # Convolve tail
        # Shapes:
        #  h: (num_entities, embedding_dim)
        #  conv_h_out: (num_entities, embedding_dim, num_filters)
        h_conv_out = h[:, :, None] * self.conv.weight[None, None, :, 0, 0, 0]

        # Combine
        conv_out = rt_conv_out[:, None, :, :] + h_conv_out[None, :, :, :]
        hidden = self.relu(conv_out)

        # Dropout
        hidden = hidden.view(-1, self.embedding_dim * self.num_filters)
        hidden = self.hidden_dropout(hidden)

        # Linear layer for final scores
        scores = self.linear(hidden)

        return scores.view(-1, self.num_entities)
