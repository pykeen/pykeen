# -*- coding: utf-8 -*-

"""Implementation of the ConvKB model."""

import logging
from typing import Optional

import torch
import torch.autograd
from torch import nn

from ..base import BaseModule
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory

__all__ = [
    'ConvKB',
]

logger = logging.getLogger(__name__)


class ConvKB(BaseModule):
    """An implementation of ConvKB from [nguyen2018]_.

    .. seealso::

       - Authors' `implementation of ConvKB <https://github.com/daiquocnguyen/ConvKBsE.py>`_
    """

    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=300, q=50),
        hidden_dropout_rate=dict(type=float, low=0.1, high=0.9),
        num_filters=dict(type=int, low=300, high=500, q=50),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        entity_embeddings: Optional[nn.Embedding] = None,
        relation_embeddings: Optional[nn.Embedding] = None,
        hidden_dropout_rate: float = 0.5,
        embedding_dim: int = 200,
        criterion: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        num_filters: int = 400,
        random_seed: Optional[int] = None,
        init: bool = True,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the model.

        To be consistent with the paper, pass entity and relation embeddings pre-trained from TransE.
        """
        super().__init__(
            triples_factory=triples_factory,
            criterion=criterion,
            embedding_dim=embedding_dim,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        self.num_filters = num_filters

        # Embeddings
        if None in (entity_embeddings, relation_embeddings):
            logger.warning('To be consistent with the paper, initialize entity and relation embeddings from TransE.')
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

        # The interaction model
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(1, 3), bias=True)
        self.relu = nn.ReLU()
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_rate)
        self.linear = nn.Linear(embedding_dim * num_filters, 1, bias=True)

        # Finalize initialization
        self._init_weights_on_device()

    def init_empty_weights_(self):  # noqa: D102
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)

        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        # TODO: How to determine whether weights have been initialized
        # Use Xavier initialization for weight; bias to zero
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.linear.bias)

        # Initialize all filters to [0.1, 0.1, -0.1],
        #  c.f. https://github.com/daiquocnguyen/ConvKB/blob/master/model.py#L34-L36
        nn.init.constant_(self.conv.weight[..., :2], 0.1)
        nn.init.constant_(self.conv.weight[..., 2], -0.1)
        nn.init.zeros_(self.conv.bias)

        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.relation_embeddings = None
        return self

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(hrt_batch[:, 0])
        r = self.relation_embeddings(hrt_batch[:, 1])
        t = self.entity_embeddings(hrt_batch[:, 2])

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

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
