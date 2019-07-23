# -*- coding: utf-8 -*-

"""Implementation of the ConvKB model."""

import logging
from typing import Optional

import torch
import torch.autograd
from torch import nn

from poem.instance_creation_factories.triples_factory import TriplesFactory
from ..base import BaseModule
from ...constants import CONV_KB_NAME, GPU
from ...utils import slice_triples

__all__ = [
    'ConvKB',
]

log = logging.getLogger(__name__)


class ConvKB(BaseModule):
    """An implementation of ConvKB [nguyen2018].

    .. [nguyen2018] A Novel Embedding Model for Knowledge Base CompletionBased on Convolutional Neural Network
                    D. Q. Nguyen and T. D. Nguyen and D. Q. Nguyen and D. Phung
                    <https://www.aclweb.org/anthology/N18-2053>
                     NAACL-HLT 2018

    .. seealso::

       - Authors' implementation: https://github.com/daiquocnguyen/ConvKBsE.py
    """

    model_name = CONV_KB_NAME

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim=200,
            criterion=nn.MarginRankingLoss(margin=1., reduction='mean'), preferred_device=GPU,
            num_filters: int = 400,
            random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            triples_factory = triples_factory,
            criterion=criterion,
            embedding_dim=embedding_dim,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        # Embeddings
        self.relation_embeddings = nn.Embedding(num_relations, self.embedding_dim)
        self.num_filters = num_filters

        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(1, 3), bias=True)
        self.relu = nn.ReLU()
        self.hidden_dropout = nn.Dropout()
        self.linear = nn.Linear(embedding_dim * num_filters, 1, bias=True)
        self._initialize()

    def _initialize(self):
        # TODO: Use TransE embeddings for initialization..
        # TODO: Initialize filters to [0.1, 0.1, -0.1], c.f. https://github.com/daiquocnguyen/ConvKB/blob/master/model.py#L34-L36

        # Use Xavier initialization for weight; bias to zero
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def _score_triples(self, triples):
        heads, relations, tails = slice_triples(triples)

        # Get embeddings
        head_embeddings = self._get_embeddings(
            heads,
            embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim,
        )
        tail_embeddings = self._get_embeddings(
            tails,
            embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim,
        )
        relation_embeddings = self._get_embeddings(
            relations,
            embedding_module=self.relation_embeddings,
            embedding_dim=self.embedding_dim,
        )

        conv_inp = torch.stack(
            [
                head_embeddings,
                relation_embeddings,
                tail_embeddings,
            ],
            dim=-1,
        ).view(-1, 1, self.embedding_dim, 3)

        # Convolution
        conv_out = self.conv(conv_inp)
        hidden = self.relu(conv_out)

        # Apply dropout, cf. https://github.com/daiquocnguyen/ConvKB/blob/master/model.py#L54-L56
        hidden = self.hidden_dropout(hidden)

        # Linear layer for final scores
        scores = self.linear(hidden).view(-1)

        return scores
