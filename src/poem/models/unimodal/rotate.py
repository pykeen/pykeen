# -*- coding: utf-8 -*-

"""Implementation of the RotatE model."""

import logging
from typing import Optional

import numpy as np
import torch
import torch.autograd
from torch import nn
from torch.nn import functional

from poem.instance_creation_factories.triples_factory import TriplesFactory
from ..base import BaseModule
from ...typing import OptionalLoss
from ...utils import slice_triples

__all__ = [
    'RotatE',
]

log = logging.getLogger(__name__)


class RotatE(BaseModule):
    """An implementation of RotatE from [sun2019]_.

     This model uses models relations as cotations in complex plane.

    .. seealso::

       - Author's `implementation of RotatE <https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/codes/model.py#L200-L228>`_
    """

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 200,
            criterion: OptionalLoss = None,
            preferred_device: Optional[str] = None,
            random_seed: Optional[int] = None,
    ) -> None:
        if criterion is None:
            criterion = nn.MarginRankingLoss(margin=1., reduction='mean')

        super().__init__(
            triples_factory=triples_factory,
            criterion=criterion,
            embedding_dim=2 * embedding_dim,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        # Embeddings
        self.relation_embeddings = nn.Embedding(self.num_relations, 2 * self.embedding_dim)

        self._initialize()

    def _initialize(self):
        entity_embeddings_init_bound = 6 / np.sqrt(
            self.entity_embeddings.num_embeddings + self.entity_embeddings.embedding_dim,
        )
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-entity_embeddings_init_bound,
            b=+entity_embeddings_init_bound,
        )
        # phases randomly between 0 and 2 pi
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=0,
            b=2.0 * np.pi,
        )

    def apply_forward_constraints(self):
        # Absolute value of complex number
        # |a+ib| = sqrt(a**2 + b**2)
        #
        # L2 norm of complex vector:
        # ||x||**2 = sum i=1..d |x_i|**2
        #          = sum i=1..d (x_i.re**2 + x_i.im**2)
        #          = (sum i=1..d x_i.re**2) + (sum i=1..d x_i.im**2)
        #          = ||x.re||**2 + ||x.im||**2
        #          = || [x.re; x.im] ||**2
        functional.normalize(self.relation_embeddings.weight.data, out=self.relation_embeddings.weight.data)
        self.forward_constraint_applied = True

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

        # rotate head embeddings in complex plane (equivalent to Hadamard product)
        h = head_embeddings.view(-1, self.embedding_dim, 2, 1)
        r = relation_embeddings.view(-1, self.embedding_dim, 1, 2)
        hr = (h * r)
        rot_h = torch.stack([
            hr[:, :, 0, 0] - hr[:, :, 1, 1],
            hr[:, :, 0, 1] + hr[:, :, 1, 0],
        ]).view(-1, 2 * self.embedding_dim)

        # use negative distance to tail as score
        scores = -torch.norm(rot_h - tail_embeddings, dim=-1)

        return scores
