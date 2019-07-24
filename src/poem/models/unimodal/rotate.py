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

    .. [sun2019] RotatE: Knowledge Graph Embeddings by relational rotation in complex space
                 Z. Sun and  Z.H. Dong and J.Y. Nie and J. Tang
                 <https://arxiv.org/pdf/1902.10197v1.pdf> ICLR 2019.

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
            embedding_dim=2 * embedding_dim,  # for complex numbers
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        # Embeddings
        self.relation_embeddings = None

        self._init_embeddings()

    def _init_embeddings(self):
        super()._init_embeddings()
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
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

    def _apply_forward_constraints_if_necessary(self):
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

    def forward_owa(
            self,
            batch: torch.tensor,
    ) -> torch.tensor:
        # Apply forward constraints if necessary
        self._apply_forward_constraints_if_necessary()

        h, r, t = slice_triples(batch)

        # rotate head embeddings in complex plane (equivalent to Hadamard product)
        h = self.entity_embeddings(h).view(-1, self.embedding_dim // 2, 2, 1)
        r = self.relation_embeddings(r).view(-1, self.embedding_dim // 2, 1, 2)

        hr = (h * r)
        rot_h = torch.cat([
            hr[:, :, 0, 0] - hr[:, :, 1, 1],
            hr[:, :, 0, 1] + hr[:, :, 1, 0],
        ], dim=-1).view(-1, self.embedding_dim)

        t = self.entity_embeddings(t)

        # use negative distance to tail as score
        scores = -torch.norm(rot_h - t, dim=-1, keepdim=True)

        return scores

    def forward_cwa(
            self,
            batch: torch.tensor,
    ) -> torch.tensor:
        # Apply forward constraints if necessary
        self._apply_forward_constraints_if_necessary()

        # rotate head embeddings in complex plane (equivalent to Hadamard product)
        h = self.entity_embeddings(batch[:, 0]).view(-1, self.embedding_dim // 2, 2, 1)
        r = self.relation_embeddings(batch[:, 1]).view(-1, self.embedding_dim // 2, 1, 2)

        hr = (h * r)
        rot_h = torch.cat([
            hr[:, :, 0, 0] - hr[:, :, 1, 1],
            hr[:, :, 0, 1] + hr[:, :, 1, 0],
        ], dim=-1).view(-1, 1, self.embedding_dim)

        # Rank against all entities
        t = self.entity_embeddings.weight.view(1, -1, self.embedding_dim)

        # use negative distance to tail as score
        scores = -torch.norm(rot_h - t, dim=-1)

        return scores

    def forward_inverse_cwa(
            self,
            batch: torch.tensor,
    ) -> torch.tensor:
        # Apply forward constraints if necessary
        self._apply_forward_constraints_if_necessary()

        # r expresses a rotation in complex plane.
        # The inverse rotation is expressed by the complex conjugate of r.
        # The score is computed as the distance of the relation-rotated head to the tail.
        # Equivalently, we can rotate the tail by the inverse relation, and measure the distance to the head, i.e.
        # |h * r - t| = |h - conj(r) * t|

        # Get relation rotations
        r = self.relation_embeddings(batch[:, 0]).view(-1, self.embedding_dim // 2, 1, 2)

        # rotate tail embeddings in complex plane (equivalent to Hadamard product)
        t = self.entity_embeddings(batch[:, 1]).view(-1, self.embedding_dim // 2, 2, 1)

        # Use complex conjugate of r
        rt = (r * t)
        rot_t = torch.cat([
            rt[:, :, 0, 0] + rt[:, :, 1, 1],
            rt[:, :, 0, 1] - rt[:, :, 1, 0],
        ], dim=-1).view(-1, 1, self.embedding_dim)

        # Rank against all entities
        h = self.entity_embeddings.weight.view(1, -1, self.embedding_dim)

        # use negative distance to tail as score
        scores = -torch.norm(h - rot_t, dim=-1)

        return scores
