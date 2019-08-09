# -*- coding: utf-8 -*-

"""Implementation of the RotatE model."""

import logging
from typing import Optional

import numpy as np
import torch
import torch.autograd
from torch import nn
from torch.nn import functional

from ..base import BaseModule
from ...instance_creation_factories import TriplesFactory
from ...typing import OptionalLoss

__all__ = [
    'RotatE',
]

log = logging.getLogger(__name__)


class RotatE(BaseModule):
    """An implementation of RotatE from [sun2019]_.

     This model uses models relations as cotations in complex plane.

    .. seealso::

       - Author's `implementation of RotatE
         <https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/codes/model.py#L200-L228>`_
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
        """Normalize the length of relation vectors, if the forward constraint has not been applied yet.

        Absolute value of complex number
        |a+ib| = sqrt(a**2 + b**2)

        L2 norm of complex vector:
        ||x||**2 = sum i=1..d |x_i|**2
                 = sum i=1..d (x_i.re**2 + x_i.im**2)
                 = (sum i=1..d x_i.re**2) + (sum i=1..d x_i.im**2)
                 = ||x.re||**2 + ||x.im||**2
                 = || [x.re; x.im] ||**2
        """
        if not self.forward_constraint_applied:
            functional.normalize(self.relation_embeddings.weight.data, out=self.relation_embeddings.weight.data)
            self.forward_constraint_applied = True

    def _rotate_entities(
            self,
            entity_indices: torch.Tensor,
            relation_indices: torch.Tensor,
            inverse: bool = False
    ) -> torch.Tensor:
        """Rotate entity embeddings in complex plane by relation embeddings.

        :param entity_indices: torch.Tensor, dtype: long, shape: (batch_size,)
            The indices of the entities.
        :param relation_indices: torch.Tensor, dtype: long, shape: (batch_size,)
            The indices of the relations.
        :param inverse: bool (default: False)
            Whether to rotate by the inverse of the relation.

        :return: torch.Tensor, dtype: float, shape: (batch_size, embedding_dim)
            The rotated entity embeddings.
        """
        # rotate head embeddings in complex plane (equivalent to Hadamard product)
        e = self.entity_embeddings(entity_indices).view(-1, self.embedding_dim // 2, 2, 1)
        r = self.relation_embeddings(relation_indices).view(-1, self.embedding_dim // 2, 1, 2)

        # r expresses a rotation in complex plane.
        # The inverse rotation is expressed by the complex conjugate of r.
        if inverse:
            r[:, :, 0, 1] *= -1

        # Compute Hadamard product
        er = (e * r)
        rot_e = torch.cat([
            er[:, :, 0, 0] - er[:, :, 1, 1],
            er[:, :, 0, 1] + er[:, :, 1, 0],
        ], dim=-1)

        return rot_e

    def forward_owa(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for training with the OWA."""
        # Apply forward constraints if necessary
        self._apply_forward_constraints_if_necessary()

        # Rotate head by relation
        rot_h = self._rotate_entities(
            entity_indices=batch[:, 0],
            relation_indices=batch[:, 1],
        )

        # Get tail embeddings
        t = self.entity_embeddings(batch[:, 2])

        # use negative distance to tail as score
        scores = -torch.norm(rot_h - t, dim=-1, keepdim=True)

        return scores

    def forward_cwa(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass using right side (object) prediction for training with the CWA."""
        # Apply forward constraints if necessary
        self._apply_forward_constraints_if_necessary()

        # Rotate head by relation
        rot_h = self._rotate_entities(
            entity_indices=batch[:, 0],
            relation_indices=batch[:, 1],
        )

        # Rank against all entities
        t = self.entity_embeddings.weight

        # use negative distance to tail as score
        scores = -torch.norm(rot_h[:, None, :] - t[None, :, :], dim=-1)

        return scores

    def forward_inverse_cwa(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass using left side (subject) prediction for training with the CWA."""
        # Apply forward constraints if necessary
        self._apply_forward_constraints_if_necessary()

        # r expresses a rotation in complex plane.
        # The inverse rotation is expressed by the complex conjugate of r.
        # The score is computed as the distance of the relation-rotated head to the tail.
        # Equivalently, we can rotate the tail by the inverse relation, and measure the distance to the head, i.e.
        # |h * r - t| = |h - conj(r) * t|

        # Rotate head by inverse of relation
        rot_t = self._rotate_entities(entity_indices=batch[:, 1], relation_indices=batch[:, 0], inverse=True)

        # Rank against all entities
        h = self.entity_embeddings.weight

        # use negative distance to tail as score
        scores = -torch.norm(h[None, :, :] - rot_t[:, None, :], dim=-1)

        return scores
