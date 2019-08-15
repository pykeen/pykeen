# -*- coding: utf-8 -*-

"""Implementation of the RotatE model."""

from typing import Optional

import numpy as np
import torch
import torch.autograd
from torch import nn
from torch.nn import functional

from ..base import BaseModule
from ..init import embedding_xavier_uniform_
from ...instance_creation_factories import TriplesFactory
from ...typing import OptionalLoss

__all__ = [
    'RotatE',
]


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
            entity_embeddings: Optional[nn.Embedding] = None,
            relation_embeddings: Optional[nn.Embedding] = None,
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
            entity_embeddings=entity_embeddings,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
        self.real_embedding_dim = embedding_dim

        # Embeddings
        self.relation_embeddings = relation_embeddings

        # Initialize if necessary
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        """Initialize entity and relation embeddings."""
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
            embedding_xavier_uniform_(self.entity_embeddings)

        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
            # phases randomly between 0 and 2 pi
            nn.init.uniform_(self.relation_embeddings.weight, a=0, b=2.0 * np.pi)

    def _apply_forward_constraints_if_necessary(self) -> None:
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

    @staticmethod
    def interaction_function(
            h: torch.FloatTensor,
            r: torch.FloatTensor,
            t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function of ComplEx for given embeddings.

        The embeddings have to be in a broadcastable shape.

        WARNING: No forward constraints are applied.

        :param h: shape: (..., e, 2)
            Head embeddings. Last dimension corresponds to (real, imag).
        :param r: shape: (..., e, 2)
            Relation embeddings. Last dimension corresponds to (real, imag).
        :param t: shape: (..., e, 2)
            Tail embeddings. Last dimension corresponds to (real, imag).

        :return: shape: (...)
            The scores.
        """
        # Decompose into real and imaginary part
        h_re = h[..., 0]
        h_im = h[..., 1]
        r_re = r[..., 0]
        r_im = r[..., 1]

        # Rotate (=Hadamard product in complex space).
        rot_h = torch.stack([
            h_re * r_re - h_im * r_im,
            h_re * r_im + h_im * r_re,
        ], dim=-1)
        scores = -torch.norm(rot_h - t, dim=(-2, -1))

        return scores

    def forward_owa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Apply forward constraints if necessary
        self._apply_forward_constraints_if_necessary()

        # Get embeddings
        h = self.entity_embeddings(batch[:, 0]).view(-1, self.real_embedding_dim, 2)
        r = self.entity_embeddings(batch[:, 1]).view(-1, self.real_embedding_dim, 2)
        t = self.entity_embeddings(batch[:, 2]).view(-1, self.real_embedding_dim, 2)

        # Compute scores
        scores = self.interaction_function(h=h, r=r, t=t).view(-1, 1)

        return scores

    def forward_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Apply forward constraints if necessary
        self._apply_forward_constraints_if_necessary()

        # Get embeddings
        h = self.entity_embeddings(batch[:, 0]).view(-1, 1, self.real_embedding_dim, 2)
        r = self.entity_embeddings(batch[:, 1]).view(-1, 1, self.real_embedding_dim, 2)

        # Rank against all entities
        t = self.entity_embeddings.weight.view(1, -1, self.real_embedding_dim, 2)

        # Compute scores
        scores = self.interaction_function(h=h, r=r, t=t)

        return scores

    def forward_inverse_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Apply forward constraints if necessary
        self._apply_forward_constraints_if_necessary()

        # Get embeddings
        t = self.entity_embeddings(batch[:, 1]).view(-1, self.embedding_dim // 2, 2, 1)
        r = self.relation_embeddings(batch[:, 0]).view(-1, self.embedding_dim // 2, 1, 2)

        # r expresses a rotation in complex plane.
        # The inverse rotation is expressed by the complex conjugate of r.
        # The score is computed as the distance of the relation-rotated head to the tail.
        # Equivalently, we can rotate the tail by the inverse relation, and measure the distance to the head, i.e.
        # |h * r - t| = |h - conj(r) * t|
        r[:, :, 0, 1] *= -1

        # Compute Hadamard product (=rotate tail by inverse relation)
        er = (t * r)
        rot_t = torch.cat([
            er[:, :, 0, 0] - er[:, :, 1, 1],
            er[:, :, 0, 1] + er[:, :, 1, 0],
        ], dim=-1)

        # Rank against all entities
        h = self.entity_embeddings.weight

        # use negative distance to tail as score
        scores = -torch.norm(h[None, :, :] - rot_t[:, None, :], dim=-1)

        return scores
