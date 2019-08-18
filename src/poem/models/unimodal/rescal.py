# -*- coding: utf-8 -*-

"""Implementation of RESCAL."""

from typing import Optional

import torch
from torch import nn

from ..base import RegularizedModel
from ...instance_creation_factories import TriplesFactory
from ...typing import OptionalLoss
from ...utils import l2_regularization

__all__ = ['RESCAL']


class RESCAL(RegularizedModel):
    """An implementation of RESCAL from [nickel2011]_.

    This model represents relations as matrices and models interactions between latent features.

    .. seealso::

       - OpenKE `implementation of RESCAL <https://github.com/thunlp/OpenKE/blob/master/models/RESCAL.py>`_
    """

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 50,
            entity_embeddings: Optional[nn.Embedding] = None,
            relation_embeddings: Optional[nn.Embedding] = None,
            criterion: OptionalLoss = None,
            preferred_device: Optional[str] = None,
            random_seed: Optional[int] = None,
            regularization_weight: float = 0.01,
            init: bool = True,
    ) -> None:
        """Initialize the model."""
        if criterion is None:
            criterion = nn.MarginRankingLoss(margin=1., reduction='mean')

        super().__init__(
            regularization_weight=regularization_weight,
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        self.relation_embeddings = relation_embeddings

        if init:
            self.init_empty_weights_()

    def init_empty_weights_(self):  # noqa: D102
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim ** 2)

        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.relation_embeddings = None
        return self

    def forward_owa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        # shape: (b, d)
        h = self.entity_embeddings(batch[:, 0]).view(-1, 1, self.embedding_dim)
        # shape: (b, d, d)
        r = self.relation_embeddings(batch[:, 1]).view(-1, self.embedding_dim, self.embedding_dim)
        # shape: (b, d)
        t = self.entity_embeddings(batch[:, 2]).view(-1, self.embedding_dim, 1)

        # Update regularization term
        self.current_regularization_term = l2_regularization(h, r, t, normalize=True)

        # Compute scores
        scores = h @ r @ t

        return scores[:, :, 0]

    def forward_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(batch[:, 0]).view(-1, 1, self.embedding_dim)
        r = self.relation_embeddings(batch[:, 1]).view(-1, self.embedding_dim, self.embedding_dim)
        t = self.entity_embeddings.weight.transpose(0, 1).view(1, self.embedding_dim, self.num_entities)

        # Update regularization term
        self.current_regularization_term = l2_regularization(h, r, t, normalize=True)

        # Compute scores
        scores = h @ r @ t

        return scores[:, 0, :]

    def forward_inverse_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        """Forward pass using left side (subject) prediction for training with the CWA."""
        # Get embeddings
        h = self.entity_embeddings.weight.view(1, self.num_entities, self.embedding_dim)
        r = self.relation_embeddings(batch[:, 0]).view(-1, self.embedding_dim, self.embedding_dim)
        t = self.entity_embeddings(batch[:, 1]).view(-1, self.embedding_dim, 1)

        # Update regularization term
        self.current_regularization_term = l2_regularization(h, r, t, normalize=True)

        # Compute scores
        scores = h @ r @ t

        return scores[:, :, 0]
