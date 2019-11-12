# -*- coding: utf-8 -*-

"""Implementation of the TransE model."""

from typing import Optional

import torch
import torch.autograd
from torch import nn
from torch.nn import functional

from ..base import BaseModule
from ..init import embedding_xavier_uniform_
from ...losses import Loss
from ...triples import TriplesFactory

__all__ = [
    'TransE',
]


class TransE(BaseModule):
    """An implementation of TransE from [bordes2013]_.

     This model considers a relation as a translation from the head to the tail entity.

    .. seealso::

       - OpenKE `implementation of TransE <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py>`_
    """

    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=300, q=50),
        scoring_fct_norm=dict(type=int, low=1, high=2)
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        entity_embeddings: Optional[nn.Embedding] = None,
        relation_embeddings: Optional[nn.Embedding] = None,
        scoring_fct_norm: int = 1,
        criterion: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        init: bool = True,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
        self.scoring_fct_norm = scoring_fct_norm
        self.relation_embeddings = relation_embeddings

        if init:
            self.init_empty_weights_()

    def init_empty_weights_(self):  # noqa: D102
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
            embedding_xavier_uniform_(self.entity_embeddings)
        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
            embedding_xavier_uniform_(self.relation_embeddings)
            # Initialise relation embeddings to unit length
            functional.normalize(self.relation_embeddings.weight.data, out=self.relation_embeddings.weight.data)

        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.relation_embeddings = None
        return self

    def _apply_forward_constraints_if_necessary(self) -> None:
        if not self.forward_constraint_applied:
            functional.normalize(self.entity_embeddings.weight.data, out=self.entity_embeddings.weight.data)
            self.forward_constraint_applied = True

    def forward_owa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Guarantee forward constraints
        self._apply_forward_constraints_if_necessary()

        # Get embeddings
        h = self.entity_embeddings(batch[:, 0])
        r = self.relation_embeddings(batch[:, 1])
        t = self.entity_embeddings(batch[:, 2])

        return -torch.norm(h + r - t, dim=-1, p=self.scoring_fct_norm, keepdim=True)

    def forward_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Guarantee forward constraints
        self._apply_forward_constraints_if_necessary()

        # Get embeddings
        h = self.entity_embeddings(batch[:, 0])
        r = self.relation_embeddings(batch[:, 1])
        t = self.entity_embeddings.weight

        return -torch.norm(h[:, None, :] + r[:, None, :] - t[None, :, :], dim=-1, p=self.scoring_fct_norm)

    def forward_inverse_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Guarantee forward constraints
        self._apply_forward_constraints_if_necessary()

        # Get embeddings
        h = self.entity_embeddings.weight
        r = self.relation_embeddings(batch[:, 0])
        t = self.entity_embeddings(batch[:, 1])

        return -torch.norm(h[None, :, :] + r[:, None, :] - t[:, None, :], dim=-1, p=self.scoring_fct_norm)
