# -*- coding: utf-8 -*-

"""Implementation of SimplE."""

from typing import Optional

import torch
import torch.autograd
from torch import nn

from ..base import BaseModule
from ...triples import TriplesFactory
from ...typing import OptionalLoss
from ...utils import slice_triples

__all__ = ['SimplE']


class SimplE(BaseModule):
    """An implementation of SimplE [kazemi2018]_.

    This model extends CP by updating a triple, and the inverse triple.

    .. seealso::

       - Official implementation: https://github.com/Mehran-k/SimplE
       - Improved implementation in pytorch: https://github.com/baharefatemi/SimplE
    """

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 200,
            entity_embeddings: Optional[nn.Embedding] = None,
            tail_entity_embeddings: Optional[nn.Embedding] = None,
            relation_embeddings: Optional[nn.Embedding] = None,
            inverse_relation_embeddings: Optional[nn.Embedding] = None,
            criterion: OptionalLoss = None,
            preferred_device: Optional[str] = None,
            random_seed: Optional[int] = None,
            init: bool = True,
    ) -> None:
        if criterion is None:
            criterion = nn.MarginRankingLoss(margin=1., reduction='mean')

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
        self.relation_embeddings = relation_embeddings
        self.tail_entity_embeddings = tail_entity_embeddings
        self.inverse_relation_embeddings = inverse_relation_embeddings

        if init:
            self.init_empty_weights_()

    def init_empty_weights_(self):  # noqa: D102
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        if self.tail_entity_embeddings is None:
            self.tail_entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        if self.inverse_relation_embeddings is None:
            self.inverse_relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.tail_entity_embeddings = None
        self.relation_embeddings = None
        self.inverse_relation_embeddings = None
        return self

    def forward_owa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Split triple in head, relation, tail
        h_ind, r_ind, t_ind = slice_triples(batch)

        # Lookup embeddings
        hh = self.entity_embeddings(h_ind)
        ht = self.entity_embeddings(t_ind)
        th = self.tail_entity_embeddings(h_ind)
        tt = self.tail_entity_embeddings(t_ind)
        r = self.relation_embeddings(r_ind)
        r_inv = self.inverse_relation_embeddings(r_ind)

        # Compute CP scores for triple, and inverse triple
        score = torch.sum(hh * r * tt, dim=-1)
        inverse_score = torch.sum(ht * r_inv * th, dim=-1)

        # Final score is average
        scores = 0.5 * (score + inverse_score)

        # Note: In the code in their repository, the score is clamped to [-20, 20].
        #       That is not mentioned in the paper, so it is omitted here.

        return scores

    def forward_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h_ind = batch[:, 0]
        r_ind = batch[:, 1]

        # Lookup embeddings
        hh = self.entity_embeddings(h_ind)
        th = self.tail_entity_embeddings(h_ind)
        r = self.relation_embeddings(r_ind)
        r_inv = self.inverse_relation_embeddings(r_ind)
        ht = self.entity_embeddings.weight
        tt = self.tail_entity_embeddings.weight

        # Compute CP scores for triple, and inverse triple
        score = torch.sum(hh[:, None, :] * r[:, None, :] * tt[None, :, :], dim=-1)
        inverse_score = torch.sum(ht[None, :, :] * r_inv[:, None, :] * th[:, None, :], dim=-1)

        # Final score is average
        scores = 0.5 * (score + inverse_score)

        return scores

    def forward_inverse_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        r_ind = batch[:, 0]
        t_ind = batch[:, 1]

        # Lookup embeddings
        hh = self.entity_embeddings.weight
        ht = self.entity_embeddings(t_ind)
        th = self.tail_entity_embeddings.weight
        tt = self.tail_entity_embeddings(t_ind)
        r = self.relation_embeddings(r_ind)
        r_inv = self.inverse_relation_embeddings(r_ind)

        # Compute CP scores for triple, and inverse triple
        score = torch.sum(hh[None, :, :] * r[:, None, :] * tt[:, None, :], dim=-1)
        inverse_score = torch.sum(ht[:, None, :] * r_inv[:, None, :] * th[None, :, :], dim=-1)

        # Final score is average
        scores = 0.5 * (score + inverse_score)

        # Note: In the code in their repository, the score is clamped to [-20, 20].
        #       That is not mentioned in the paper, so it is omitted here.

        return scores
