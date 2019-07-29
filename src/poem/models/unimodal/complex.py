# -*- coding: utf-8 -*-

"""Implementation of the Complex model based on the open world assumption (OWA)."""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from poem.customized_loss_functions.softplus_loss import SoftplusLoss
from poem.instance_creation_factories.triples_factory import TriplesFactory
from ..base import BaseModule
from ...typing import OptionalLoss


def _compute_regularization_term(
        h: torch.tensor,
        r: torch.tensor,
        t: torch.tensor,
) -> torch.tensor:
    return (torch.mean(h ** 2) + torch.mean(r ** 2) + torch.mean(t ** 2)) / 3.


class ComplEx(BaseModule):
    """An implementation of ComplEx [trouillon2016]_."""

    def __init__(
            self,
            triples_factory: TriplesFactory,
            entity_embeddings: Optional[nn.Embedding] = None,
            relation_embeddings: Optional[nn.Embedding] = None,
            embedding_dim: int = 200,
            neg_label: float = -1.,
            regularization_factor: float = 0.01,
            criterion: OptionalLoss = None,
            preferred_device: Optional[str] = None,
            random_seed: Optional[int] = None,
    ):
        if criterion is None:
            criterion = SoftplusLoss(reduction='mean')

        super(ComplEx, self).__init__(
            triples_factory=triples_factory,
            embedding_dim=2 * embedding_dim,  # complex embeddings
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        self.neg_label = neg_label
        self.regularization_factor = torch.tensor([regularization_factor], requires_grad=False)
        self.current_regularization_term = None
        self.criterion = criterion

        # The embeddings are first initialized when calling the get_grad_params function
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

        if None in [
            self.entity_embeddings,
            self.relation_embeddings,
        ]:
            self._init_embeddings()

    def _init_embeddings(self):
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        xavier_normal_(self.entity_embeddings.weight.data)
        xavier_normal_(self.relation_embeddings.weight.data)

    def compute_label_loss(self, predictions: torch.Tensor, labels: torch.Tensor):
        loss = super(ComplEx, self)._compute_label_loss(predictions=predictions, labels=labels)
        loss += self.regularization_factor * self.current_regularization_term
        return loss

    def forward_owa(
            self,
            batch: torch.tensor,
    ) -> torch.tensor:
        h = self.entity_embeddings(batch[:, 0])
        r = self.relation_embeddings(batch[:, 1])
        t = self.entity_embeddings(batch[:, 2])

        # Regularization term
        self.current_regularization_term = _compute_regularization_term(h, r, t)

        # ComplEx space bilinear product (equivalent to HolE)
        # *: Elementwise multiplication
        i = self.embedding_dim // 2
        re_re_re = h[:, :i] * r[:, :i] * t[:, :i]
        re_im_im = h[:, :i] * r[:, i:] * t[:, i:]
        im_re_im = h[:, i:] * r[:, :i] * t[:, i:]
        im_im_re = h[:, i:] * r[:, i:] * t[:, :i]
        scores = torch.sum(re_re_re + re_im_im + im_re_im + im_im_re, dim=-1, keepdim=True)

        return scores

    def forward_cwa(
            self,
            batch: torch.tensor,
    ) -> torch.tensor:
        h = self.entity_embeddings(batch[:, 0])
        r = self.relation_embeddings(batch[:, 1])
        t = self.entity_embeddings.weight

        # Regularization term
        self.current_regularization_term = _compute_regularization_term(h, r, t)

        # ComplEx space bilinear product (equivalent to HolE)
        # *: Elementwise multiplication
        i = self.embedding_dim // 2
        re_re_re = h[:, None, :i] * r[:, None, :i] * t[None, :, :i]
        re_im_im = h[:, None, :i] * r[:, None, i:] * t[None, :, i:]
        im_re_im = h[:, None, i:] * r[:, None, :i] * t[None, :, i:]
        im_im_re = h[:, None, i:] * r[:, None, i:] * t[None, :, :i]
        scores = torch.sum(re_re_re + re_im_im + im_re_im + im_im_re, dim=-1)

        return scores

    def forward_inverse_cwa(
            self,
            batch: torch.tensor,
    ) -> torch.tensor:
        h = self.entity_embeddings.weight
        r = self.relation_embeddings(batch[:, 0])
        t = self.entity_embeddings(batch[:, 1])

        # Regularization term
        self.current_regularization_term = _compute_regularization_term(h, r, t)

        # ComplEx space bilinear product (equivalent to HolE)
        # *: Elementwise multiplication
        i = self.embedding_dim // 2
        re_re_re = h[None, :, :i] * r[:, None, :i] * t[:, None, :i]
        re_im_im = h[None, :, :i] * r[:, None, i:] * t[:, None, i:]
        im_re_im = h[None, :, i:] * r[:, None, :i] * t[:, None, i:]
        im_im_re = h[None, :, i:] * r[:, None, i:] * t[:, None, :i]
        scores = torch.sum(re_re_re + re_im_im + im_re_im + im_im_re, dim=-1)

        return scores
