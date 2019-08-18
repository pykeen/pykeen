# -*- coding: utf-8 -*-

"""An implementation of TransH."""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional

from ..base import BaseModule
from ...instance_creation_factories import TriplesFactory
from ...typing import OptionalLoss


class TransH(BaseModule):
    """An implementation of TransH [wang2014]_.

    This model extends TransE by applying the translation from head to tail entity in a relational-specific hyperplane.

    .. seealso::

       - OpenKE `implementation of TransH <https://github.com/thunlp/OpenKE/blob/master/models/TransH.py>`_
    """

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 50,
            entity_embeddings: Optional[nn.Embedding] = None,
            relation_embeddings: Optional[nn.Embedding] = None,
            normal_vector_embeddings: Optional[nn.Embedding] = None,
            scoring_fct_norm: int = 1,
            soft_weight_constraint: float = 0.05,
            epsilon: float = 0.005,
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
        self.regularization_factor = soft_weight_constraint
        self.epsilon = torch.tensor([epsilon], requires_grad=False)

        self.current_regularization_term = None

        self.scoring_fct_norm = scoring_fct_norm
        self.relation_embeddings = relation_embeddings
        self.normal_vector_embeddings = normal_vector_embeddings

        if init:
            self.init_empty_weights_()

    def init_empty_weights_(self):  # noqa: D102
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        if self.normal_vector_embeddings is None:
            self.normal_vector_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        # TODO: Add initialization
        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.relation_embeddings = None
        return self

    def _apply_forward_constraints_if_necessary(self) -> None:
        if not self.forward_constraint_applied:
            # Normalise the normal vectors by their l2 norms
            functional.normalize(
                self.normal_vector_embeddings.weight.data,
                out=self.normal_vector_embeddings.weight.data,
            )

            self.forward_constraint_applied = True

    def _update_regularization_term(self) -> None:
        w_r = self.normal_vector_embeddings.weight
        d_r = self.relation_embeddings.weight
        d_r_n = functional.normalize(d_r, dim=-1)
        ortho_constraint = torch.sum(functional.relu(torch.sum((w_r * d_r_n) ** 2, dim=-1) - self.epsilon))
        entity_constraint = torch.sum(functional.relu(torch.norm(self.entity_embeddings.weight, dim=-1) ** 2 - 1.0))
        self.current_regularization_term = ortho_constraint + entity_constraint

    def forward_owa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Guarantee forward constraints
        self._apply_forward_constraints_if_necessary()

        # Get embeddings
        h = self.entity_embeddings(batch[:, 0])
        d_r = self.relation_embeddings(batch[:, 1])
        w_r = self.normal_vector_embeddings(batch[:, 1])
        t = self.entity_embeddings(batch[:, 2])

        # Project to hyperplane
        ph = h - torch.sum(w_r * h, dim=-1, keepdim=True) * w_r
        pt = t - torch.sum(w_r * t, dim=-1, keepdim=True) * w_r

        # Regularization term
        self._update_regularization_term()

        return -torch.norm(ph + d_r - pt, p=2, dim=-1, keepdim=True)

    def forward_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Guarantee forward constraints
        self._apply_forward_constraints_if_necessary()

        # Get embeddings
        h = self.entity_embeddings(batch[:, 0])
        d_r = self.relation_embeddings(batch[:, 1])
        w_r = self.normal_vector_embeddings(batch[:, 1])
        t = self.entity_embeddings.weight

        # Project to hyperplane
        ph = h - torch.sum(w_r * h, dim=-1, keepdim=True) * w_r
        pt = t[None, :, :] - torch.sum(w_r[:, None, :] * t[None, :, :], dim=-1, keepdim=True) * w_r[:, None, :]

        # Regularization term
        self._update_regularization_term()

        return -torch.norm(ph[:, None, :] + d_r[:, None, :] - pt, p=2, dim=-1)

    def forward_inverse_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Guarantee forward constraints
        self._apply_forward_constraints_if_necessary()

        # Get embeddings
        h = self.entity_embeddings.weight
        d_r = self.relation_embeddings(batch[:, 0])
        w_r = self.normal_vector_embeddings(batch[:, 0])
        t = self.entity_embeddings(batch[:, 0])

        # Project to hyperplane
        ph = h[None, :, :] - torch.sum(w_r[:, None, :] * h[None, :, :], dim=-1, keepdim=True) * w_r[:, None, :]
        pt = t - torch.sum(w_r * t, dim=-1, keepdim=True) * w_r

        # Regularization term
        self._update_regularization_term()

        return -torch.norm(ph + d_r[:, None, :] - pt[:, None, :], p=2, dim=-1)

    def compute_mr_loss(
            self,
            positive_scores: torch.FloatTensor,
            negative_scores: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        loss = super().compute_mr_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        loss += self.regularization_factor * self.current_regularization_term
        return loss

    def compute_label_loss(
            self,
            predictions: torch.FloatTensor,
            labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        loss = super().compute_label_loss(predictions=predictions, labels=labels)
        loss += self.regularization_factor * self.current_regularization_term
        return loss
