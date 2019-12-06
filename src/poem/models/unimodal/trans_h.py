# -*- coding: utf-8 -*-

"""An implementation of TransH."""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional

from ..base import BaseModule
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory

__all__ = [
    'TransH',
]


class TransHRegularizer(Regularizer):
    """Regularizer for TransH's soft constraints."""

    def __init__(
        self,
        device: torch.device,
        weight: float,
        epsilon: float,
    ):
        super().__init__(device=device, weight=weight)
        self.epsilon = epsilon

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        raise NotImplementedError('TransH regularizer is order-sensitive!')

    def update(self, *tensors: torch.FloatTensor) -> None:  # noqa: D102
        if len(tensors) != 4:
            raise KeyError('Expects exactly four tensors')

        h, t, w_r, d_r = tensors

        # Entity soft constraint
        self.regularization_term += torch.sum(functional.relu(torch.norm(h, dim=-1)) ** 2 - 1.0)
        self.regularization_term += torch.sum(functional.relu(torch.norm(t, dim=-1)) ** 2 - 1.0)

        # Orthogonality soft constraint
        d_r_n = functional.normalize(d_r, dim=-1)
        self.regularization_term += torch.sum(functional.relu(torch.sum((w_r * d_r_n) ** 2, dim=-1) - self.epsilon))


class TransH(BaseModule):
    """An implementation of TransH [wang2014]_.

    This model extends TransE by applying the translation from head to tail entity in a relational-specific hyperplane.

    .. seealso::

       - OpenKE `implementation of TransH <https://github.com/thunlp/OpenKE/blob/master/models/TransH.py>`_
    """

    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=300, q=50),
        regularization_weight=dict(type=float, low=0.001, high=1., scale='log'),
        epsilon=dict(type=float, low=0.05, high=1.5, scale='log'),
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    #: The custom regularizer used by [wang2014]_ for TransH
    regularizer_default = TransHRegularizer
    #: The settings used by [wang2014]_ for TransH
    regularizer_default_kwargs = dict(
        weight=0.05,
        epsilon=1e-5,
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        entity_embeddings: Optional[nn.Embedding] = None,
        relation_embeddings: Optional[nn.Embedding] = None,
        normal_vector_embeddings: Optional[nn.Embedding] = None,
        scoring_fct_norm: int = 1,
        criterion: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        self.scoring_fct_norm = scoring_fct_norm
        self.relation_embeddings = relation_embeddings
        self.normal_vector_embeddings = normal_vector_embeddings

        # Finalize initialization
        self._init_weights_on_device()

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
        self.normal_vector_embeddings = None
        return self

    def post_parameter_update(self) -> None:  # noqa: D102
        # Make sure to call super first
        super().post_parameter_update()

        # Normalise the normal vectors by their l2 norms
        functional.normalize(
            self.normal_vector_embeddings.weight.data,
            out=self.normal_vector_embeddings.weight.data,
        )

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hrt_batch[:, 0])
        d_r = self.relation_embeddings(hrt_batch[:, 1])
        w_r = self.normal_vector_embeddings(hrt_batch[:, 1])
        t = self.entity_embeddings(hrt_batch[:, 2])

        # Project to hyperplane
        ph = h - torch.sum(w_r * h, dim=-1, keepdim=True) * w_r
        pt = t - torch.sum(w_r * t, dim=-1, keepdim=True) * w_r

        # Regularization term
        self.regularize_if_necessary(h, t, w_r, d_r)

        return -torch.norm(ph + d_r - pt, p=2, dim=-1, keepdim=True)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hr_batch[:, 0])
        d_r = self.relation_embeddings(hr_batch[:, 1])
        w_r = self.normal_vector_embeddings(hr_batch[:, 1])
        t = self.entity_embeddings.weight

        # Project to hyperplane
        ph = h - torch.sum(w_r * h, dim=-1, keepdim=True) * w_r
        pt = t[None, :, :] - torch.sum(w_r[:, None, :] * t[None, :, :], dim=-1, keepdim=True) * w_r[:, None, :]

        # Regularization term
        self.regularize_if_necessary(h, t, w_r, d_r)

        return -torch.norm(ph[:, None, :] + d_r[:, None, :] - pt, p=2, dim=-1)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings.weight
        rel_id = rt_batch[:, 0]
        d_r = self.relation_embeddings(rel_id)
        w_r = self.normal_vector_embeddings(rel_id)
        t = self.entity_embeddings(rt_batch[:, 1])

        # Project to hyperplane
        ph = h[None, :, :] - torch.sum(w_r[:, None, :] * h[None, :, :], dim=-1, keepdim=True) * w_r[:, None, :]
        pt = t - torch.sum(w_r * t, dim=-1, keepdim=True) * w_r

        # Regularization term
        self.regularize_if_necessary(h, t, w_r, d_r)

        return -torch.norm(ph + d_r[:, None, :] - pt[:, None, :], p=2, dim=-1)
