# -*- coding: utf-8 -*-

"""Regularization in PyKEEN."""

from typing import Any, ClassVar, Mapping

import torch
from torch import nn
from torch.nn import functional

from ...regularizers import Regularizer

__all__ = [
    'TransHRegularizer',
]


class TransHRegularizer(Regularizer):
    """A regularizer for the soft constraints in TransH."""

    #: The default strategy for optimizing the TransH regularizer's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        weight=dict(type=float, low=0.01, high=1.0, scale='log'),
    )

    def __init__(
        self,
        entity_embeddings: nn.Parameter,
        normal_vector_embeddings: nn.Parameter,
        relation_embeddings: nn.Parameter,
        weight: float = 0.05,
        epsilon: float = 1e-5,
    ):
        # The regularization in TransH enforces the defined soft constraints that should computed only for every batch.
        # Therefore, apply_only_once is always set to True.
        super().__init__(weight=weight, apply_only_once=True, parameters=[])
        self.normal_vector_embeddings = normal_vector_embeddings
        self.relation_embeddings = relation_embeddings
        self.entity_embeddings = entity_embeddings
        self.epsilon = epsilon

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        raise NotImplementedError('TransH regularizer is order-sensitive!')

    def pop_regularization_term(self, *tensors: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        # Entity soft constraint
        self.regularization_term += torch.sum(functional.relu(torch.norm(self.entity_embeddings, dim=-1) ** 2 - 1.0))

        # Orthogonality soft constraint
        d_r_n = functional.normalize(self.relation_embeddings, dim=-1)
        self.regularization_term += torch.sum(
            functional.relu(torch.sum((self.normal_vector_embeddings * d_r_n) ** 2, dim=-1) - self.epsilon),
        )
        return super().pop_regularization_term()
