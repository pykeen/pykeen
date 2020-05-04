# -*- coding: utf-8 -*-

"""Implementation of RESCAL."""

from typing import Optional

import torch

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss
from ...regularizers import LpRegularizer, Regularizer
from ...triples import TriplesFactory

__all__ = [
    'RESCAL',
]


class RESCAL(EntityRelationEmbeddingModel):
    """An implementation of RESCAL from [nickel2011]_.

    This model represents relations as matrices and models interactions between latent features.

    .. seealso::

       - OpenKE `implementation of RESCAL <https://github.com/thunlp/OpenKE/blob/master/models/RESCAL.py>`_
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
    )
    #: The regularizer used by [nickel2011]_ for for RESCAL
    #: According to https://github.com/mnick/rescal.py/blob/master/examples/kinships.py
    #: a normalized weight of 10 is used.
    regularizer_default = LpRegularizer
    #: The LP settings used by [nickel2011]_ for for RESCAL
    regularizer_default_kwargs = dict(
        weight=10,
        p=2.,
        normalize=True,
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        loss: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            relation_dim=embedding_dim ** 2,  # d x d matrices
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
        # Finalize initialization
        self.reset_parameters_()

    def _reset_parameters_(self):  # noqa: D102
        self.entity_embeddings.reset_parameters()
        self.relation_embeddings.reset_parameters()

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        # shape: (b, d)
        h = self.entity_embeddings(hrt_batch[:, 0]).view(-1, 1, self.embedding_dim)
        # shape: (b, d, d)
        r = self.relation_embeddings(hrt_batch[:, 1]).view(-1, self.embedding_dim, self.embedding_dim)
        # shape: (b, d)
        t = self.entity_embeddings(hrt_batch[:, 2]).view(-1, self.embedding_dim, 1)

        # Compute scores
        scores = h @ r @ t

        # Regularization
        self.regularize_if_necessary(h, r, t)

        return scores[:, :, 0]

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(hr_batch[:, 0]).view(-1, 1, self.embedding_dim)
        r = self.relation_embeddings(hr_batch[:, 1]).view(-1, self.embedding_dim, self.embedding_dim)
        t = self.entity_embeddings.weight.transpose(0, 1).view(1, self.embedding_dim, self.num_entities)

        # Compute scores
        scores = h @ r @ t

        # Regularization
        self.regularize_if_necessary(h, r, t)

        return scores[:, 0, :]

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        """Forward pass using left side (head) prediction."""
        # Get embeddings
        h = self.entity_embeddings.weight.view(1, self.num_entities, self.embedding_dim)
        r = self.relation_embeddings(rt_batch[:, 0]).view(-1, self.embedding_dim, self.embedding_dim)
        t = self.entity_embeddings(rt_batch[:, 1]).view(-1, self.embedding_dim, 1)

        # Compute scores
        scores = h @ r @ t

        # Regularization
        self.regularize_if_necessary(h, r, t)

        return scores[:, :, 0]
