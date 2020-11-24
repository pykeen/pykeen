# -*- coding: utf-8 -*-

"""Implementation of RESCAL."""

from typing import Optional

import torch

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss
from ...regularizers import LpRegularizer, Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'RESCAL',
]


class RESCAL(EntityRelationEmbeddingModel):
    r"""An implementation of RESCAL from [nickel2011]_.

    This model represents relations as matrices and models interactions between latent features.

    RESCAL is a bilinear model that models entities as vectors and relations as matrices.
    The relation matrices $\textbf{W}_{r} \in \mathbb{R}^{d \times d}$ contain weights $w_{i,j}$ that
    capture the amount of interaction between the $i$-th latent factor of $\textbf{e}_h \in \mathbb{R}^{d}$ and the
    $j$-th latent factor of $\textbf{e}_t \in \mathbb{R}^{d}$.

    Thus, the plausibility score of $(h,r,t) \in \mathbb{K}$ is given by:

    .. math::

        f(h,r,t) = \textbf{e}_h^{T} \textbf{W}_{r} \textbf{e}_t = \sum_{i=1}^{d}\sum_{j=1}^{d} w_{ij}^{(r)}
        (\textbf{e}_h)_{i} (\textbf{e}_t)_{j}
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
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        r"""Initialize RESCAL.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.

        .. seealso::

            - OpenKE `implementation of RESCAL <https://github.com/thunlp/OpenKE/blob/master/models/RESCAL.py>`_
        """
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

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        # shape: (b, d)
        h = self.entity_embeddings(indices=hrt_batch[:, 0]).view(-1, 1, self.embedding_dim)
        # shape: (b, d, d)
        r = self.relation_embeddings(indices=hrt_batch[:, 1]).view(-1, self.embedding_dim, self.embedding_dim)
        # shape: (b, d)
        t = self.entity_embeddings(indices=hrt_batch[:, 2]).view(-1, self.embedding_dim, 1)

        # Compute scores
        scores = h @ r @ t

        # Regularization
        self.regularize_if_necessary(h, r, t)

        return scores[:, :, 0]

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(indices=hr_batch[:, 0]).view(-1, 1, self.embedding_dim)
        r = self.relation_embeddings(indices=hr_batch[:, 1]).view(-1, self.embedding_dim, self.embedding_dim)
        t = self.entity_embeddings(indices=None).transpose(0, 1).view(1, self.embedding_dim, self.num_entities)

        # Compute scores
        scores = h @ r @ t

        # Regularization
        self.regularize_if_necessary(h, r, t)

        return scores[:, 0, :]

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        """Forward pass using left side (head) prediction."""
        # Get embeddings
        h = self.entity_embeddings(indices=None).view(1, self.num_entities, self.embedding_dim)
        r = self.relation_embeddings(indices=rt_batch[:, 0]).view(-1, self.embedding_dim, self.embedding_dim)
        t = self.entity_embeddings(indices=rt_batch[:, 1]).view(-1, self.embedding_dim, 1)

        # Compute scores
        scores = h @ r @ t

        # Regularization
        self.regularize_if_necessary(h, r, t)

        return scores[:, :, 0]
