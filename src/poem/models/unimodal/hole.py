# -*- coding: utf-8 -*-

"""Implementation of the HolE model."""

import logging
from typing import Optional

import numpy as np
import torch
import torch.autograd
from torch import nn

from poem.instance_creation_factories.triples_factory import TriplesFactory
from ..base import BaseModule
from ...constants import SCORING_FUNCTION_NORM
from ...typing import OptionalLoss
from ...utils import slice_triples

__all__ = [
    'HolE',
]

log = logging.getLogger(__name__)


def circular_correlation(
        a: torch.tensor,
        b: torch.tensor,
) -> torch.tensor:
    r"""Compute the batched circular correlation between a and b using FFT.

    `a \ast b = \mathcal{F}^{-1}(\overline{\mathcal{F}(a)} \odot \mathcal{F}(b))`

    :param a: torch.tensor, shape: (batch_size, dim)
    :param b: torch.tensor, shape: (batch_size, dim)

    :return: torch.tensor, shape: (batch_size, dim)
    """
    # TODO: Explicitly exploit symmetry and set onesided=True
    a_fft = torch.rfft(a, signal_ndim=1, onesided=False)
    b_fft = torch.rfft(b, signal_ndim=1, onesided=False)

    # complex conjugate
    a_fft[:, :, 1] *= -1

    # Hadamard product in frequency domain
    p_fft = a_fft * b_fft

    # inverse real FFT
    corr = torch.irfft(p_fft, signal_ndim=1, onesided=False, signal_sizes=a.shape[1:])

    return corr


class HolE(BaseModule):
    """An implementation of HolE [nickel2016]_.

     This model uses circular correlation to compose subject and object embeddings to afterwards compute the inner
     product with a relation embedding.

    .. seealso::

       - `author's implementation of HolE <https://github.com/mnick/holographic-embeddings>`_
       - `scikit-kge implementation of HolE <https://github.com/mnick/scikit-kge>`_
       - OpenKE `implementation of HolE <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py>`_
    """

    hyper_params = BaseModule.hyper_params + (SCORING_FUNCTION_NORM,)
    entity_embedding_max_norm = 1

    def __init__(
            self,
            triples_factory: TriplesFactory,
            entity_embeddings: Optional[nn.Embedding] = None,
            relation_embeddings: Optional[nn.Embedding] = None,
            embedding_dim: int = 200,
            criterion: OptionalLoss = None,
            preferred_device: Optional[str] = None,
            random_seed: Optional[int] = None,
    ) -> None:
        if criterion is None:
            criterion = nn.MarginRankingLoss(margin=1., reduction='mean')

        super().__init__(
            triples_factory=triples_factory,
            criterion=criterion,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        self.relation_embeddings = relation_embeddings

        if None in [self.entity_embeddings, self.relation_embeddings]:
            self._initialize()

    def _initialize(self):
        """."""
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        # Initialisation, cf. https://github.com/mnick/scikit-kge/blob/master/skge/param.py#L18-L27
        entity_embeddings_init_bound = 6 / np.sqrt(
            self.entity_embeddings.num_embeddings + self.entity_embeddings.embedding_dim,
        )
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-entity_embeddings_init_bound,
            b=+entity_embeddings_init_bound,
        )
        relation_embeddings_init_bound = 6 / np.sqrt(
            self.relation_embeddings.num_embeddings + self.relation_embeddings.embedding_dim,
        )
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=-relation_embeddings_init_bound,
            b=+relation_embeddings_init_bound,
        )

    def _score_triples(self, triples):
        heads, relations, tails = slice_triples(triples)

        # Get embeddings
        head_embeddings = self._get_embeddings(
            heads, embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim,
        )
        tail_embeddings = self._get_embeddings(
            tails, embedding_module=self.entity_embeddings,
            embedding_dim=self.embedding_dim,
        )
        relation_embeddings = self._get_embeddings(
            relations, embedding_module=self.relation_embeddings,
            embedding_dim=self.embedding_dim,
        )

        # Circular correlation of entity embeddings
        composite = circular_correlation(a=head_embeddings, b=tail_embeddings)

        # inner product with relation embedding
        scores = torch.sum(relation_embeddings * composite, dim=1)

        return scores
