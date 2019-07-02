# -*- coding: utf-8 -*-

"""Implementation of the HolE model."""

import logging

import numpy as np
import torch
import torch.autograd
from torch import nn

from poem.models.base import BaseModule
from ...constants import GPU, HOL_E_NAME, SCORING_FUNCTION_NORM
from ...utils import slice_triples

__all__ = [
    'HolE',
]

log = logging.getLogger(__name__)


def circular_correlation(
        a: torch.tensor,
        b: torch.tensor
) -> torch.tensor:
    r"""
    Computes the batched circular correlation between a and b using FFT.

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
    """An implementation of HolE [nickel2016].

     This model uses circular correlation to compose subject and object embeddings to afterwards compute the inner product with a relation embedding.

    .. [nickel2016] Holographic Embeddings of Knowledge Graphs
                    M. Nickel and L. Rosasco and T. Poggio
                    <https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828>
                    AAAI 2016.

    .. seealso::

       - Authors' implementation: https://github.com/mnick/holographic-embeddings
       - Implementation in scikit-kge: https://github.com/mnick/scikit-kge
       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py
    """

    model_name = HOL_E_NAME
    hyper_params = BaseModule.hyper_params + (SCORING_FUNCTION_NORM,)

    def __init__(
            self,
            num_entities,
            num_relations,
            embedding_dim=200,
            criterion=nn.MarginRankingLoss(margin=1., reduction='mean'),
            preferred_device=GPU,
    ) -> None:
        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            criterion=criterion,
            embedding_dim=embedding_dim,
            preferred_device=preferred_device,
        )

        # Embeddings
        self.relation_embeddings = nn.Embedding(num_relations, self.embedding_dim)

        self._initialize()

    def _initialize(self):
        # Initialisation, cf. https://github.com/mnick/scikit-kge/blob/master/skge/param.py#L18-L27
        entity_embeddings_init_bound = 6 / np.sqrt(
            self.entity_embeddings.num_embeddings + self.entity_embeddings.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-entity_embeddings_init_bound,
            b=+entity_embeddings_init_bound,
        )
        relation_embeddings_init_bound = 6 / np.sqrt(
            self.relation_embeddings.num_embeddings + self.relation_embeddings.embedding_dim)
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=-relation_embeddings_init_bound,
            b=+relation_embeddings_init_bound,
        )

    def apply_forward_constraints(self):
        """."""
        # Do not compute gradients for forward constraints
        with torch.no_grad():
            # Ensure norm of entity embeddings is at most 1
            norms = torch.norm(self.entity_embeddings.weight, p=2, dim=1, keepdim=True)
            self.entity_embeddings.weight /= torch.max(norms, torch.ones(size=(), device=self.device))

    def _score_triples(self, triples):
        heads, relations, tails = slice_triples(triples)

        # Get embeddings
        head_embeddings = self._get_embeddings(heads, embedding_module=self.entity_embeddings,
                                               embedding_dim=self.embedding_dim)
        tail_embeddings = self._get_embeddings(tails, embedding_module=self.entity_embeddings,
                                               embedding_dim=self.embedding_dim)
        relation_embeddings = self._get_embeddings(relations, embedding_module=self.relation_embeddings,
                                                   embedding_dim=self.embedding_dim)

        # Circular correlation of entity embeddings
        composite = circular_correlation(a=head_embeddings, b=tail_embeddings)

        # inner product with relation embedding
        scores = torch.sum(relation_embeddings * composite, dim=1)

        return scores
