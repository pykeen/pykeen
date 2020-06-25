# -*- coding: utf-8 -*-

"""Implementation of the HolE model."""

from typing import Optional

import torch
import torch.autograd

from ..base import EntityRelationEmbeddingModel
from ..init import embedding_xavier_uniform_
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...utils import clamp_norm

__all__ = [
    'HolE',
]


class HolE(EntityRelationEmbeddingModel):
    """An implementation of HolE [nickel2016]_.

     This model uses circular correlation to compose head and tail embeddings to afterwards compute the inner
     product with a relation embedding.

    .. seealso::

       - `author's implementation of HolE <https://github.com/mnick/holographic-embeddings>`_
       - `scikit-kge implementation of HolE <https://github.com/mnick/scikit-kge>`_
       - OpenKE `implementation of HolE <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py>`_
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 200,
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
            loss=loss,
            automatic_memory_optimization=automatic_memory_optimization,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
        # Finalize initialization
        self.reset_parameters_()

    def post_parameter_update(self) -> None:  # noqa: D102
        # Make sure to call super first
        super().post_parameter_update()

        # Normalize entity embeddings
        self.entity_embeddings.weight.data = clamp_norm(x=self.entity_embeddings.weight.data, maxnorm=1., p=2, dim=-1)

    def _reset_parameters_(self):  # noqa: D102
        # Initialisation, cf. https://github.com/mnick/scikit-kge/blob/master/skge/param.py#L18-L27
        embedding_xavier_uniform_(self.entity_embeddings)
        embedding_xavier_uniform_(self.relation_embeddings)

    @staticmethod
    def interaction_function(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function for given embeddings.

        The embeddings have to be in a broadcastable shape.

        :param h: shape: (batch_size, num_entities, d)
            Head embeddings.
        :param r: shape: (batch_size, num_entities, d)
            Relation embeddings.
        :param t: shape: (batch_size, num_entities, d)
            Tail embeddings.

        :return: shape: (batch_size, num_entities)
            The scores.
        """
        # Circular correlation of entity embeddings
        a_fft = torch.rfft(h, signal_ndim=1, onesided=True)
        b_fft = torch.rfft(t, signal_ndim=1, onesided=True)

        # complex conjugate, a_fft.shape = (batch_size, num_entities, d', 2)
        a_fft[:, :, :, 1] *= -1

        # Hadamard product in frequency domain
        p_fft = a_fft * b_fft

        # inverse real FFT, shape: (batch_size, num_entities, d)
        composite = torch.irfft(p_fft, signal_ndim=1, onesided=True, signal_sizes=(h.shape[-1],))

        # inner product with relation embedding
        scores = torch.sum(r * composite, dim=-1, keepdim=False)

        return scores

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(hrt_batch[:, 0]).unsqueeze(dim=1)
        r = self.relation_embeddings(hrt_batch[:, 1]).unsqueeze(dim=1)
        t = self.entity_embeddings(hrt_batch[:, 2]).unsqueeze(dim=1)

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        scores = self.interaction_function(h=h, r=r, t=t).view(-1, 1)

        return scores

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(hr_batch[:, 0]).unsqueeze(dim=1)
        r = self.relation_embeddings(hr_batch[:, 1]).unsqueeze(dim=1)
        t = self.entity_embeddings.weight.unsqueeze(dim=0)

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        scores = self.interaction_function(h=h, r=r, t=t)

        return scores

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings.weight.unsqueeze(dim=0)
        r = self.relation_embeddings(rt_batch[:, 0]).unsqueeze(dim=1)
        t = self.entity_embeddings(rt_batch[:, 1]).unsqueeze(dim=1)

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        scores = self.interaction_function(h=h, r=r, t=t)

        return scores
