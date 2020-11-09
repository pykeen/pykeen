# -*- coding: utf-8 -*-

"""Implementation of the HolE model."""

from typing import Optional

import torch
import torch.autograd

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss
from ...nn.init import xavier_uniform_
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint
from ...utils import clamp_norm

__all__ = [
    'HolE',
]


class HolE(EntityRelationEmbeddingModel):
    r"""An implementation of HolE [nickel2016]_.

    Holographic embeddings (HolE) make use of the circular correlation operator to compute interactions between
    latent features of entities and relations:

    .. math::

        f(h,r,t) = \sigma(\textbf{r}^{T}(\textbf{h} \star \textbf{t}))

    where the circular correlation $\star: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}^d$ is defined as:

    .. math::

        [\textbf{a} \star \textbf{b}]_i = \sum_{k=0}^{d-1} \textbf{a}_{k} * \textbf{b}_{(i+k)\ mod \ d}

    By using the correlation operator each component $[\textbf{h} \star \textbf{t}]_i$ represents a sum over a
    fixed partition over pairwise interactions. This enables the model to put semantic similar interactions into the
    same partition and share weights through $\textbf{r}$. Similarly irrelevant interactions of features could also
    be placed into the same partition which could be assigned a small weight in $\textbf{r}$.

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
        preferred_device: DeviceHint = None,
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
            # Initialisation, cf. https://github.com/mnick/scikit-kge/blob/master/skge/param.py#L18-L27
            entity_initializer=xavier_uniform_,
            relation_initializer=xavier_uniform_,
            entity_constrainer=clamp_norm,
            entity_constrainer_kwargs=dict(maxnorm=1., p=2, dim=-1),
        )

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
        h = self.entity_embeddings(indices=hrt_batch[:, 0]).unsqueeze(dim=1)
        r = self.relation_embeddings(indices=hrt_batch[:, 1]).unsqueeze(dim=1)
        t = self.entity_embeddings(indices=hrt_batch[:, 2]).unsqueeze(dim=1)

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        scores = self.interaction_function(h=h, r=r, t=t).view(-1, 1)

        return scores

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(indices=hr_batch[:, 0]).unsqueeze(dim=1)
        r = self.relation_embeddings(indices=hr_batch[:, 1]).unsqueeze(dim=1)
        t = self.entity_embeddings(indices=None).unsqueeze(dim=0)

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        scores = self.interaction_function(h=h, r=r, t=t)

        return scores

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(indices=None).unsqueeze(dim=0)
        r = self.relation_embeddings(indices=rt_batch[:, 0]).unsqueeze(dim=1)
        t = self.entity_embeddings(indices=rt_batch[:, 1]).unsqueeze(dim=1)

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        scores = self.interaction_function(h=h, r=r, t=t)

        return scores
