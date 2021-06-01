# -*- coding: utf-8 -*-

"""Implementation of the HolE model."""

from typing import Any, ClassVar, Mapping, Optional

import torch

from ..base import EntityRelationEmbeddingModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...moves import irfft, rfft
from ...nn.emb import EmbeddingSpecification
from ...nn.init import xavier_uniform_
from ...typing import Constrainer, Hint, Initializer
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
    ---
    citation:
        author: Nickel
        year: 2016
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828
        github: mnick/holographic-embeddings
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    #: The default settings for the entity constrainer
    entity_constrainer_default_kwargs = dict(maxnorm=1., p=2, dim=-1)

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = clamp_norm,  # type: ignore
        entity_constrainer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Constrainer] = xavier_uniform_,
        **kwargs,
    ) -> None:
        """Initialize the model."""
        super().__init__(
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                # Initialisation, cf. https://github.com/mnick/scikit-kge/blob/master/skge/param.py#L18-L27
                initializer=entity_initializer,
                constrainer=entity_constrainer,
                constrainer_kwargs=entity_constrainer_kwargs or self.entity_constrainer_default_kwargs,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=relation_initializer,
            ),
            **kwargs,
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
        a_fft = rfft(h, dim=-1)
        b_fft = rfft(t, dim=-1)

        # complex conjugate, a_fft.shape = (batch_size, num_entities, d', 2)
        # compatibility: new style fft returns complex tensor
        if a_fft.ndimension() > 3:
            a_fft[:, :, :, 1] *= -1
        else:
            a_fft = torch.conj(a_fft)

        # Hadamard product in frequency domain
        p_fft = a_fft * b_fft

        # inverse real FFT, shape: (batch_size, num_entities, d)
        composite = irfft(p_fft, dim=-1, n=h.shape[-1])

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
