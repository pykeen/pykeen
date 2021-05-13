# -*- coding: utf-8 -*-

"""Implementation of the ComplEx model."""

from typing import Any, ClassVar, Mapping, Optional, Type

import torch
from torch.nn.init import normal_

from ..base import EntityRelationEmbeddingModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import Loss, SoftplusLoss
from ...nn.emb import EmbeddingSpecification
from ...regularizers import LpRegularizer, Regularizer
from ...typing import Hint, Initializer
from ...utils import split_complex

__all__ = [
    'ComplEx',
]


class ComplEx(EntityRelationEmbeddingModel):
    r"""An implementation of ComplEx [trouillon2016]_.

    ComplEx is an extension of :class:`pykeen.models.DistMult` that uses complex valued representations for the
    entities and relations. Entities and relations are represented as vectors
    $\textbf{e}_i, \textbf{r}_i \in \mathbb{C}^d$, and the plausibility score is computed using the
    Hadamard product:

    .. math::

        f(h,r,t) =  Re(\mathbf{e}_h\odot\mathbf{r}_r\odot\mathbf{e}_t)

    Which expands to:

    .. math::

        f(h,r,t) = \left\langle Re(\mathbf{e}_h),Re(\mathbf{r}_r),Re(\mathbf{e}_t)\right\rangle
        + \left\langle Im(\mathbf{e}_h),Re(\mathbf{r}_r),Im(\mathbf{e}_t)\right\rangle
        + \left\langle Re(\mathbf{e}_h),Re(\mathbf{r}_r),Im(\mathbf{e}_t)\right\rangle
        - \left\langle Im(\mathbf{e}_h),Im(\mathbf{r}_r),Im(\mathbf{e}_t)\right\rangle

    where $Re(\textbf{x})$ and $Im(\textbf{x})$ denote the real and imaginary parts of the complex valued vector
    $\textbf{x}$. Because the Hadamard product is not commutative in the complex space, ComplEx can model
    anti-symmetric relations in contrast to DistMult.

    .. seealso ::

        Official implementation: https://github.com/ttrouill/complex/
    ---
    citation:
        author: Trouillon
        year: 2016
        link: https://arxiv.org/abs/1606.06357
        github: ttrouill/complex
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )
    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = SoftplusLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = dict(reduction='mean')
    #: The regularizer used by [trouillon2016]_ for ComplEx.
    regularizer_default: ClassVar[Type[Regularizer]] = LpRegularizer
    #: The LP settings used by [trouillon2016]_ for ComplEx.
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=0.01,
        p=2.0,
        normalize=True,
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        # initialize with entity and relation embeddings with standard normal distribution, cf.
        # https://github.com/ttrouill/complex/blob/dc4eb93408d9a5288c986695b58488ac80b1cc17/efe/models.py#L481-L487
        entity_initializer: Hint[Initializer] = normal_,
        relation_initializer: Hint[Initializer] = normal_,
        **kwargs,
    ) -> None:
        """Initialize ComplEx.

        :param embedding_dim:
            The embedding dimensionality of the entity embeddings.
        :param entity_initializer: Entity initializer function. Defaults to :func:`torch.nn.init.normal_`
        :param relation_initializer: Relation initializer function. Defaults to :func:`torch.nn.init.normal_`
        :param kwargs:
            Remaining keyword arguments to forward to :class:`pykeen.models.EntityRelationEmbeddingModel`
        """
        super().__init__(
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=entity_initializer,
                dtype=torch.cfloat,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=relation_initializer,
                dtype=torch.cfloat,
            ),
            **kwargs,
        )

    @staticmethod
    def interaction_function(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function of ComplEx for given embeddings.

        The embeddings have to be in a broadcastable shape.

        :param h:
            Head embeddings.
        :param r:
            Relation embeddings.
        :param t:
            Tail embeddings.

        :return: shape: (...)
            The scores.
        """
        # split into real and imaginary part
        (h_re, h_im), (r_re, r_im), (t_re, t_im) = [split_complex(x=x) for x in (h, r, t)]

        # ComplEx space bilinear product
        # *: Elementwise multiplication
        return sum(
            (hh * rr * tt).sum(dim=-1)
            for hh, rr, tt in [
                (h_re, r_re, t_re),
                (h_re, r_im, t_im),
                (h_im, r_re, t_im),
                (-h_im, r_im, t_re),
            ]
        )

    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
    ) -> torch.FloatTensor:
        """Unified score function."""
        # get embeddings
        h = self.entity_embeddings.get_in_canonical_shape(indices=h_indices)
        r = self.relation_embeddings.get_in_canonical_shape(indices=r_indices)
        t = self.entity_embeddings.get_in_canonical_shape(indices=t_indices)

        # Regularization
        self.regularize_if_necessary(h, r, t)

        # Compute scores
        return self.interaction_function(h=h, r=r, t=t)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self(h_indices=hrt_batch[:, 0], r_indices=hrt_batch[:, 1], t_indices=hrt_batch[:, 2]).view(-1, 1)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self(h_indices=hr_batch[:, 0], r_indices=hr_batch[:, 1], t_indices=None)

    def score_r(self, ht_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self(h_indices=ht_batch[:, 0], r_indices=None, t_indices=ht_batch[:, 1])

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self(h_indices=None, r_indices=rt_batch[:, 0], t_indices=rt_batch[:, 1])
