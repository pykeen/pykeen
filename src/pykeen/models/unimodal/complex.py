# -*- coding: utf-8 -*-

"""Implementation of the ComplEx model."""

from typing import Optional

import torch
import torch.nn as nn

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss, SoftplusLoss
from ...regularizers import LpRegularizer, Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint
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
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=300, q=50),
    )
    #: The default loss function class
    loss_default = SoftplusLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs = dict(reduction='mean')
    #: The regularizer used by [trouillon2016]_ for ComplEx.
    regularizer_default = LpRegularizer
    #: The LP settings used by [trouillon2016]_ for ComplEx.
    regularizer_default_kwargs = dict(
        weight=0.01,
        p=2.0,
        normalize=True,
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
        """Initialize ComplEx.

        :param triples_factory: TriplesFactory
            The triple factory connected to the model.
        :param embedding_dim:
            The embedding dimensionality of the entity embeddings.
        :param automatic_memory_optimization: bool
            Whether to automatically optimize the sub-batch size during training and batch size during evaluation with
            regards to the hardware at hand.
        :param loss: OptionalLoss (optional)
            The loss to use. Defaults to SoftplusLoss.
        :param preferred_device: str (optional)
            The default device where to model is located.
        :param random_seed: int (optional)
            An optional random seed to set before the initialization of weights.
        :param regularizer: BaseRegularizer
            The regularizer to use.
        """
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=2 * embedding_dim,  # complex embeddings
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            # initialize with entity and relation embeddings with standard normal distribution, cf.
            # https://github.com/ttrouill/complex/blob/dc4eb93408d9a5288c986695b58488ac80b1cc17/efe/models.py#L481-L487
            entity_initializer=nn.init.normal_,
            relation_initializer=nn.init.normal_,
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
                (h_im, r_im, t_re),
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
