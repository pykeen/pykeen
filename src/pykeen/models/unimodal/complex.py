# -*- coding: utf-8 -*-

"""Implementation of the ComplEx model."""

from typing import Optional

import torch
import torch.nn as nn

from ..base import InteractionFunction, SimpleVectorEntityRelationEmbeddingModel, normalize_for_einsum
from ...losses import Loss, SoftplusLoss
from ...regularizers import LpRegularizer, Regularizer
from ...triples import TriplesFactory
from ...utils import split_complex

__all__ = [
    'ComplEx',
    'ComplexInteractionFunction',
]


class ComplexInteractionFunction(InteractionFunction):
    """Interaction function of Complex."""

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        batch_size = max(h.shape[0], r.shape[0], t.shape[0])
        h_term, h = normalize_for_einsum(x=h, batch_size=batch_size, symbol='h')
        r_term, r = normalize_for_einsum(x=r, batch_size=batch_size, symbol='r')
        t_term, t = normalize_for_einsum(x=t, batch_size=batch_size, symbol='t')
        (h_re, h_im), (r_re, r_im), (t_re, t_im) = [split_complex(x=x) for x in (h, r, t)]
        return sum(
            torch.einsum(f'{h_term},{r_term},{t_term}->bhrt', hh, rr, tt)
            for hh, rr, tt in [
                (h_re, r_re, t_re),
                (h_re, r_im, t_im),
                (h_im, r_re, t_im),
                (h_im, r_im, t_re),
            ]
        )


class ComplEx(SimpleVectorEntityRelationEmbeddingModel):
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
        preferred_device: Optional[str] = None,
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
        interaction_function = ComplexInteractionFunction.from_model(self)

        super().__init__(
            triples_factory=triples_factory,
            interaction_function=interaction_function,
            embedding_dim=2 * embedding_dim,  # complex embeddings
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

    def _reset_parameters_(self):  # noqa: D102
        # initialize with entity and relation embeddings with standard normal distribution, cf.
        # https://github.com/ttrouill/complex/blob/dc4eb93408d9a5288c986695b58488ac80b1cc17/efe/models.py#L481-L487
        nn.init.normal_(tensor=self.entity_embeddings.weight, mean=0., std=1.)
        nn.init.normal_(tensor=self.relation_embeddings.weight, mean=0., std=1.)
