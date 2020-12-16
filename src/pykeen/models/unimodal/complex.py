# -*- coding: utf-8 -*-

"""Implementation of the ComplEx model."""

from typing import Any, ClassVar, Mapping, Optional, Type

import torch
import torch.nn as nn

from ..base import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import Loss, SoftplusLoss
from ...nn import EmbeddingSpecification
from ...nn.modules import ComplExInteraction
from ...regularizers import LpRegularizer, Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'ComplEx',
]


class ComplEx(ERModel):
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
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )
    #: The default loss function class
    loss_default: Type[Loss] = SoftplusLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = dict(reduction='mean')
    #: The regularizer used by [trouillon2016]_ for ComplEx.
    regularizer_default: Type[Regularizer] = LpRegularizer
    #: The LP settings used by [trouillon2016]_ for ComplEx.
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=0.01,
        p=2.0,
        normalize=True,
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 200,
        loss: Optional[Loss] = None,
        regularizer: Optional[Regularizer] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        embedding_specification: Optional[EmbeddingSpecification] = None,
        relation_embedding_specification: Optional[EmbeddingSpecification] = None,
    ) -> None:
        """Initialize ComplEx.

        :param triples_factory: TriplesFactory
            The triple factory connected to the model.
        :param embedding_dim:
            The embedding dimensionality of the entity embeddings.
        :param loss:
            The loss to use. Defaults to :data:`loss_default`.
        :param regularizer:
            The regularizer to use. Defaults to :data:`regularizer_default`.
        :param preferred_device:
            The default device where to model is located.
        :param random_seed:
            An optional random seed to set before the initialization of weights.
        """
        if regularizer is None:
            regularizer = self._instantiate_default_regularizer()
        # initialize with entity and relation embeddings with standard normal distribution, cf.
        # https://github.com/ttrouill/complex/blob/dc4eb93408d9a5288c986695b58488ac80b1cc17/efe/models.py#L481-L487
        if embedding_specification is None:
            embedding_specification = EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=nn.init.normal_,
                regularizer=regularizer,
                dtype=torch.complex64,
            )
        if relation_embedding_specification is None:
            relation_embedding_specification = EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=nn.init.normal_,
                regularizer=regularizer,
                dtype=torch.complex64,
            )
        super().__init__(
            triples_factory=triples_factory,
            interaction=ComplExInteraction(),
            entity_representations=embedding_specification,
            relation_representations=relation_embedding_specification,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
