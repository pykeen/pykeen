# -*- coding: utf-8 -*-

"""Implementation of the ComplEx model."""

from typing import Any, ClassVar, Mapping, Optional, Type

import torch
from class_resolver.api import HintOrType
from torch.nn.init import normal_

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import Loss, SoftplusLoss
from ...nn.modules import ComplExInteraction
from ...regularizers import LpRegularizer, Regularizer
from ...typing import Hint, Initializer

__all__ = [
    "ComplEx",
]


class ComplEx(ERModel):
    r"""An implementation of ComplEx [trouillon2016]_.

    ComplEx is an extension of :class:`pykeen.models.DistMult` that uses complex valued representations for the
    entities and relations. Entities and relations are represented as vectors
    $\textbf{e}_i, \textbf{r}_i \in \mathbb{C}^d$, and the plausibility score is computed using the
    Hadamard product:

    .. math::

        f(h,r,t) =  Re(\mathbf{e}_h\odot\mathbf{r}_r\odot\bar{\mathbf{e}}_t)

    Which expands to:

    .. math::

        f(h,r,t) = \left\langle Re(\mathbf{e}_h),Re(\mathbf{r}_r),Re(\mathbf{e}_t)\right\rangle
        + \left\langle Im(\mathbf{e}_h),Re(\mathbf{r}_r),Im(\mathbf{e}_t)\right\rangle
        + \left\langle Re(\mathbf{e}_h),Im(\mathbf{r}_r),Im(\mathbf{e}_t)\right\rangle
        - \left\langle Im(\mathbf{e}_h),Im(\mathbf{r}_r),Re(\mathbf{e}_t)\right\rangle

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
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = dict(reduction="mean")
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
        regularizer: HintOrType[Regularizer] = LpRegularizer,
        regularizer_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize ComplEx.

        :param embedding_dim:
            The embedding dimensionality of the entity embeddings.
        :param entity_initializer: Entity initializer function. Defaults to :func:`torch.nn.init.normal_`
        :param relation_initializer: Relation initializer function. Defaults to :func:`torch.nn.init.normal_`
        :param regularizer:
            the regularizer to apply.
        :param regularizer_kwargs:
            additional keyword arguments passed to the regularizer. Defaults to `ComplEx.regularizer_default_kwargs`.
        :param kwargs:
            Remaining keyword arguments to forward to :class:`pykeen.models.EntityRelationEmbeddingModel`
        """
        regularizer_kwargs = regularizer_kwargs or ComplEx.regularizer_default_kwargs
        super().__init__(
            interaction=ComplExInteraction,
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                # use torch's native complex data type
                dtype=torch.cfloat,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
                # use torch's native complex data type
                dtype=torch.cfloat,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            **kwargs,
        )
