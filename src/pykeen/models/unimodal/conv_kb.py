# -*- coding: utf-8 -*-

"""Implementation of the ConvKB model."""

import logging
from typing import Any, ClassVar, Mapping, Optional, Type

from torch.nn.init import uniform_

from ..nbase import ERModel
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.modules import ConvKBInteraction
from ...regularizers import LpRegularizer, Regularizer
from ...typing import Hint, Initializer

__all__ = [
    "ConvKB",
]

logger = logging.getLogger(__name__)


class ConvKB(ERModel):
    r"""An implementation of ConvKB from [nguyen2018]_.

    ConvKB uses a convolutional neural network (CNN) whose feature maps capture global interactions of the input.
    Each triple $(h,r,t) \in \mathbb{K}$ is represented as a input matrix
    $\mathbf{A} = [\mathbf{h}; \mathbf{r}; \mathbf{t}] \in \mathbb{R}^{d \times 3}$ in which the columns represent
    the embeddings for $h$, $r$, and $t$. In the convolution layer, a set of convolutional filters
    $\omega_i \in \mathbb{R}^{1 \times 3}, i=1, \dots, \tau,$ are applied on the input in order to compute for
    each dimension global interactions of the embedded triple. Each $\omega_i $ is applied on every row of
    $\mathbf{A}$ creating a feature map $\mathbf{v}_i = [v_{i,1},...,v_{i,d}] \in \mathbb{R}^d$:

    .. math::

        \mathbf{v}_i = g(\omega_j \mathbf{A} + \mathbf{b})

    where $\mathbf{b} \in \mathbb{R}$ denotes a bias term and $g$ an activation function which is employed element-wise.
    Based on the resulting feature maps $\mathbf{v}_1, \dots, \mathbf{v}_{\tau}$, the plausibility score of a triple
    is given by:

    .. math::

        f(h,r,t) = [\mathbf{v}_i; \ldots ;\mathbf{v}_\tau] \cdot \mathbf{w}

    where $[\mathbf{v}_i; \ldots ;\mathbf{v}_\tau] \in \mathbb{R}^{\tau d \times 1}$ and
    $\mathbf{w} \in \mathbb{R}^{\tau d \times 1} $ is a shared weight vector.
    ConvKB may be seen as a restriction of :class:`pykeen.models.ERMLP` with a certain weight sharing pattern in the
    first layer.

    .. seealso::

       - Authors' `implementation of ConvKB <https://github.com/daiquocnguyen/ConvKB>`_
    ---
    citation:
        author: Nguyen
        year: 2018
        link: https://www.aclweb.org/anthology/N18-2053
        github: daiquocnguyen/ConvKB
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        hidden_dropout_rate=DEFAULT_DROPOUT_HPO_RANGE,
        num_filters=dict(type=int, low=7, high=9, scale="power_two"),
    )
    #: The regularizer used by [nguyen2018]_ for ConvKB.
    regularizer_default: ClassVar[Type[Regularizer]] = LpRegularizer
    #: The LP settings used by [nguyen2018]_ for ConvKB.
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=0.001 / 2,
        p=2.0,
        normalize=True,
        apply_only_once=True,
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        hidden_dropout_rate: float = 0.0,
        num_filters: int = 400,
        regularizer: Optional[Regularizer] = None,
        entity_initializer: Hint[Initializer] = uniform_,
        relation_initializer: Hint[Initializer] = uniform_,
        **kwargs,
    ) -> None:
        """Initialize the model.

        :param embedding_dim: The entity embedding dimension $d$.
        :param hidden_dropout_rate: The hidden dropout rate
        :param num_filters: The number of convolutional filters to use
        :param regularizer: The regularizer to use. Defaults to $L_p$
        :param entity_initializer: Entity initializer function. Defaults to :func:`torch.nn.init.uniform_`
        :param relation_initializer: Relation initializer function. Defaults to :func:`torch.nn.init.uniform_`
        :param kwargs:
            Remaining keyword arguments passed through to :class:`pykeen.models.EntityRelationEmbeddingModel`.

        To be consistent with the paper, pass entity and relation embeddings pre-trained from TransE.
        """
        super().__init__(
            interaction=ConvKBInteraction,
            interaction_kwargs=dict(
                hidden_dropout_rate=hidden_dropout_rate,
                embedding_dim=embedding_dim,
                num_filters=num_filters,
            ),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
            ),
            **kwargs,
        )
        if regularizer is None:
            regularizer = self._instantiate_default_regularizer()
        # In the code base only the weights of the output layer are used for regularization
        # c.f. https://github.com/daiquocnguyen/ConvKB/blob/73a22bfa672f690e217b5c18536647c7cf5667f1/model.py#L60-L66
        if regularizer is not None:
            self.append_weight_regularizer(
                parameter=self.interaction.linear.parameters(),
                regularizer=regularizer,
            )
        logger.warning("To be consistent with the paper, initialize entity and relation embeddings from TransE.")
