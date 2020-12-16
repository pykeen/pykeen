# -*- coding: utf-8 -*-

"""Implementation of the ConvKB model."""

import logging
from typing import Any, ClassVar, Mapping, Optional

from ..base import ERModel
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import Loss
from ...nn import EmbeddingSpecification
from ...nn.modules import ConvKBInteraction
from ...regularizers import LpRegularizer, Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'ConvKB',
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

       - Authors' `implementation of ConvKB <https://github.com/daiquocnguyen/ConvKBsE.py>`_
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        hidden_dropout_rate=DEFAULT_DROPOUT_HPO_RANGE,
        num_filters=dict(type=int, low=7, high=9, scale='power_two'),
    )
    #: The regularizer used by [nguyen2018]_ for ConvKB.
    regularizer_default = LpRegularizer
    #: The LP settings used by [nguyen2018]_ for ConvKB.
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=0.001 / 2,
        p=2.0,
        normalize=True,
        apply_only_once=True,
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        hidden_dropout_rate: float = 0.,
        embedding_dim: int = 200,
        loss: Optional[Loss] = None,
        regularizer: Optional[Regularizer] = None,
        preferred_device: DeviceHint = None,
        num_filters: int = 400,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the model.

        To be consistent with the paper, pass entity and relation embeddings pre-trained from TransE.
        """
        super().__init__(
            triples_factory=triples_factory,
            interaction=ConvKBInteraction(
                hidden_dropout_rate=hidden_dropout_rate,
                embedding_dim=embedding_dim,
                num_filters=num_filters,
            ),
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
            ),
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
        if regularizer is None:
            regularizer = self._instantiate_default_regularizer()
        # In the code base only the weights of the output layer are used for regularization
        # c.f. https://github.com/daiquocnguyen/ConvKB/blob/73a22bfa672f690e217b5c18536647c7cf5667f1/model.py#L60-L66
        if regularizer is not None:
            self.append_weight_regularizer(
                parameter=self.interaction.parameters(),
                regularizer=regularizer,
            )
        logger.warning('To be consistent with the paper, initialize entity and relation embeddings from TransE.')
