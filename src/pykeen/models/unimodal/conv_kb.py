# -*- coding: utf-8 -*-

"""Implementation of the ConvKB model."""

import logging
from typing import Optional

import torch
import torch.autograd
from torch import nn

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss
from ...regularizers import LpRegularizer, Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'ConvKB',
]

logger = logging.getLogger(__name__)


class ConvKB(EntityRelationEmbeddingModel):
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
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=300, q=50),
        hidden_dropout_rate=dict(type=float, low=0.1, high=0.9),
        num_filters=dict(type=int, low=300, high=500, q=50),
    )
    #: The regularizer used by [nguyen2018]_ for ConvKB.
    regularizer_default = LpRegularizer
    #: The LP settings used by [nguyen2018]_ for ConvKB.
    regularizer_default_kwargs = dict(
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
        automatic_memory_optimization: Optional[bool] = None,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        num_filters: int = 400,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the model.

        To be consistent with the paper, pass entity and relation embeddings pre-trained from TransE.
        """
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            loss=loss,
            automatic_memory_optimization=automatic_memory_optimization,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        self.num_filters = num_filters

        # The interaction model
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(1, 3), bias=True)
        self.relu = nn.ReLU()
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_rate)
        self.linear = nn.Linear(embedding_dim * num_filters, 1, bias=True)

    def _reset_parameters_(self):  # noqa: D102
        # embeddings
        logger.warning('To be consistent with the paper, initialize entity and relation embeddings from TransE.')
        super()._reset_parameters_()

        # Use Xavier initialization for weight; bias to zero
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.linear.bias)

        # Initialize all filters to [0.1, 0.1, -0.1],
        #  c.f. https://github.com/daiquocnguyen/ConvKB/blob/master/model.py#L34-L36
        nn.init.constant_(self.conv.weight[..., :2], 0.1)
        nn.init.constant_(self.conv.weight[..., 2], -0.1)
        nn.init.zeros_(self.conv.bias)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(indices=hrt_batch[:, 0])
        r = self.relation_embeddings(indices=hrt_batch[:, 1])
        t = self.entity_embeddings(indices=hrt_batch[:, 2])

        # Output layer regularization
        # In the code base only the weights of the output layer are used for regularization
        # c.f. https://github.com/daiquocnguyen/ConvKB/blob/73a22bfa672f690e217b5c18536647c7cf5667f1/model.py#L60-L66
        self.regularize_if_necessary(self.linear.weight, self.linear.bias)

        # Stack to convolution input
        conv_inp = torch.stack([h, r, t], dim=-1).view(-1, 1, self.embedding_dim, 3)

        # Convolution
        conv_out = self.conv(conv_inp).view(-1, self.embedding_dim * self.num_filters)
        hidden = self.relu(conv_out)

        # Apply dropout, cf. https://github.com/daiquocnguyen/ConvKB/blob/master/model.py#L54-L56
        hidden = self.hidden_dropout(hidden)

        # Linear layer for final scores
        scores = self.linear(hidden)

        return scores
