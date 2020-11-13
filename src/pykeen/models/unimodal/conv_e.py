# -*- coding: utf-8 -*-

"""Implementation of ConvE."""
import logging
from typing import Optional, Type

import torch
from torch import nn

from ..base import EntityRelationEmbeddingModel
from ...losses import BCEAfterSigmoidLoss, Loss
from ...nn import Embedding
from ...nn.init import xavier_normal_
from ...nn.modules import ConvEInteractionFunction
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'ConvE',
]

logger = logging.getLogger(__name__)


class ConvE(EntityRelationEmbeddingModel):
    r"""An implementation of ConvE from [dettmers2018]_.

    ConvE  is a CNN-based approach. For each triple $(h,r,t)$, the input to ConvE is a matrix
    $\mathbf{A} \in \mathbb{R}^{2 \times d}$ where the first row of $\mathbf{A}$ represents
    $\mathbf{h} \in \mathbb{R}^d$ and the second row represents $\mathbf{r} \in \mathbb{R}^d$. $\mathbf{A}$ is
    reshaped to a matrix $\mathbf{B} \in \mathbb{R}^{m \times n}$ where the first $m/2$ half rows represent
    $\mathbf{h}$ and the remaining $m/2$ half rows represent $\mathbf{r}$. In the convolution layer, a set of
    \textit{2-dimensional} convolutional filters
    $\Omega = \{\omega_i \ | \ \omega_i \in \mathbb{R}^{r \times c}\}$ are applied on
    $\mathbf{B}$ that capture interactions between $\mathbf{h}$ and $\mathbf{r}$. The resulting feature maps are
    reshaped and concatenated in order to create a feature vector $\mathbf{v} \in \mathbb{R}^{|\Omega|rc}$. In the
    next step, $\mathbf{v}$ is mapped into the entity space using a linear transformation
    $\mathbf{W} \in \mathbb{R}^{|\Omega|rc \times d}$, that is $\mathbf{e}_{h,r} = \mathbf{v}^{T} \mathbf{W}$.
    The score for the triple $(h,r,t) \in \mathbb{K}$ is then given by:

    .. math::

        f(h,r,t) = \mathbf{e}_{h,r} \mathbf{t}

    Since the interaction model can be decomposed into
    $f(h,r,t) = \left\langle f'(\mathbf{h}, \mathbf{r}), \mathbf{t} \right\rangle$, the model is particularly
    designed to 1-N scoring, i.e. efficient computation of scores for $(h,r,t)$ for fixed $h,r$ and
    many different $t$.

    .. seealso::

        - Official Implementation: https://github.com/TimDettmers/ConvE/blob/master/model.py

    The default setting uses batch normalization. Batch normalization normalizes the output of the activation functions,
    in order to ensure that the weights of the NN don't become imbalanced and to speed up training.
    However, batch normalization is not the only way to achieve more robust and effective training [1]. Therefore,
    we added the flag 'apply_batch_normalization' to turn batch normalization on/off (it's turned on as default).

    [1]: Santurkar, Shibani, et al. "How does batch normalization help optimization?."
    Advances in Neural Information Processing Systems. 2018.

    Example usage:

    >>> # Step 1: Get triples
    >>> from pykeen.datasets import Nations
    >>> dataset = Nations(create_inverse_triples=True)
    >>> # Step 2: Configure the model
    >>> from pykeen.models import ConvE
    >>> model = ConvE(
    ...     embedding_dim       = 200,
    ...     input_channels      = 1,
    ...     output_channels     = 32,
    ...     embedding_height    = 10,
    ...     embedding_width     = 20,
    ...     kernel_height       = 3,
    ...     kernel_width        = 3,
    ...     input_dropout       = 0.2,
    ...     feature_map_dropout = 0.2,
    ...     output_dropout      = 0.3,
    ...     preferred_device    = 'gpu',
    ... )
    >>> # Step 3: Configure the loop
    >>> from torch.optim import Adam
    >>> optimizer = Adam(params=model.get_grad_params())
    >>> from pykeen.training import LCWATrainingLoop
    >>> training_loop = LCWATrainingLoop(model=model, optimizer=optimizer)
    >>> # Step 4: Train
    >>> losses = training_loop.train(num_epochs=5, batch_size=256)
    >>> # Step 5: Evaluate the model
    >>> from pykeen.evaluation import RankBasedEvaluator
    >>> evaluator = RankBasedEvaluator()
    >>> metric_result = evaluator.evaluate(model=model, mapped_triples=dataset.testing.mapped_triples, batch_size=8192)
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        output_channels=dict(type=int, low=16, high=64),
        input_dropout=dict(type=float, low=0.0, high=1.0),
        output_dropout=dict(type=float, low=0.0, high=1.0),
        feature_map_dropout=dict(type=float, low=0.0, high=1.0),
    )
    #: The default loss function class
    loss_default: Type[Loss] = BCEAfterSigmoidLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs = {}

    def __init__(
        self,
        triples_factory: TriplesFactory,
        input_channels: Optional[int] = None,
        output_channels: int = 32,
        embedding_height: Optional[int] = None,
        embedding_width: Optional[int] = None,
        kernel_height: int = 3,
        kernel_width: int = 3,
        input_dropout: float = 0.2,
        output_dropout: float = 0.3,
        feature_map_dropout: float = 0.2,
        embedding_dim: int = 200,
        automatic_memory_optimization: Optional[bool] = None,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
        apply_batch_normalization: bool = True,
    ) -> None:
        """Initialize the model."""
        # ConvE should be trained with inverse triples
        if not triples_factory.create_inverse_triples:
            logger.warning(
                '\nThe ConvE model should be trained with inverse triples.\n'
                'This can be done by defining the TriplesFactory class with the _create_inverse_triples_ parameter set '
                'to true.',
            )

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            entity_initializer=xavier_normal_,
            relation_initializer=xavier_normal_,
        )

        # ConvE uses one bias for each entity
        self.bias_term = Embedding.init_with_device(
            num_embeddings=triples_factory.num_entities,
            embedding_dim=1,
            device=self.device,
            initializer=nn.init.zeros_,
        )

        self.interaction_function = ConvEInteractionFunction(
            input_channels=input_channels,
            output_channels=output_channels,
            embedding_height=embedding_height,
            embedding_width=embedding_width,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            input_dropout=input_dropout,
            output_dropout=output_dropout,
            feature_map_dropout=feature_map_dropout,
            embedding_dim=embedding_dim,
            apply_batch_normalization=apply_batch_normalization,
        )

    def _reset_parameters_(self):  # noqa: D102
        super()._reset_parameters_()
        self.bias_term.reset_parameters()
        self.interaction_function.reset_parameters()

    def score(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
    ) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings.get_in_canonical_shape(indices=h_indices)
        r = self.relation_embeddings.get_in_canonical_shape(indices=r_indices)
        t = self.entity_embeddings.get_in_canonical_shape(indices=t_indices)
        t_bias = self.bias_term.get_in_canonical_shape(indices=t_indices)
        self.regularize_if_necessary(h, r, t)
        return self.interaction_function(h=h, r=r, t=t, t_bias=t_bias)
