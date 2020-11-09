# -*- coding: utf-8 -*-

"""Implementation of ConvE."""

import logging
import math
from typing import Optional, Tuple, Type

import torch
from torch import nn

from ..base import EntityRelationEmbeddingModel, InteractionFunction
from ...losses import BCEAfterSigmoidLoss, Loss
from ...nn import Embedding
from ...nn.init import xavier_normal_
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...utils import is_cudnn_error

__all__ = [
    'ConvE',
]

logger = logging.getLogger(__name__)


def _calculate_missing_shape_information(
    embedding_dim: int,
    input_channels: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Automatically calculates missing dimensions for ConvE.

    :param embedding_dim:
    :param input_channels:
    :param width:
    :param height:

    :return: (input_channels, width, height), such that
            `embedding_dim = input_channels * width * height`
    :raises:
        If no factorization could be found.
    """
    # Store initial input for error message
    original = (input_channels, width, height)

    # All are None
    if all(factor is None for factor in [input_channels, width, height]):
        input_channels = 1
        result_sqrt = math.floor(math.sqrt(embedding_dim))
        height = max(factor for factor in range(1, result_sqrt + 1) if embedding_dim % factor == 0)
        width = embedding_dim // height

    # input_channels is None, and any of height or width is None -> set input_channels=1
    if input_channels is None and any(remaining is None for remaining in [width, height]):
        input_channels = 1

    # input channels is not None, and one of height or width is None
    assert len([factor for factor in [input_channels, width, height] if factor is None]) <= 1
    if width is None:
        width = embedding_dim // (height * input_channels)
    if height is None:
        height = embedding_dim // (width * input_channels)
    if input_channels is None:
        input_channels = embedding_dim // (width * height)
    assert not any(factor is None for factor in [input_channels, width, height])

    if input_channels * width * height != embedding_dim:
        raise ValueError(f'Could not resolve {original} to a valid factorization of {embedding_dim}.')

    return input_channels, width, height


def _add_cuda_warning(func):
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if not is_cudnn_error(e):
                raise e
            raise RuntimeError(
                '\nThis code crash might have been caused by a CUDA bug, see '
                'https://github.com/allenai/allennlp/issues/2888, '
                'which causes the code to crash during evaluation mode.\n'
                'To avoid this error, the batch size has to be reduced.',
            ) from e

    return wrapped


class ConvEInteractionFunction(InteractionFunction):
    """ConvE interaction function."""

    #: If batch normalization is enabled, this is: num_features – C from an expected input of size (N,C,L)
    bn0: Optional[torch.nn.BatchNorm2d]
    #: If batch normalization is enabled, this is: num_features – C from an expected input of size (N,C,H,W)
    bn1: Optional[torch.nn.BatchNorm2d]
    bn2: Optional[torch.nn.BatchNorm2d]

    def __init__(
        self,
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
        apply_batch_normalization: bool = True,
    ):
        super().__init__()

        # Automatic calculation of remaining dimensions
        logger.info(f'Resolving {input_channels} * {embedding_width} * {embedding_height} = {embedding_dim}.')
        if embedding_dim is None:
            embedding_dim = input_channels * embedding_width * embedding_height

        # Parameter need to fulfil:
        #   input_channels * embedding_height * embedding_width = embedding_dim
        input_channels, embedding_width, embedding_height = _calculate_missing_shape_information(
            embedding_dim=embedding_dim,
            input_channels=input_channels,
            width=embedding_width,
            height=embedding_height,
        )
        logger.info(f'Resolved to {input_channels} * {embedding_width} * {embedding_height} = {embedding_dim}.')
        self.embedding_dim = embedding_dim
        self.embedding_height = embedding_height
        self.embedding_width = embedding_width
        self.input_channels = input_channels

        if self.input_channels * self.embedding_height * self.embedding_width != self.embedding_dim:
            raise ValueError(
                f'Product of input channels ({self.input_channels}), height ({self.embedding_height}), and width '
                f'({self.embedding_width}) does not equal target embedding dimension ({self.embedding_dim})',
            )

        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(output_dropout)
        self.feature_map_drop = nn.Dropout2d(feature_map_dropout)

        self.conv1 = torch.nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=output_channels,
            kernel_size=(kernel_height, kernel_width),
            stride=1,
            padding=0,
            bias=True,
        )

        self.apply_batch_normalization = apply_batch_normalization
        if self.apply_batch_normalization:
            self.bn0 = nn.BatchNorm2d(self.input_channels)
            self.bn1 = nn.BatchNorm2d(output_channels)
            self.bn2 = nn.BatchNorm1d(self.embedding_dim)
        else:
            self.bn0 = None
            self.bn1 = None
            self.bn2 = None
        self.num_in_features = (
            output_channels
            * (2 * self.embedding_height - kernel_height + 1)
            * (self.embedding_width - kernel_width + 1)
        )
        self.fc = nn.Linear(self.num_in_features, self.embedding_dim)
        self.activation = nn.ReLU()

    @_add_cuda_warning
    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa: D102
        # get tail bias term
        if "t_bias" not in kwargs:
            raise TypeError(f"{self.__class__.__name__}.forward expects keyword argument 't_bias'.")
        t_bias: torch.FloatTensor = kwargs.pop("t_bias")
        self._check_for_empty_kwargs(kwargs)

        # bind sizes
        batch_size = max(x.shape[0] for x in (h, r, t))
        num_heads = h.shape[1]
        num_relations = r.shape[1]
        num_tails = t.shape[1]

        # repeat if necessary
        h = h.unsqueeze(dim=2).repeat(1 if h.shape[0] == batch_size else batch_size, 1, num_relations, 1)
        r = r.unsqueeze(dim=1).repeat(1 if r.shape[0] == batch_size else batch_size, num_heads, 1, 1)

        # resize and concat head and relation, batch_size', num_input_channels, 2*height, width
        # with batch_size' = batch_size * num_heads * num_relations
        x = torch.cat([
            h.view(-1, self.input_channels, self.embedding_height, self.embedding_width),
            r.view(-1, self.input_channels, self.embedding_height, self.embedding_width),
        ], dim=2)

        # batch_size, num_input_channels, 2*height, width
        if self.apply_batch_normalization:
            x = self.bn0(x)

        # batch_size, num_input_channels, 2*height, width
        x = self.inp_drop(x)

        # (N,C_out,H_out,W_out)
        x = self.conv1(x)

        if self.apply_batch_normalization:
            x = self.bn1(x)

        x = self.activation(x)
        x = self.feature_map_drop(x)

        # batch_size', num_output_channels * (2 * height - kernel_height + 1) * (width - kernel_width + 1)
        x = x.view(-1, self.num_in_features)
        x = self.fc(x)
        x = self.hidden_drop(x)

        if self.apply_batch_normalization:
            x = self.bn2(x)
        x = self.activation(x)

        # reshape: (batch_size', embedding_dim)
        x = x.view(batch_size, num_heads, num_relations, 1, self.embedding_dim)

        # For efficient calculation, each of the convolved [h, r] rows has only to be multiplied with one t row
        # output_shape: (batch_size, num_heads, num_relations, num_tails)
        t = t.view(t.shape[0], 1, 1, num_tails, self.embedding_dim).transpose(-1, -2)
        x = (x @ t).squeeze(dim=-2)

        # add bias term
        x = x + t_bias.view(t.shape[0], 1, 1, num_tails)

        return x


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
        preferred_device: Optional[str] = None,
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

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(indices=hrt_batch[:, 0])
        r = self.relation_embeddings(indices=hrt_batch[:, 1])
        t = self.entity_embeddings(indices=hrt_batch[:, 2])
        t_bias = self.bias_term(indices=hrt_batch[:, 2])
        self.regularize_if_necessary(h, r, t)
        return self.interaction_function.score_hrt(h=h, r=r, t=t, t_bias=t_bias)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(indices=hr_batch[:, 0])
        r = self.relation_embeddings(indices=hr_batch[:, 1])
        all_entities = self.entity_embeddings(indices=None)
        t_bias = self.bias_term(indices=None)
        self.regularize_if_necessary(h, r, all_entities)
        return self.interaction_function.score_t(h=h, r=r, all_entities=all_entities, t_bias=t_bias)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        all_entities = self.entity_embeddings(indices=None)
        r = self.relation_embeddings(indices=rt_batch[:, 0])
        t = self.entity_embeddings(indices=rt_batch[:, 1])
        t_bias = self.bias_term(indices=rt_batch[:, 1])
        self.regularize_if_necessary(all_entities, r, t)
        return self.interaction_function.score_h(all_entities=all_entities, r=r, t=t, t_bias=t_bias)
