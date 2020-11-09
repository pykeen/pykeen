# -*- coding: utf-8 -*-

"""Implementation of ConvE."""

import logging
import math
import sys
from typing import Optional, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from ..base import EntityRelationEmbeddingModel
from ...losses import BCEAfterSigmoidLoss, Loss
from ...nn import Embedding
from ...nn.init import xavier_normal_
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint
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

    #: If batch normalization is enabled, this is: num_features – C from an expected input of size (N,C,L)
    bn0: Optional[torch.nn.BatchNorm2d]
    #: If batch normalization is enabled, this is: num_features – C from an expected input of size (N,C,H,W)
    bn1: Optional[torch.nn.BatchNorm2d]
    bn2: Optional[torch.nn.BatchNorm2d]

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
        num_in_features = (
            output_channels
            * (2 * self.embedding_height - kernel_height + 1)
            * (self.embedding_width - kernel_width + 1)
        )
        self.fc = nn.Linear(num_in_features, self.embedding_dim)

    def _reset_parameters_(self):  # noqa: D102
        super()._reset_parameters_()

        self.bias_term.reset_parameters()

        # weights
        for module in [
            self.conv1,
            self.bn0,
            self.bn1,
            self.bn2,
            self.fc,
        ]:
            if module is None:
                continue
            module.reset_parameters()

    def _convolve_entity_relation(self, h: torch.LongTensor, r: torch.LongTensor) -> torch.FloatTensor:
        """Perform the main calculations of the ConvE model."""
        batch_size = h.shape[0]

        # batch_size, num_input_channels, 2*height, width
        x = torch.cat([h, r], dim=2)

        try:
            # batch_size, num_input_channels, 2*height, width
            if self.apply_batch_normalization:
                x = self.bn0(x)

            # batch_size, num_input_channels, 2*height, width
            x = self.inp_drop(x)
            # (N,C_out,H_out,W_out)
            x = self.conv1(x)

            if self.apply_batch_normalization:
                x = self.bn1(x)
            x = F.relu(x)
            x = self.feature_map_drop(x)
            # batch_size, num_output_channels * (2 * height - kernel_height + 1) * (width - kernel_width + 1)
            x = x.view(batch_size, -1)
            x = self.fc(x)
            x = self.hidden_drop(x)

            if self.apply_batch_normalization:
                x = self.bn2(x)
            x = F.relu(x)
        except RuntimeError as e:
            if not is_cudnn_error(e):
                raise e
            logger.warning(
                '\nThis code crash might have been caused by a CUDA bug, see '
                'https://github.com/allenai/allennlp/issues/2888, '
                'which causes the code to crash during evaluation mode.\n'
                'To avoid this error, the batch size has to be reduced.\n'
                f'The original error message: \n{e.args[0]}',
            )
            sys.exit(1)

        return x

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(indices=hrt_batch[:, 0]).view(
            -1,
            self.input_channels,
            self.embedding_height,
            self.embedding_width,
        )
        r = self.relation_embeddings(indices=hrt_batch[:, 1]).view(
            -1,
            self.input_channels,
            self.embedding_height,
            self.embedding_width,
        )
        t = self.entity_embeddings(indices=hrt_batch[:, 2])

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        x = self._convolve_entity_relation(h, r)

        # For efficient calculation, each of the convolved [h, r] rows has only to be multiplied with one t row
        x = (x.view(-1, self.embedding_dim) * t).sum(dim=1, keepdim=True)

        """
        In ConvE the bias term add the end is added for each tail item. In the sLCWA training approach we only have
        one tail item for each head and relation. Accordingly the relevant bias for each tail item and triple has to be
        looked up.
        """
        x = x + self.bias_term(indices=hrt_batch[:, 2])
        # The application of the sigmoid during training is automatically handled by the default loss.

        return x

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(indices=hr_batch[:, 0]).view(
            -1,
            self.input_channels,
            self.embedding_height,
            self.embedding_width,
        )
        r = self.relation_embeddings(indices=hr_batch[:, 1]).view(
            -1,
            self.input_channels,
            self.embedding_height,
            self.embedding_width,
        )
        t = self.entity_embeddings(indices=None).transpose(1, 0)

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        x = self._convolve_entity_relation(h, r)

        x = x @ t
        x = x + self.bias_term(indices=None).t()
        # The application of the sigmoid during training is automatically handled by the default loss.

        return x

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        rt_batch_size = rt_batch.shape[0]
        h = self.entity_embeddings(indices=None)
        r = self.relation_embeddings(indices=rt_batch[:, 0]).view(
            -1,
            self.input_channels,
            self.embedding_height,
            self.embedding_width,
        )
        t = self.entity_embeddings(indices=rt_batch[:, 1])

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        '''
        Every head has to be convolved with every relation in the rt_batch. Hence we repeat the
        relation _num_entities_ times and the head _rt_batch_size_ times.
        '''
        r = r.repeat(h.shape[0], 1, 1, 1)
        # Code to repeat each item successively instead of the entire tensor
        h = h.unsqueeze(1).repeat(1, rt_batch_size, 1).view(
            -1,
            self.input_channels,
            self.embedding_height,
            self.embedding_width,
        )

        x = self._convolve_entity_relation(h, r)

        '''
        For efficient computation, each convolved [h, r] pair has only to be multiplied with the corresponding t
        embedding found in the rt_batch with [r, t] pairs.
        '''
        x = (x.view(self.num_entities, rt_batch_size, self.embedding_dim) * t[None, :, :]).sum(2).transpose(1, 0)

        """
        In ConvE the bias term at the end is added for each tail item. In the score_h function, each row holds
        the same tail for many different heads, meaning that these items have to be looked up for each tail of each row
        and only then can be added correctly.
        """
        x = x + self.bias_term(indices=rt_batch[:, 1])
        # The application of the sigmoid during training is automatically handled by the default loss.

        return x
