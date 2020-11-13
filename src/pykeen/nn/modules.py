# -*- coding: utf-8 -*-

"""Stateful interaction functions."""

import logging
import math
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from . import functional as pykeen_functional
from ..utils import check_shapes

logger = logging.getLogger(__name__)


class InteractionFunction(nn.Module):
    """Base class for interaction functions."""

    BATCH_DIM: int = 0
    NUM_DIM: int = 1
    HEAD_DIM: int = 1
    RELATION_DIM: int = 2
    TAIL_DIM: int = 3

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        """Score the given triples.

        :param h: shape: (batch_size, num_heads, d_e)
            The head representations.
        :param r: shape: (batch_size, num_relations, d_r)
            The relation representations.
        :param t: shape: (batch_size, num_tails, d_e)
            The tail representations.
        :param kwargs:
            Additional key-word based arguments.

        :return: shape: (batch_size, num_heads, num_relations, num_tails)
            The scores.
        """
        raise NotImplementedError

    @classmethod
    def _check_for_empty_kwargs(cls, kwargs: Mapping[str, Any]) -> None:
        """Check that kwargs is empty."""
        if len(kwargs) > 0:
            raise ValueError(f"{cls.__name__} does not take the following kwargs: {kwargs}")

    @staticmethod
    def _add_dim(*x: torch.FloatTensor, dim: int) -> Sequence[torch.FloatTensor]:
        """
        Add a dimension to tensors.

        :param x: shape: (d1, ..., dk)
            The tensor.

        :return: shape: (1, d1, ..., dk)
            The tensor with batch dimension.
        """
        out = [xx.unsqueeze(dim=dim) for xx in x]
        if len(x) > 1:
            return out
        return out[0]

    @staticmethod
    def _remove_dim(x: torch.FloatTensor, *dims: int) -> torch.FloatTensor:
        """
        Remove dimensions from a tensor.

        :param x:
            The tensor.
        :param dims:
            The dimensions to remove.

        :return:
            The squeezed tensor.
        """
        # normalize dimensions
        dims = [d if d >= 0 else len(x.shape) + d for d in dims]
        if len(set(dims)) != len(dims):
            raise ValueError(f"Duplicate dimensions: {dims}")
        assert all(0 <= d < len(x.shape) for d in dims)
        for dim in sorted(dims, reverse=True):
            x = x.squeeze(dim=dim)
        return x

    def score_hrt(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Score a batch of triples..

        :param h: shape: (batch_size, d_e)
            The head representations.
        :param r: shape: (batch_size, d_r)
            The relation representations.
        :param t: shape: (batch_size, d_e)
            The tail representations.
        :param kwargs:
            Additional key-word based arguments.

        :return: shape: (batch_size, 1)
            The scores.
        """
        # check shape
        assert check_shapes((h, "be"), (r, "br"), (t, "be"))

        # prepare input to generic score function
        h, r, t = self._add_dim(h, r, t, dim=self.NUM_DIM)

        # get scores
        scores = self(h=h, r=r, t=t, **kwargs)

        # prepare output shape, (batch_size, num_heads, num_relations, num_tails) -> (batch_size, 1)
        return self._remove_dim(scores, self.HEAD_DIM, self.RELATION_DIM, self.TAIL_DIM).unsqueeze(dim=-1)

    def score_h(
        self,
        all_entities: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Score all head entities.

        :param all_entities: shape: (num_entities, d_e)
            The head representations.
        :param r: shape: (batch_size, d_r)
            The relation representations.
        :param t: shape: (batch_size, d_e)
            The tail representations.
        :param kwargs:
            Additional key-word based arguments.

        :return: shape: (batch_size, num_entities)
            The scores.
        """
        # check shape
        assert check_shapes((all_entities, "ne"), (r, "br"), (t, "be"))

        # TODO: What about unsqueezing for additional e.g. head arguments
        # prepare input to generic score function
        r, t = self._add_dim(r, t, dim=self.NUM_DIM)
        h = self._add_dim(all_entities, dim=self.BATCH_DIM)

        # get scores
        scores = self(h=h, r=r, t=t, **kwargs)

        # prepare output shape, (batch_size, num_heads, num_relations, num_tails) -> (batch_size, num_heads)
        return self._remove_dim(scores, self.RELATION_DIM, self.TAIL_DIM)

    def score_r(
        self,
        h: torch.FloatTensor,
        all_relations: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Score all relations.

        :param h: shape: (batch_size, d_e)
            The head representations.
        :param all_relations: shape: (batch_size, d_r)
            The relation representations.
        :param t: shape: (num_entities, d_e)
            The tail representations.
        :param kwargs:
            Additional key-word based arguments.

        :return: shape: (batch_size, num_entities)
            The scores.
        """
        # check shape
        assert check_shapes((all_relations, "nr"), (h, "be"), (t, "be"))

        # prepare input to generic score function
        h, t = self._add_dim(h, t, dim=self.NUM_DIM)
        r = self._add_dim(all_relations, dim=self.BATCH_DIM)

        # get scores
        scores = self(h=h, r=r, t=t, **kwargs)

        # prepare output shape
        return self._remove_dim(scores, self.HEAD_DIM, self.TAIL_DIM)

    def score_t(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        all_entities: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Score all tail entities.

        :param h: shape: (batch_size, d_e)
            The head representations.
        :param r: shape: (batch_size, d_r)
            The relation representations.
        :param all_entities: shape: (num_entities, d_e)
            The tail representations.
        :param kwargs:
            Additional key-word based arguments.

        :return: shape: (batch_size, num_entities)
            The scores.
        """
        # check shape
        assert check_shapes((all_entities, "ne"), (r, "br"), (h, "be"))

        # prepare input to generic score function
        h, r = self._add_dim(h, r, dim=self.NUM_DIM)
        t = self._add_dim(all_entities, dim=self.BATCH_DIM)

        # get scores
        scores = self(h=h, r=r, t=t, **kwargs)

        # prepare output shape
        return self._remove_dim(scores, self.HEAD_DIM, self.RELATION_DIM)

    def reset_parameters(self):
        """Reset parameters the interaction function may have."""
        for mod in self.modules():
            if mod is self:
                continue
            if hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()


class FunctionalInteractionFunction(InteractionFunction):
    """Interaction function without state or additional parameters."""
    interaction: Callable[[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa: D102
        self._check_for_empty_kwargs(kwargs)
        return self.interaction(h, r, t)


class TranslationalInteractionFunction(InteractionFunction):
    """The translational interaction function shared by the TransE, TransR, TransH, and other Trans<X> models."""

    def __init__(self, p: int):
        """Initialize the translational interaction function.

        :param p: The norm used with :func:`torch.norm`. Typically is 1 or 2.
        """
        super().__init__()
        self.p = p

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa:D102
        self._check_for_empty_kwargs(kwargs=kwargs)
        return pykeen_functional.translational_interaction(h=h, r=r, t=t, p=self.p)


class ComplExInteractionFunction(FunctionalInteractionFunction):
    """Interaction function of ComplEx."""

    interaction = pykeen_functional.complex_interaction


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
        return pykeen_functional.conve_interaction(
            h=h,
            r=r,
            t=t,
            t_bias=t_bias,
            input_channels=self.input_channels,
            embedding_height=self.embedding_height,
            embedding_width=self.embedding_width,
            num_in_features=self.num_in_features,
            bn0=self.bn0,
            bn1=self.bn1,
            bn2=self.bn2,
            inp_drop=self.inp_drop,
            feature_map_drop=self.feature_map_drop,
            hidden_drop=self.hidden_drop,
            conv1=self.conv1,
            activation=self.activation,
            fc=self.fc,
        )


class ConvKBInteractionFunction(InteractionFunction):
    """Interaction function of ConvKB."""

    def __init__(
        self,
        hidden_dropout_rate: float = 0.,
        embedding_dim: int = 200,
        num_filters: int = 400,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters

        # The interaction model
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(1, 3), bias=True)
        self.activation = nn.ReLU()
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_rate)
        self.linear = nn.Linear(embedding_dim * num_filters, 1, bias=True)

    def reset_parameters(self):  # noqa: D102
        # Use Xavier initialization for weight; bias to zero
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.linear.bias)

        # Initialize all filters to [0.1, 0.1, -0.1],
        #  c.f. https://github.com/daiquocnguyen/ConvKB/blob/master/model.py#L34-L36
        nn.init.constant_(self.conv.weight[..., :2], 0.1)
        nn.init.constant_(self.conv.weight[..., 2], -0.1)
        nn.init.zeros_(self.conv.bias)

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa: D102
        return pykeen_functional.convkb_interaction(
            h=h,
            r=r,
            t=t,
            conv=self.conv,
            activation=self.activation,
            hidden_dropout=self.hidden_dropout,
            linear=self.linear,
        )


class DistMultInteractionFunction(FunctionalInteractionFunction):
    """Interaction function of DistMult."""

    interaction = pykeen_functional.distmult_interaction


class ERMLPInteractionFunction(InteractionFunction):
    """
    Interaction function of ER-MLP.

    .. math ::
        f(h, r, t) = W_2 ReLU(W_1 cat(h, r, t) + b_1) + b_2
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
    ):
        """
        Initialize the interaction function.

        :param embedding_dim:
            The embedding vector dimension.
        :param hidden_dim:
            The hidden dimension of the MLP.
        """
        super().__init__()
        """The multi-layer perceptron consisting of an input layer with 3 * self.embedding_dim neurons, a  hidden layer
           with self.embedding_dim neurons and output layer with one neuron.
           The input is represented by the concatenation embeddings of the heads, relations and tail embeddings.
        """
        self.hidden = nn.Linear(in_features=3 * embedding_dim, out_features=hidden_dim, bias=True)
        self.activation = nn.ReLU()
        self.hidden_to_score = nn.Linear(in_features=hidden_dim, out_features=1, bias=True)

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa: D102
        self._check_for_empty_kwargs(kwargs)
        return pykeen_functional.ermlp_interaction(
            h=h,
            r=r,
            t=t,
            hidden=self.hidden,
            activation=self.activation,
            final=self.hidden_to_score,
        )

    def reset_parameters(self):  # noqa: D102
        # Initialize biases with zero
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.hidden_to_score.bias)
        # In the original formulation,
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(
            self.hidden_to_score.weight,
            gain=nn.init.calculate_gain(self.activation.__class__.__name__.lower()),
        )


class ERMLPEInteractionFunction(InteractionFunction):
    """Interaction function of ER-MLP."""

    def __init__(
        self,
        hidden_dim: int = 300,
        input_dropout: float = 0.2,
        hidden_dropout: float = 0.3,
        embedding_dim: int = 200,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.Dropout(hidden_dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(hidden_dropout),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa: D102
        self._check_for_empty_kwargs(kwargs=kwargs)
        return pykeen_functional.ermlpe_interaction(h=h, r=r, t=t, mlp=self.mlp)


class TransDInteractionFunction(TranslationalInteractionFunction):
    """The interaction function for TransD."""

    def __init__(self, p: int = 2, power: int = 2):
        """Initialize the TransD interaction function.

        :param p: The norm applied by :func:`torch.norm`
        :param power: The power applied after :func:`torch.norm`.
        """
        super().__init__(p=p)
        self.power = power

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa:D102
        return super().forward(h=h, r=r, t=t, **kwargs) ** self.power


class TransRInteractionFunction(InteractionFunction):
    """The TransR interaction function."""

    def __init__(self, p: int):
        """Initialize the TransR interaction function.

        :param p: The norm applied to the translation
        """
        super().__init__()
        self.p = p

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa:D102
        m_r = kwargs.pop('m_r')
        self._check_for_empty_kwargs(kwargs=kwargs)
        return pykeen_functional.transr_interaction(h=h, r=r, t=t, m_r=m_r, p=self.p, power_norm=True)


class RotatEInteraction(FunctionalInteractionFunction):
    """Interaction function of RotatE."""

    interaction = pykeen_functional.rotate_interaction


class HolEInteractionFunction(FunctionalInteractionFunction):
    """Interaction function for HolE."""

    interaction = pykeen_functional.hole_interaction


class ProjEInteractionFunction(InteractionFunction):
    """Interaction function for ProjE."""

    def __init__(
        self,
        embedding_dim: int = 50,
        inner_non_linearity: Optional[nn.Module] = None,
    ):
        super().__init__()

        # Global entity projection
        self.d_e = nn.Parameter(torch.empty(embedding_dim), requires_grad=True)

        # Global relation projection
        self.d_r = nn.Parameter(torch.empty(embedding_dim), requires_grad=True)

        # Global combination bias
        self.b_c = nn.Parameter(torch.empty(embedding_dim), requires_grad=True)

        # Global combination bias
        self.b_p = nn.Parameter(torch.empty(1), requires_grad=True)

        if inner_non_linearity is None:
            inner_non_linearity = nn.Tanh()
        self.inner_non_linearity = inner_non_linearity

    def reset_parameters(self):  # noqa: D102
        embedding_dim = self.d_e.shape[0]
        bound = math.sqrt(6) / embedding_dim
        for p in self.parameters():
            nn.init.uniform_(p, a=-bound, b=bound)

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        self._check_for_empty_kwargs(kwargs=kwargs)

        # Compute score
        return pykeen_functional.proje_interaction(h=h, r=r, t=t, d_e=self.d_e, d_r=self.d_r, b_c=self.b_c, b_p=self.b_p, activation=self.inner_non_linearity).view(-1, 1)


class RESCALInteractionFunction(FunctionalInteractionFunction):
    """RESCAL interaction function."""

    interaction = pykeen_functional.rescal_interaction


class StructuredEmbeddingInteractionFunction(InteractionFunction):
    """Interaction function of Structured Embedding."""

    def __init__(
        self,
        p: int,
        power_norm: bool = False
    ):
        super().__init__()
        self.p = p
        self.power_norm = power_norm

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        dim = h.shape[-1]
        rh, rt = r.split(dim, dim=-1)
        rh = rh.view(rh.shape[:-1], dim, dim)
        rt = rt.view(rt.shape[:-1], dim, dim)
        return pykeen_functional.structured_embedding_interaction(
            h=h,
            r_h=rh,
            r_t=rt,
            t=t,
            p=self.p,
            power_norm=self.power_norm,
        )
