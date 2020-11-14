# -*- coding: utf-8 -*-

"""Stateful interaction functions."""
import itertools
import logging
import math
from abc import ABC
from typing import Callable, Generic, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from . import functional as pkf
from .functional import KG2E_SIMILARITIES
from ..typing import HeadRepresentation, RelationRepresentation, Representation, TailRepresentation
from ..utils import check_shapes

logger = logging.getLogger(__name__)


def _ensure_tuple(*x: Union[Representation, Sequence[Representation]]) -> Tuple[Sequence[Representation], ...]:
    return tuple(xx if isinstance(xx, Sequence) else (xx,) for xx in x)


class InteractionFunction(nn.Module, Generic[HeadRepresentation, RelationRepresentation, TailRepresentation]):
    """Base class for interaction functions."""

    # Dimensions
    BATCH_DIM: int = 0
    NUM_DIM: int = 1
    HEAD_DIM: int = 1
    RELATION_DIM: int = 2
    TAIL_DIM: int = 3

    #: The symbolic shapes for entity representations
    entity_shape: Tuple[str, ...] = ("d",)
    tail_entity_shape: Union[None, Tuple[str, ...]] = None

    #: The symbolic shapes for relation representations
    relation_shape: Tuple[str, ...] = ("d",)

    def forward(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> torch.FloatTensor:
        """Compute broadcasted triple scores given representations for head, relation and tails.

        :param h: shape: (batch_size, num_heads, *)
            The head representations.
        :param r: shape: (batch_size, num_relations, *)
            The relation representations.
        :param t: shape: (batch_size, num_tails, *)
            The tail representations.

        :return: shape: (batch_size, num_heads, num_relations, num_tails)
            The scores.
        """
        raise NotImplementedError

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
        if len(x) == 1:
            return out[0]
        return out

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

    def _check_shapes(
        self,
        h: HeadRepresentation = tuple(),
        r: RelationRepresentation = tuple(),
        t: TailRepresentation = tuple(),
        h_prefix: str = "b",
        r_prefix: str = "b",
        t_prefix: str = "b",
        raise_on_error: bool = True,
    ) -> bool:
        entity_shape = self.entity_shape
        if isinstance(entity_shape, str):
            entity_shape = (entity_shape,)
        relation_shape = self.relation_shape
        if isinstance(relation_shape, str):
            relation_shape = (relation_shape,)
        tail_entity_shape = self.tail_entity_shape
        if tail_entity_shape is None:
            tail_entity_shape = entity_shape
        if isinstance(tail_entity_shape, str):
            tail_entity_shape = (tail_entity_shape,)
        if len(h) != len(entity_shape):
            if raise_on_error:
                raise ValueError
            return False
        if len(r) != len(relation_shape):
            if raise_on_error:
                raise ValueError
            return False
        if len(t) != len(tail_entity_shape):
            if raise_on_error:
                raise ValueError
            return False
        return check_shapes(*itertools.chain(
            ((hh, h_prefix + hs) for hh, hs in zip(h, entity_shape)),
            ((rr, r_prefix + rs) for rr, rs in zip(r, relation_shape)),
            ((tt, t_prefix + ts) for tt, ts in zip(t, tail_entity_shape)),
        ), raise_or_error=raise_on_error)

    def score_hrt(
        self,
        h: HeadRepresentation = tuple(),
        r: RelationRepresentation = tuple(),
        t: TailRepresentation = tuple(),
    ) -> torch.FloatTensor:
        """
        Score a batch of triples..

        :param h: shape: (batch_size, d_e)
            The head representations.
        :param r: shape: (batch_size, d_r)
            The relation representations.
        :param t: shape: (batch_size, d_e)
            The tail representations.

        :return: shape: (batch_size, 1)
            The scores.
        """
        h, r, t = _ensure_tuple(h, r, t)
        assert self._check_shapes(h=h, r=r, t=t)

        # prepare input to generic score function
        h = self._add_dim(*h, dim=self.NUM_DIM)
        r = self._add_dim(*r, dim=self.NUM_DIM)
        t = self._add_dim(*t, dim=self.NUM_DIM)

        # get scores
        scores = self(h=h, r=r, t=t)

        # prepare output shape, (batch_size, num_heads, num_relations, num_tails) -> (batch_size, 1)
        return self._remove_dim(scores, self.HEAD_DIM, self.RELATION_DIM, self.TAIL_DIM).unsqueeze(dim=-1)

    def score_h(
        self,
        all_entities: HeadRepresentation = tuple(),
        r: RelationRepresentation = tuple(),
        t: TailRepresentation = tuple(),
    ) -> torch.FloatTensor:
        """
        Score all head entities.

        :param all_entities: shape: (num_entities, d_e)
            The head representations.
        :param r: shape: (batch_size, d_r)
            The relation representations.
        :param t: shape: (batch_size, d_e)
            The tail representations.

        :return: shape: (batch_size, num_entities)
            The scores.
        """
        h, r, t = _ensure_tuple(all_entities, r, t)
        assert self._check_shapes(h=h, r=r, t=t, h_prefix="n")

        # prepare input to generic score function
        h = self._add_dim(*h, dim=self.BATCH_DIM)
        r = self._add_dim(*r, dim=self.NUM_DIM)
        t = self._add_dim(*t, dim=self.NUM_DIM)

        # get scores
        scores = self(h=h, r=r, t=t)

        # prepare output shape, (batch_size, num_heads, num_relations, num_tails) -> (batch_size, num_heads)
        return self._remove_dim(scores, self.RELATION_DIM, self.TAIL_DIM)

    def score_r(
        self,
        h: HeadRepresentation = tuple(),
        all_relations: RelationRepresentation = tuple(),
        t: TailRepresentation = tuple(),
    ) -> torch.FloatTensor:
        """
        Score all relations.

        :param h: shape: (batch_size, d_e)
            The head representations.
        :param all_relations: shape: (num_relations, d_r)
            The relation representations.
        :param t: shape: (batch_size, d_e)
            The tail representations.

        :return: shape: (batch_size, num_entities)
            The scores.
        """
        h, r, t = _ensure_tuple(h, all_relations, t)
        assert self._check_shapes(h=h, r=r, t=t, r_prefix="n")

        # prepare input to generic score function
        h = self._add_dim(*h, dim=self.NUM_DIM)
        r = self._add_dim(*r, dim=self.BATCH_DIM)
        t = self._add_dim(*t, dim=self.NUM_DIM)

        # get scores
        scores = self(h=h, r=r, t=t)

        # prepare output shape
        return self._remove_dim(scores, self.HEAD_DIM, self.TAIL_DIM)

    def score_t(
        self,
        h: HeadRepresentation = tuple(),
        r: RelationRepresentation = tuple(),
        all_entities: TailRepresentation = tuple(),
    ) -> torch.FloatTensor:
        """
        Score all tail entities.

        :param h: shape: (batch_size, d_e)
            The head representations.
        :param r: shape: (batch_size, d_r)
            The relation representations.
        :param all_entities: shape: (num_entities, d_e)
            The tail representations.

        :return: shape: (batch_size, num_entities)
            The scores.
        """
        h, r, t = _ensure_tuple(h, r, all_entities)
        assert self._check_shapes(h=h, r=r, t=t, t_prefix="n")

        # prepare input to generic score function
        h = self._add_dim(*h, dim=self.NUM_DIM)
        r = self._add_dim(*r, dim=self.NUM_DIM)
        t = self._add_dim(*t, dim=self.BATCH_DIM)

        # get scores
        scores = self(h=h, r=r, t=t)

        # prepare output shape
        return self._remove_dim(scores, self.HEAD_DIM, self.RELATION_DIM)

    def reset_parameters(self):
        """Reset parameters the interaction function may have."""
        for mod in self.modules():
            if mod is self:
                continue
            if hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()


class StatelessInteractionFunction(InteractionFunction[HeadRepresentation, RelationRepresentation, TailRepresentation]):
    """Interaction function without state."""

    def __init__(self, f: Callable[..., torch.FloatTensor]):
        super().__init__()
        self.f = f

    def forward(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> torch.FloatTensor:  # noqa: D102
        # normalization
        h, r, t = _ensure_tuple(h, r, t)
        return self.f(*h, *r, *t)


class TranslationalInteractionFunction(InteractionFunction[HeadRepresentation, RelationRepresentation, TailRepresentation], ABC):
    """The translational interaction function shared by the TransE, TransR, TransH, and other Trans<X> models."""

    def __init__(self, p: int, power_norm: bool = False):
        """Initialize the translational interaction function.

        :param p:
            The norm used with :func:`torch.norm`. Typically is 1 or 2.
        :param power_norm:
            Whether to use the p-th power of the L_p norm. It has the advantage of being differentiable around 0,
            and numerically more stable.
        """
        super().__init__()
        self.p = p
        self.power_norm = power_norm


class TransEInteractionFunction(TranslationalInteractionFunction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
    """The TransE interaction function."""

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa:D102
        return pkf.transe_interaction(h=h, r=r, t=t, p=self.p, power_norm=self.power_norm)


class ComplExInteractionFunction(StatelessInteractionFunction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
    """Interaction function of ComplEx."""

    def __init__(self):
        super().__init__(f=pkf.complex_interaction)


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


class ConvEInteractionFunction(InteractionFunction[torch.FloatTensor, torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]):
    """ConvE interaction function."""

    tail_entity_shape = ("d", "k")  # with k=1

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
        t: Tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> torch.FloatTensor:  # noqa: D102
        # get tail bias term
        t, t_bias = t
        return pkf.conve_interaction(
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


class ConvKBInteractionFunction(InteractionFunction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
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
    ) -> torch.FloatTensor:  # noqa: D102
        return pkf.convkb_interaction(
            h=h,
            r=r,
            t=t,
            conv=self.conv,
            activation=self.activation,
            hidden_dropout=self.hidden_dropout,
            linear=self.linear,
        )


class DistMultInteractionFunction(StatelessInteractionFunction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
    """Interaction function of DistMult."""

    def __init__(self):
        super().__init__(f=pkf.distmult_interaction)


class ERMLPInteractionFunction(InteractionFunction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
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
    ) -> torch.FloatTensor:  # noqa: D102
        return pkf.ermlp_interaction(
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


class ERMLPEInteractionFunction(InteractionFunction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
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
    ) -> torch.FloatTensor:  # noqa: D102
        return pkf.ermlpe_interaction(h=h, r=r, t=t, mlp=self.mlp)


class TransRInteractionFunction(TranslationalInteractionFunction[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]):
    """The TransR interaction function."""

    relation_shape = ("e", "de")

    def __init__(self, p: int, power_norm: bool = True):
        super().__init__(p=p, power_norm=power_norm)

    def forward(
        self,
        h: torch.FloatTensor,
        r: Tuple[torch.FloatTensor, torch.FloatTensor],
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa:D102
        r, m_r = r
        return pkf.transr_interaction(h=h, r=r, t=t, m_r=m_r, p=self.p, power_norm=self.power_norm)


class RotatEInteractionFunction(StatelessInteractionFunction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
    """Interaction function of RotatE."""

    def __init__(self):
        super().__init__(f=pkf.rotate_interaction)


class HolEInteractionFunction(StatelessInteractionFunction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
    """Interaction function for HolE."""

    def __init__(self):
        super().__init__(f=pkf.hole_interaction)


class ProjEInteractionFunction(InteractionFunction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
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
    ) -> torch.FloatTensor:  # noqa:D102
        return pkf.proje_interaction(
            h=h, r=r, t=t,
            d_e=self.d_e, d_r=self.d_r, b_c=self.b_c, b_p=self.b_p, activation=self.inner_non_linearity,
        )


class RESCALInteractionFunction(StatelessInteractionFunction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
    """Interaction function of RESCAL."""

    relation_shape = ("dd",)

    def __init__(self):
        super().__init__(f=pkf.rescal_interaction)


class StructuredEmbeddingInteractionFunction(TranslationalInteractionFunction[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]):
    """Interaction function of Structured Embedding."""

    relation_shape = ("dd", "dd")

    def forward(
        self,
        h: torch.FloatTensor,
        r: Tuple[torch.FloatTensor, torch.FloatTensor],
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa:D102
        rh, rt = r
        return pkf.structured_embedding_interaction(h=h, r_h=rh, r_t=rt, t=t, p=self.p, power_norm=self.power_norm)


class TuckerInteractionFunction(InteractionFunction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
    """Interaction function of Tucker."""

    def __init__(
        self,
        embedding_dim: int = 200,
        relation_dim: Optional[int] = None,
        dropout_0: float = 0.3,
        dropout_1: float = 0.4,
        dropout_2: float = 0.5,
        apply_batch_normalization: bool = True,
    ):
        """Initialize the Tucker interaction function.

        :param embedding_dim:
        :param relation_dim:
        :param dropout_0:
        :param dropout_1:
        :param dropout_2:
        :param apply_batch_normalization:
        """
        super().__init__()

        if relation_dim is None:
            relation_dim = embedding_dim

        # Core tensor
        # Note: we use a different dimension permutation as in the official implementation to match the paper.
        self.core_tensor = nn.Parameter(
            torch.empty(embedding_dim, relation_dim, embedding_dim),
            requires_grad=True,
        )

        # Dropout
        self.input_dropout = nn.Dropout(dropout_0)
        self.hidden_dropout_1 = nn.Dropout(dropout_1)
        self.hidden_dropout_2 = nn.Dropout(dropout_2)

        if apply_batch_normalization:
            self.bn1 = nn.BatchNorm1d(embedding_dim)
            self.bn2 = nn.BatchNorm1d(embedding_dim)
        else:
            self.bn1 = self.bn2 = None

    def reset_parameters(self):  # noqa:D102
        # Initialize core tensor, cf. https://github.com/ibalazevic/TuckER/blob/master/model.py#L12
        nn.init.uniform_(self.core_tensor, -1., 1.)

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa:D102
        return pkf.tucker_interaction(
            h=h,
            r=r,
            t=t,
            core_tensor=self.core_tensor,
            do0=self.input_dropout,
            do1=self.hidden_dropout_1,
            do2=self.hidden_dropout_2,
            bn1=self.bn1,
            bn2=self.bn2,
        )


class UnstructuredModelInteractionFunction(TranslationalInteractionFunction[torch.FloatTensor, None, torch.FloatTensor]):
    """Interaction function of UnstructuredModel."""

    # shapes
    relation_shape = tuple()

    def __init__(self, p: int, power_norm: bool = True):
        super().__init__(p=p, power_norm=power_norm)

    def forward(
        self,
        h: torch.FloatTensor,
        r: None,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa:D102
        return pkf.unstructured_model_interaction(h, t, p=self.p, power_norm=self.power_norm)


class TransDInteractionFunction(
    TranslationalInteractionFunction[
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
    ]
):
    """Interaction function of TransD."""

    entity_shape = ("d", "d")
    relation_shape = ("e", "e")

    def forward(
        self,
        h: Tuple[torch.FloatTensor, torch.FloatTensor],
        r: Tuple[torch.FloatTensor, torch.FloatTensor],
        t: Tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> torch.FloatTensor:  # noqa:D102
        h, h_p = h
        r, r_p = r
        t, t_p = t
        return pkf.transd_interaction(h=h, r=r, t=t, h_p=h_p, r_p=r_p, t_p=t_p, p=self.p, power_norm=self.power_norm)


class NTNInteractionFunction(
    InteractionFunction[
        torch.FloatTensor,
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
        torch.FloatTensor,
    ]
):
    """The interaction function of NTN."""

    relation_shape = ("kdd", "k", "k", "kd", "kd")

    def __init__(
        self,
        non_linearity: Optional[nn.Module] = None,
    ):
        super().__init__()
        if non_linearity is None:
            non_linearity = nn.Tanh()
        self.non_linearity = non_linearity

    def forward(
        self,
        h: torch.FloatTensor,
        r: Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        w, b, u, vh, vt = r
        return pkf.ntn_interaction(h=h, t=t, w=w, b=b, u=u, vh=vh, vt=vt, activation=self.non_linearity)


class KG2EInteractionFunction(
    InteractionFunction[
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
    ]
):
    """Interaction function of KG2E."""

    entity_shape = ("d", "d")
    relation_shape = ("d", "d")

    def __init__(
        self,
        similarity: str = "KL",
        exact: bool = True,
    ):
        super().__init__()
        similarity = similarity.upper()
        if similarity not in KG2E_SIMILARITIES:
            raise ValueError(similarity)
        self.similarity = similarity
        self.exact = exact

    def forward(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> torch.FloatTensor:
        h_mean, h_var = h
        r_mean, r_var = r
        t_mean, t_var = t
        return pkf.kg2e_interaction(
            h_mean=h_mean,
            h_var=h_var,
            r_mean=r_mean,
            r_var=r_var,
            t_mean=t_mean,
            t_var=t_var,
            similarity=self.similarity,
            exact=self.exact,
        )


class TransHInteractionFunction(
    TranslationalInteractionFunction[
        torch.FloatTensor,
        Tuple[torch.FloatTensor, torch.FloatTensor],
        torch.FloatTensor,
    ]
):
    """Interaction function of TransH."""

    def forward(
        self,
        h: torch.FloatTensor,
        r: Tuple[torch.FloatTensor, torch.FloatTensor],
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        w_r, d_r = r
        return pkf.transh_interaction(h, w_r, d_r, t, p=self.scoring_fct_norm)
