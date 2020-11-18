# -*- coding: utf-8 -*-

"""Stateful interaction functions."""

import itertools
import logging
import math
from abc import ABC
from typing import Any, Callable, Generic, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch
from torch import FloatTensor, nn

from . import functional as pkf
from ..typing import HeadRepresentation, RelationRepresentation, Representation, TailRepresentation
from ..utils import check_shapes

__all__ = [
    # Base Classes
    'Interaction',
    'TranslationalInteraction',
    # Concrete Classes
    'ComplExInteraction',
    'ConvEInteraction',
    'ConvKBInteraction',
    'DistMultInteraction',
    'ERMLPInteraction',
    'ERMLPEInteraction',
    'HolEInteraction',
    'KG2EInteraction',
    'NTNInteraction',
    'ProjEInteraction',
    'RESCALInteraction',
    'RotatEInteraction',
    'StructuredEmbeddingInteraction',
    'TransDInteraction',
    'TransEInteraction',
    'TransHInteraction',
    'TransRInteraction',
    'TuckerInteraction',
    'UnstructuredModelInteraction',
]

logger = logging.getLogger(__name__)


def _upgrade_to_sequence(x: Union[FloatTensor, Sequence[FloatTensor]]) -> Sequence[FloatTensor]:
    return x if isinstance(x, Sequence) else (x,)


def _ensure_tuple(*x: Union[Representation, Sequence[Representation]]) -> Sequence[Sequence[Representation]]:
    return tuple(_upgrade_to_sequence(xx) for xx in x)


def _unpack_singletons(*xs: Tuple) -> Sequence[Tuple]:
    return [
        x[0] if len(x) == 1 else x
        for x in xs
    ]


def _get_prefix(slice_size, slice_dim, d) -> str:
    if slice_size is None or slice_dim != d:
        return 'b'
    else:
        return 'n'


def _get_batches(z, slice_size):
    for batch in zip(*(hh.split(slice_size, dim=1) for hh in _ensure_tuple(z)[0])):
        if len(batch) == 1:
            batch = batch[0]
        yield batch


class Interaction(nn.Module, Generic[HeadRepresentation, RelationRepresentation, TailRepresentation], ABC):
    """Base class for interaction functions."""

    # Dimensions
    BATCH_DIM: int = 0
    NUM_DIM: int = 1
    HEAD_DIM: int = 1
    RELATION_DIM: int = 2
    TAIL_DIM: int = 3

    #: The symbolic shapes for entity representations
    entity_shape: Sequence[str] = ("d",)

    #: The symbolic shapes for entity representations for tail entities, if different. This is ony relevant for ConvE.
    tail_entity_shape: Optional[Sequence[str]] = None

    #: The symbolic shapes for relation representations
    relation_shape: Sequence[str] = ("d",)

    #: The functional interaction form
    func: Callable[..., torch.FloatTensor]

    def _prepare_hrt_for_functional(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:
        """Conversion utility to prepare the h/r/t representations for the functional form."""
        assert all(torch.is_tensor(x) for x in (h, r, t))
        return dict(h=h, r=r, t=t)

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:
        """Conversion utility to prepare the state to be passed to the functional form."""
        return dict()

    def _prepare_for_functional(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> Mapping[str, torch.FloatTensor]:
        """Conversion utility to prepare the arguments for the functional form."""
        kwargs = self._prepare_hrt_for_functional(h=h, r=r, t=t)
        kwargs.update(self._prepare_state_for_functional())
        return kwargs

    def forward(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> torch.FloatTensor:
        """Compute broadcasted triple scores given representations for head, relation and tails.

        :param h: shape: (batch_size, num_heads, ``*``)
            The head representations.
        :param r: shape: (batch_size, num_relations, ``*``)
            The relation representations.
        :param t: shape: (batch_size, num_tails, ``*``)
            The tail representations.

        :return: shape: (batch_size, num_heads, num_relations, num_tails)
            The scores.
        """
        return self.__class__.func(**self._prepare_for_functional(h=h, r=r, t=t))

    @staticmethod
    def _add_dim(*x: torch.FloatTensor, dim: int) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
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
        return tuple(out)

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
        dims = tuple(d if d >= 0 else len(x.shape) + d for d in dims)
        if len(set(dims)) != len(dims):
            raise ValueError(f"Duplicate dimensions: {dims}")
        assert all(0 <= d < len(x.shape) for d in dims)
        for dim in sorted(dims, reverse=True):
            x = x.squeeze(dim=dim)
        return x

    def _check_shapes(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
        h_prefix: str = "b",
        r_prefix: str = "b",
        t_prefix: str = "b",
        raise_on_errors: bool = True,
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
            if raise_on_errors:
                raise ValueError
            return False
        if len(r) != len(relation_shape):
            if raise_on_errors:
                raise ValueError
            return False
        if len(t) != len(tail_entity_shape):
            if raise_on_errors:
                raise ValueError
            return False

        # TODO make helper function + unit test
        a = ((hh, h_prefix + hs) for hh, hs in zip(h, entity_shape))  # type: ignore
        b = ((rr, r_prefix + rs) for rr, rs in zip(r, relation_shape))  # type: ignore
        c = ((tt, t_prefix + ts) for tt, ts in zip(t, tail_entity_shape))  # type: ignore

        return check_shapes(
            *itertools.chain(a, b, c),
            raise_on_errors=raise_on_errors,
        )

    def score(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
        slice_size: Optional[int] = None,
        slice_dim: Optional[str] = None,
    ) -> torch.FloatTensor:
        """
        Compute broadcasted triple scores with optional slicing.

        .. note ::
            At most one of the slice sizes may be not None.

        :param h: shape: (batch_size, num_heads, ``*``)
            The head representations.
        :param r: shape: (batch_size, num_relations, ``*``)
            The relation representations.
        :param t: shape: (batch_size, num_tails, ``*``)
            The tail representations.
        :param slice_size:
            The slice size.
        :param slice_dim:
            The dimension along which to slice. From {"h", "r", "t"}

        :return: shape: (batch_size, num_heads, num_relations, num_tails)
            The scores.
        """
        return self._forward_slicing_wrapper(h=h, r=r, t=t, slice_size=slice_size, slice_dim=slice_dim)

    def _score(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
        h_prefix: str = "b",
        r_prefix: str = "b",
        t_prefix: str = "b",
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        assert {h_prefix, r_prefix, t_prefix}.issubset(list("bn"))
        # at most one of h_prefix, r_prefix, t_prefix equals n
        slice_dims: List[str] = [
            dim
            for dim, prefix in zip("hrt", (h_prefix, r_prefix, t_prefix))
            if prefix == "n"
        ]
        slice_dim: Optional[str] = slice_dims[0] if len(slice_dims) == 1 else None

        # FIXME typing does not work well for this
        h = _upgrade_to_sequence(h)
        r = _upgrade_to_sequence(r)
        t = _upgrade_to_sequence(t)
        assert self._check_shapes(h=h, r=r, t=t, h_prefix=h_prefix, r_prefix=r_prefix, t_prefix=t_prefix)

        # prepare input to generic score function: bh*, br*, bt*
        h = self._add_dim(*h, dim=self.BATCH_DIM if h_prefix == "n" else self.NUM_DIM)
        r = self._add_dim(*r, dim=self.BATCH_DIM if r_prefix == "n" else self.NUM_DIM)
        t = self._add_dim(*t, dim=self.BATCH_DIM if t_prefix == "n" else self.NUM_DIM)

        scores = self._forward_slicing_wrapper(h=h, r=r, t=t, slice_dim=slice_dim, slice_size=slice_size)

        remove_dims = [
            dim
            for dim, prefix in zip(
                (self.HEAD_DIM, self.RELATION_DIM, self.TAIL_DIM),
                (h_prefix, r_prefix, t_prefix),
            )
            if prefix == "b"
        ]
        # prepare output shape
        return self._remove_dim(scores, *remove_dims)

    def _forward_slicing_wrapper(
        self,
        h: Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]],
        r: Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]],
        t: Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]],
        slice_size: Optional[int],
        slice_dim: Optional[str],
    ) -> torch.FloatTensor:
        """
        Compute broadcasted triple scores with optional slicing for representations in canonical shape.

        .. note ::
            Depending on the interaction function, there may be more than one representation for h/r/t. In that case,
            a tuple of at least two tensors is passed.

        :param h: shape: (batch_size, num_heads, ``*``)
            The head representations.
        :param r: shape: (batch_size, num_relations, ``*``)
            The relation representations.
        :param t: shape: (batch_size, num_tails, ``*``)
            The tail representations.
        :param slice_size:
            The slice size.
        :param slice_dim:
            The dimension along which to slice. From {"h", "r", "t"}

        :return: shape: (batch_size, num_heads, num_relations, num_tails)
            The scores.
        """
        if slice_size is None:
            scores = self(h=h, r=r, t=t)
        elif slice_dim == "h":
            scores = torch.cat([
                self(h=h_batch, r=r, t=t)
                for h_batch in _get_batches(h, slice_size)
            ], dim=self.HEAD_DIM)
        elif slice_dim == "r":
            scores = torch.cat([
                self(h=h, r=r_batch, t=t)
                for r_batch in _get_batches(r, slice_size)
            ], dim=self.RELATION_DIM)
        elif slice_dim == "t":
            scores = torch.cat([
                self(h=h, r=r, t=t_batch)
                for t_batch in _get_batches(t, slice_size)
            ], dim=self.TAIL_DIM)
        else:
            raise ValueError(f'Invalid slice_dim: {slice_dim}')
        return scores

    def score_hrt(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
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
        return self._score(h=h, r=r, t=t).unsqueeze(dim=-1)

    def score_h(
        self,
        all_entities: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """
        Score all head entities.

        :param all_entities: shape: (num_entities, d_e)
            The head representations.
        :param r: shape: (batch_size, d_r)
            The relation representations.
        :param t: shape: (batch_size, d_e)
            The tail representations.
        :param slice_size:
            The slice size.

        :return: shape: (batch_size, num_entities)
            The scores.
        """
        return self._score(h=all_entities, r=r, t=t, h_prefix="n", slice_size=slice_size)

    def score_r(
        self,
        h: HeadRepresentation,
        all_relations: RelationRepresentation,
        t: TailRepresentation,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """
        Score all relations.

        :param h: shape: (batch_size, d_e)
            The head representations.
        :param all_relations: shape: (num_relations, d_r)
            The relation representations.
        :param t: shape: (batch_size, d_e)
            The tail representations.
        :param slice_size:
            The slice size.

        :return: shape: (batch_size, num_entities)
            The scores.
        """
        return self._score(h=h, r=all_relations, t=t, r_prefix="n", slice_size=slice_size)

    def score_t(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        all_entities: TailRepresentation,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """
        Score all tail entities.

        :param h: shape: (batch_size, d_e)
            The head representations.
        :param r: shape: (batch_size, d_r)
            The relation representations.
        :param all_entities: shape: (num_entities, d_e)
            The tail representations.
        :param slice_size:
            The slice size.

        :return: shape: (batch_size, num_entities)
            The scores.
        """
        return self._score(h=h, r=r, t=all_entities, t_prefix="n", slice_size=slice_size)

    def reset_parameters(self):
        """Reset parameters the interaction function may have."""
        for mod in self.modules():
            if mod is self:
                continue
            if hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()


class TranslationalInteraction(Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation], ABC):
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


class TransEInteraction(TranslationalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """The TransE interaction function."""

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa:D102
        return pkf.transe_interaction(h=h, r=r, t=t, p=self.p, power_norm=self.power_norm)


class ComplExInteraction(Interaction[FloatTensor, FloatTensor, FloatTensor]):
    """Interaction function of ComplEx."""

    func = pkf.complex_interaction


def _calculate_missing_shape_information(
    embedding_dim: int,
    input_channels: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Automatically calculates missing dimensions for ConvE.

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

    # All are None -> try and make closest to square
    if input_channels is None and width is None and height is None:
        input_channels = 1
        result_sqrt = math.floor(math.sqrt(embedding_dim))
        height = max(factor for factor in range(1, result_sqrt + 1) if embedding_dim % factor == 0)
        width = embedding_dim // height
    # Only input channels is None
    elif input_channels is None and width is not None and height is not None:
        input_channels = embedding_dim // (width * height)
    # Only width is None
    elif input_channels is not None and width is None and height is not None:
        width = embedding_dim // (height * input_channels)
    # Only height is none
    elif height is None and width is not None and input_channels is not None:
        height = embedding_dim // (width * input_channels)
    # Width and input_channels are None -> set input_channels to 1 and calculage height
    elif input_channels is None and height is None and width is not None:
        input_channels = 1
        height = embedding_dim // width
    # Width and input channels are None -> set input channels to 1 and calculate width
    elif input_channels is None and height is not None and width is None:
        input_channels = 1
        width = embedding_dim // height

    if input_channels * width * height != embedding_dim:  # type: ignore
        raise ValueError(f'Could not resolve {original} to a valid factorization of {embedding_dim}.')

    return input_channels, width, height  # type: ignore


class ConvEInteraction(Interaction[torch.FloatTensor, torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]):
    """ConvE interaction function."""

    tail_entity_shape = ("d", "k")  # with k=1

    #: The head-relation encoder operating on 2D "images"
    hr2d: nn.Module

    #: The head-relation encoder operating on the 1D flattened version
    hr1d: nn.Module

    #: The interaction function
    func = pkf.conve_interaction

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

        if input_channels * embedding_height * embedding_width != embedding_dim:
            raise ValueError(
                f'Product of input channels ({input_channels}), height ({embedding_height}), and width '
                f'({embedding_width}) does not equal target embedding dimension ({embedding_dim})',
            )

        # encoders
        # 1: 2D encoder: BN?, DO, Conv, BN?, Act, DO
        hr2d_layers = [
            nn.BatchNorm2d(input_channels) if apply_batch_normalization else None,
            nn.Dropout(input_dropout),
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_height, kernel_width),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(output_channels) if apply_batch_normalization else None,
            nn.ReLU(),
            nn.Dropout2d(feature_map_dropout),
        ]
        self.hr2d = nn.Sequential(*(layer for layer in hr2d_layers if layer is not None))

        # 2: 1D encoder: FC, DO, BN?, Act
        num_in_features = (
            output_channels
            * (2 * embedding_height - kernel_height + 1)
            * (embedding_width - kernel_width + 1)
        )
        hr1d_layers = [
            nn.Linear(num_in_features, embedding_dim),
            nn.Dropout(output_dropout),
            nn.BatchNorm1d(embedding_dim) if apply_batch_normalization else None,
            nn.ReLU(),
        ]
        self.hr1d = nn.Sequential(*(layer for layer in hr1d_layers if layer is not None))

        # store reshaping dimensions
        self.embedding_height = embedding_height
        self.embedding_width = embedding_width
        self.input_channels = input_channels

    def _prepare_hrt_for_functional(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(h=h, r=r, t=t[0], t_bias=t[1])

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(
            input_channels=self.input_channels,
            embedding_height=self.embedding_height,
            embedding_width=self.embedding_width,
            hr2d=self.hr2d,
            hr1d=self.hr1d,
        )


class ConvKBInteraction(Interaction[FloatTensor, FloatTensor, FloatTensor]):
    """Interaction function of ConvKB.

    .. seealso:: :func:`pykeen.nn.functional.convkb_interaction``
    """

    func = pkf.convkb_interaction

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

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(
            conv=self.conv,
            activation=self.activation,
            hidden_dropout=self.hidden_dropout,
            linear=self.linear,
        )


class DistMultInteraction(Interaction[FloatTensor, FloatTensor, FloatTensor]):
    """A module wrapping the DistMult interaction function at :func:`pykeen.nn.functional.distmult_interaction`."""

    func = pkf.distmult_interaction


class ERMLPInteraction(Interaction[FloatTensor, FloatTensor, FloatTensor]):
    """A module wrapping the ER-MLP interaction function from :func:`pykeen.nn.functional.ermlp_interaction`.

    .. math ::
        f(h, r, t) = W_2 ReLU(W_1 cat(h, r, t) + b_1) + b_2
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
    ):
        """Initialize the interaction function.

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


class ERMLPEInteraction(Interaction[FloatTensor, FloatTensor, FloatTensor]):
    """Interaction function of ER-MLP (E)."""

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


class TransRInteraction(
    TranslationalInteraction[
        torch.FloatTensor,
        Tuple[torch.FloatTensor, torch.FloatTensor],
        torch.FloatTensor,
    ],
):
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


class RotatEInteraction(Interaction[FloatTensor, FloatTensor, FloatTensor]):
    """Interaction function of RotatE."""

    func = pkf.rotate_interaction


class HolEInteraction(Interaction[FloatTensor, FloatTensor, FloatTensor]):
    """Interaction function for HolE."""

    func = pkf.hole_interaction


class ProjEInteraction(Interaction[FloatTensor, FloatTensor, FloatTensor]):
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


class RESCALInteraction(Interaction[FloatTensor, FloatTensor, FloatTensor]):
    """Interaction function of RESCAL."""

    relation_shape = ("dd",)
    func = pkf.rescal_interaction


class StructuredEmbeddingInteraction(
    TranslationalInteraction[
        torch.FloatTensor,
        Tuple[torch.FloatTensor, torch.FloatTensor],
        torch.FloatTensor,
    ],
):
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


class TuckerInteraction(Interaction[FloatTensor, FloatTensor, FloatTensor]):
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


class UnstructuredModelInteraction(
    TranslationalInteraction[torch.FloatTensor, None, torch.FloatTensor],
):
    """Interaction function of UnstructuredModel."""

    # shapes
    relation_shape: Sequence[str] = tuple()

    def __init__(self, p: int, power_norm: bool = True):
        super().__init__(p=p, power_norm=power_norm)

    def forward(
        self,
        h: torch.FloatTensor,
        r: None,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa:D102
        return pkf.unstructured_model_interaction(h, t, p=self.p, power_norm=self.power_norm)


class TransDInteraction(
    TranslationalInteraction[
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
    ],
):
    """Interaction function of TransD."""

    entity_shape = ("d", "d")
    relation_shape = ("e", "e")

    def __init__(self, p: int = 2, power_norm: bool = True):
        super().__init__(p=p, power_norm=power_norm)

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


class NTNInteraction(
    Interaction[
        torch.FloatTensor,
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
        torch.FloatTensor,
    ],
):
    """The interaction function of NTN."""

    relation_shape = ("kdd", "kd", "kd", "k", "k")

    def __init__(
        self,
        non_linearity: Optional[nn.Module] = None,
    ):
        super().__init__()
        if non_linearity is None:
            non_linearity = nn.Tanh()
        self.non_linearity = non_linearity

    @staticmethod
    def unpack(h, r, t) -> Mapping[str, torch.FloatTensor]:
        """Unpack (h, r, t) to arguments for functional ntn_interaction."""
        w, vh, vt, b, u = r
        return dict(h=h, t=t, w=w, b=b, u=u, vh=vh, vt=vt)

    def forward(
        self,
        h: torch.FloatTensor,
        r: Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa:D102
        return pkf.ntn_interaction(**self.unpack(h=h, r=r, t=t), activation=self.non_linearity)


class KG2EInteraction(
    Interaction[
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
    ],
):
    """Interaction function of KG2E."""

    entity_shape = ("d", "d")
    relation_shape = ("d", "d")
    similarity: str
    exact: bool

    def __init__(
        self,
        similarity: Optional[str] = None,
        exact: bool = True,
    ):
        super().__init__()
        if similarity is None:
            similarity = 'KL'
        self.similarity = similarity
        self.exact = exact

    def forward(
        self,
        h: Tuple[torch.FloatTensor, torch.FloatTensor],
        r: Tuple[torch.FloatTensor, torch.FloatTensor],
        t: Tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> torch.FloatTensor:  # noqa:D102
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


class TransHInteraction(TranslationalInteraction[FloatTensor, Tuple[FloatTensor, FloatTensor], FloatTensor]):
    """Interaction function of TransH."""

    relation_shape = ("d", "d")

    def forward(
        self,
        h: torch.FloatTensor,
        r: Tuple[torch.FloatTensor, torch.FloatTensor],
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        w_r, d_r = r
        return pkf.transh_interaction(h, w_r, d_r, t, p=self.p, power_norm=self.power_norm)
