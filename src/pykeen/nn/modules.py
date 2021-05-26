# -*- coding: utf-8 -*-

"""Stateful interaction functions."""

from __future__ import annotations

import itertools as itt
import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union, cast

import torch
from class_resolver import Resolver
from torch import FloatTensor, nn

from . import functional as pkf
from .combinations import Combination
from ..typing import HeadRepresentation, HintOrType, RelationRepresentation, TailRepresentation
from ..utils import (
    CANONICAL_DIMENSIONS, activation_resolver, convert_to_canonical_shape, ensure_tuple, upgrade_to_sequence,
)

__all__ = [
    'interaction_resolver',
    # Base Classes
    'Interaction',
    'FunctionalInteraction',
    'LiteralInteraction',
    'TranslationalInteraction',
    # Adapter classes
    'MonotonicAffineTransformationInteraction',
    # Concrete Classes
    'ComplExInteraction',
    'ConvEInteraction',
    'ConvKBInteraction',
    'CrossEInteraction',
    'DistMultInteraction',
    'ERMLPInteraction',
    'ERMLPEInteraction',
    'HolEInteraction',
    'KG2EInteraction',
    'MuREInteraction',
    'NTNInteraction',
    'PairREInteraction',
    'ProjEInteraction',
    'RESCALInteraction',
    'RotatEInteraction',
    'SimplEInteraction',
    'StructuredEmbeddingInteraction',
    'TransDInteraction',
    'TransEInteraction',
    'TransHInteraction',
    'TransRInteraction',
    'TuckerInteraction',
    'UnstructuredModelInteraction',
]

logger = logging.getLogger(__name__)


def _get_batches(z, slice_size):
    for batch in zip(*(hh.split(slice_size, dim=1) for hh in ensure_tuple(z)[0])):
        if len(batch) == 1:
            batch = batch[0]
        yield batch


class Interaction(nn.Module, Generic[HeadRepresentation, RelationRepresentation, TailRepresentation], ABC):
    """Base class for interaction functions."""

    #: The symbolic shapes for entity representations
    entity_shape: Sequence[str] = ("d",)

    #: The symbolic shapes for entity representations for tail entities, if different. This is ony relevant for ConvE.
    tail_entity_shape: Optional[Sequence[str]] = None

    #: The symbolic shapes for relation representations
    relation_shape: Sequence[str] = ("d",)

    @classmethod
    def get_dimensions(cls) -> Set[str]:
        """Get all of the relevant dimension keys.

        This draws from :data:`Interaction.entity_shape`, :data:`Interaction.relation_shape`, and in the case of
        :class:`ConvEInteraction`, the :data:`Interaction.tail_entity_shape`.

        :returns: a set of strings representting the dimension keys.
        """
        return set(itt.chain(cls.entity_shape, cls.tail_entity_shape or set(), cls.relation_shape))

    @abstractmethod
    def forward(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> torch.FloatTensor:
        """Compute broadcasted triple scores given broadcasted representations for head, relation and tails.

        :param h: shape: (batch_size, num_heads, 1, 1, ``*``)
            The head representations.
        :param r: shape: (batch_size, 1, num_relations, 1, ``*``)
            The relation representations.
        :param t: shape: (batch_size, 1, 1, num_tails, ``*``)
            The tail representations.

        :return: shape: (batch_size, num_heads, num_relations, num_tails)
            The scores.
        """

    def score(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
        slice_size: Optional[int] = None,
        slice_dim: Optional[str] = None,
    ) -> torch.FloatTensor:
        """Compute broadcasted triple scores with optional slicing.

        .. note ::
            At most one of the slice sizes may be not None.

        :param h: shape: (batch_size, num_heads, `1, 1, `*``)
            The head representations.
        :param r: shape: (batch_size, 1, num_relations, 1, ``*``)
            The relation representations.
        :param t: shape: (batch_size, 1, 1, num_tails, ``*``)
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
        slice_size: Optional[int] = None,
        slice_dim: str = None,
    ) -> torch.FloatTensor:
        """Compute scores for the score_* methods outside of models.

        TODO: merge this with the Model utilities?

        :param h: shape: (b, h, *)
        :param r: shape: (b, r, *)
        :param t: shape: (b, t, *)
        :param slice_size:
            The slice size.
        :param slice_dim:
            The dimension along which to slice. From {"h", "r", "t"}
        :return: shape: (b, h, r, t)
        """
        args = []
        for key, x in zip("hrt", (h, r, t)):
            value = []
            for xx in upgrade_to_sequence(x):  # type: torch.FloatTensor
                # bring to (b, n, *)
                xx = xx.unsqueeze(dim=1 if key != slice_dim else 0)
                # bring to (b, h, r, t, *)
                xx = convert_to_canonical_shape(
                    x=xx,
                    dim=key,
                    num=xx.shape[1],
                    batch_size=xx.shape[0],
                    suffix_shape=xx.shape[2:],
                )
                value.append(xx)
            # unpack singleton
            if len(value) == 1:
                value = value[0]
            args.append(value)
        h, r, t = cast(Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation], args)
        return self._forward_slicing_wrapper(h=h, r=r, t=t, slice_dim=slice_dim, slice_size=slice_size)

    def _forward_slicing_wrapper(
        self,
        h: Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]],
        r: Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]],
        t: Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]],
        slice_size: Optional[int],
        slice_dim: Optional[str],
    ) -> torch.FloatTensor:
        """Compute broadcasted triple scores with optional slicing for representations in canonical shape.

        .. note ::
            Depending on the interaction function, there may be more than one representation for h/r/t. In that case,
            a tuple of at least two tensors is passed.

        :param h: shape: (batch_size, num_heads, 1, 1, ``*``)
            The head representations.
        :param r: shape: (batch_size, 1, num_relations, 1, ``*``)
            The relation representations.
        :param t: shape: (batch_size, 1, 1, num_tails, ``*``)
            The tail representations.
        :param slice_size:
            The slice size.
        :param slice_dim:
            The dimension along which to slice. From {"h", "r", "t"}

        :return: shape: (batch_size, num_heads, num_relations, num_tails)
            The scores.

        :raises ValueError:
            If slice_dim is invalid.
        """
        if slice_size is None:
            scores = self(h=h, r=r, t=t)
        elif slice_dim == "h":
            scores = torch.cat([
                self(h=h_batch, r=r, t=t)
                for h_batch in _get_batches(h, slice_size)
            ], dim=CANONICAL_DIMENSIONS[slice_dim])
        elif slice_dim == "r":
            scores = torch.cat([
                self(h=h, r=r_batch, t=t)
                for r_batch in _get_batches(r, slice_size)
            ], dim=CANONICAL_DIMENSIONS[slice_dim])
        elif slice_dim == "t":
            scores = torch.cat([
                self(h=h, r=r, t=t_batch)
                for t_batch in _get_batches(t, slice_size)
            ], dim=CANONICAL_DIMENSIONS[slice_dim])
        else:
            raise ValueError(f'Invalid slice_dim: {slice_dim}')
        return scores

    def score_hrt(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> torch.FloatTensor:
        """Score a batch of triples.

        :param h: shape: (batch_size, d_e)
            The head representations.
        :param r: shape: (batch_size, d_r)
            The relation representations.
        :param t: shape: (batch_size, d_e)
            The tail representations.

        :return: shape: (batch_size, 1)
            The scores.
        """
        return self._score(h=h, r=r, t=t)[:, 0, 0, 0, None]

    def score_h(
        self,
        all_entities: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """Score all head entities.

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
        return self._score(h=all_entities, r=r, t=t, slice_dim="h", slice_size=slice_size)[:, :, 0, 0]

    def score_r(
        self,
        h: HeadRepresentation,
        all_relations: RelationRepresentation,
        t: TailRepresentation,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """Score all relations.

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
        return self._score(h=h, r=all_relations, t=t, slice_dim="r", slice_size=slice_size)[:, 0, :, 0]

    def score_t(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        all_entities: TailRepresentation,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """Score all tail entities.

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
        return self._score(h=h, r=r, t=all_entities, slice_dim="t", slice_size=slice_size)[:, 0, 0, :]

    def reset_parameters(self):
        """Reset parameters the interaction function may have."""
        for mod in self.modules():
            if mod is self:
                continue
            if hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()


class LiteralInteraction(
    Interaction,
    Generic[HeadRepresentation, RelationRepresentation, TailRepresentation],
):
    """The interaction function shared by literal-containing interactions."""

    def __init__(
        self,
        base: HintOrType[Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation]],
        combination: Combination,
        base_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """Instantiate the module.

        :param combination: The module used to concatenate the literals to the entity representations
        :param base: The interaction module
        :param base_kwargs: Keyword arguments for the interaction module
        """
        super().__init__()
        self.base = interaction_resolver.make(base, base_kwargs)
        self.combination = combination
        # The appended "e" represents the literals that get concatenated
        # on the entity representations. It does not necessarily have the
        # same dimension "d" as the entity representations.
        self.entity_shape = tuple(self.base.entity_shape) + ("e",)

    def forward(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> torch.FloatTensor:
        """Compute broadcasted triple scores given broadcasted representations for head, relation and tails.

        :param h: shape: (batch_size, num_heads, 1, 1, ``*``)
            The head representations.
        :param r: shape: (batch_size, 1, num_relations, 1, ``*``)
            The relation representations.
        :param t: shape: (batch_size, 1, 1, num_tails, ``*``)
            The tail representations.

        :return: shape: (batch_size, num_heads, num_relations, num_tails)
            The scores.
        """
        # alternate way of combining entity embeddings + literals
        # h = torch.cat(h, dim=-1)
        # h = self.combination(h.view(-1, h.shape[-1])).view(*h.shape[:-1], -1)  # type: ignore
        # t = torch.cat(t, dim=-1)
        # t = self.combination(t.view(-1, t.shape[-1])).view(*t.shape[:-1], -1)  # type: ignore
        h_proj = self.combination(*h)
        t_proj = self.combination(*t)
        return self.base(h=h_proj, r=r, t=t_proj)


class FunctionalInteraction(Interaction, Generic[HeadRepresentation, RelationRepresentation, TailRepresentation]):
    """Base class for interaction functions."""

    #: The functional interaction form
    func: Callable[..., torch.FloatTensor]

    def forward(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> torch.FloatTensor:
        """Compute broadcasted triple scores given broadcasted representations for head, relation and tails.

        :param h: shape: (batch_size, num_heads, 1, 1, ``*``)
            The head representations.
        :param r: shape: (batch_size, 1, num_relations, 1, ``*``)
            The relation representations.
        :param t: shape: (batch_size, 1, 1, num_tails, ``*``)
            The tail representations.

        :return: shape: (batch_size, num_heads, num_relations, num_tails)
            The scores.
        """
        return self.__class__.func(**self._prepare_for_functional(h=h, r=r, t=t))

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

    @staticmethod
    def _prepare_hrt_for_functional(
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


class TranslationalInteraction(
    FunctionalInteraction,
    Generic[HeadRepresentation, RelationRepresentation, TailRepresentation],
    ABC,
):
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

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(p=self.p, power_norm=self.power_norm)


class TransEInteraction(TranslationalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A stateful module for the TransE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.transe_interaction`
    """

    func = pkf.transe_interaction


class ComplExInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A module wrapper for the stateless ComplEx interaction function.

    .. seealso:: :func:`pykeen.nn.functional.complex_interaction`
    """

    func = pkf.complex_interaction


def _calculate_missing_shape_information(
    embedding_dim: int,
    input_channels: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Automatically calculates missing dimensions for ConvE.

    :param embedding_dim:
        The embedding dimension.
    :param input_channels:
        The number of input channels for the convolution.
    :param width:
        The width of the embedding "image".
    :param height:
        The height of the embedding "image".

    :return: (input_channels, width, height), such that
            `embedding_dim = input_channels * width * height`

    :raises ValueError:
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


class ConvEInteraction(
    FunctionalInteraction[torch.FloatTensor, torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]],
):
    """A stateful module for the ConvE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.conve_interaction`
    """

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

    @staticmethod
    def _prepare_hrt_for_functional(
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


class ConvKBInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A stateful module for the ConvKB interaction function.

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


class DistMultInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A module wrapper for the stateless DistMult interaction function.

    .. seealso:: :func:`pykeen.nn.functional.distmult_interaction`
    """

    func = pkf.distmult_interaction


class ERMLPInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A stateful module for the ER-MLP interaction.

    .. seealso:: :func:`pykeen.nn.functional.ermlp_interaction`

    .. math ::
        f(h, r, t) = W_2 ReLU(W_1 cat(h, r, t) + b_1) + b_2
    """

    func = pkf.ermlp_interaction

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

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(
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


class ERMLPEInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A stateful module for the ER-MLP (E) interaction function.

    .. seealso:: :func:`pykeen.nn.functional.ermlpe_interaction`
    """

    func = pkf.ermlpe_interaction

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

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(mlp=self.mlp)


class TransRInteraction(
    TranslationalInteraction[
        torch.FloatTensor,
        Tuple[torch.FloatTensor, torch.FloatTensor],
        torch.FloatTensor,
    ],
):
    """A stateful module for the TransR interaction function.

    .. seealso:: :func:`pykeen.nn.functional.transr_interaction`
    """

    relation_shape = ("e", "de")
    func = pkf.transr_interaction

    def __init__(self, p: int, power_norm: bool = True):
        super().__init__(p=p, power_norm=power_norm)

    @staticmethod
    def _prepare_hrt_for_functional(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(h=h, r=r[0], t=t, m_r=r[1])


class RotatEInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A module wrapper for the stateless RotatE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.rotate_interaction`
    """

    func = pkf.rotate_interaction


class HolEInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A module wrapper for the stateless HolE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.hole_interaction`
    """

    func = pkf.hole_interaction


class ProjEInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A stateful module for the ProjE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.proje_interaction`
    """

    func = pkf.proje_interaction

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

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:
        return dict(d_e=self.d_e, d_r=self.d_r, b_c=self.b_c, b_p=self.b_p, activation=self.inner_non_linearity)


class RESCALInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A module wrapper for the stateless RESCAL interaction function.

    .. seealso:: :func:`pykeen.nn.functional.rescal_interaction`
    """

    relation_shape = ("dd",)
    func = pkf.rescal_interaction


class StructuredEmbeddingInteraction(
    TranslationalInteraction[
        torch.FloatTensor,
        Tuple[torch.FloatTensor, torch.FloatTensor],
        torch.FloatTensor,
    ],
):
    """A stateful module for the Structured Embedding (SE) interaction function.

    .. seealso:: :func:`pykeen.nn.functional.structured_embedding_interaction`
    """

    relation_shape = ("dd", "dd")
    func = pkf.structured_embedding_interaction

    @staticmethod
    def _prepare_hrt_for_functional(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(h=h, t=t, r_h=r[0], r_t=r[1])


class TuckerInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A stateful module for the stateless Tucker interaction function.

    .. seealso:: :func:`pykeen.nn.functional.tucker_interaction`
    """

    func = pkf.tucker_interaction

    def __init__(
        self,
        embedding_dim: int = 200,
        relation_dim: Optional[int] = None,
        head_dropout: float = 0.3,
        relation_dropout: float = 0.4,
        head_relation_dropout: float = 0.5,
        apply_batch_normalization: bool = True,
    ):
        """Initialize the Tucker interaction function.

        :param embedding_dim:
            The entity embedding dimension.
        :param relation_dim:
            The relation embedding dimension.
        :param head_dropout:
            The dropout rate applied to the head representations.
        :param relation_dropout:
            The dropout rate applied to the relation representations.
        :param head_relation_dropout:
            The dropout rate applied to the combined head and relation representations.
        :param apply_batch_normalization:
            Whether to use batch normalization on head representations and the combination of head and relation.
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
        self.head_dropout = nn.Dropout(head_dropout)
        self.relation_dropout = nn.Dropout(relation_dropout)
        self.head_relation_dropout = nn.Dropout(head_relation_dropout)

        if apply_batch_normalization:
            self.head_batch_norm = nn.BatchNorm1d(embedding_dim)
            self.head_relation_batch_norm = nn.BatchNorm1d(embedding_dim)
        else:
            self.head_batch_norm = self.head_relation_batch_norm = None

    def reset_parameters(self):  # noqa:D102
        # Initialize core tensor, cf. https://github.com/ibalazevic/TuckER/blob/master/model.py#L12
        nn.init.uniform_(self.core_tensor, -1., 1.)
        # batch norm gets reset automatically, since it defines reset_parameters

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:
        return dict(
            core_tensor=self.core_tensor,
            do_h=self.head_dropout,
            do_r=self.relation_dropout,
            do_hr=self.head_relation_dropout,
            bn_h=self.head_batch_norm,
            bn_hr=self.head_relation_batch_norm,
        )


class UnstructuredModelInteraction(
    TranslationalInteraction[torch.FloatTensor, None, torch.FloatTensor],
):
    """A stateful module for the UnstructuredModel interaction function.

    .. seealso:: :func:`pykeen.nn.functional.unstructured_model_interaction`
    """

    # shapes
    relation_shape: Sequence[str] = tuple()

    func = pkf.unstructured_model_interaction

    def __init__(self, p: int, power_norm: bool = True):
        super().__init__(p=p, power_norm=power_norm)

    @staticmethod
    def _prepare_hrt_for_functional(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(h=h, t=t)


class TransDInteraction(
    TranslationalInteraction[
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
    ],
):
    """A stateful module for the TransD interaction function.

    .. seealso:: :func:`pykeen.nn.functional.transd_interaction`
    """

    entity_shape = ("d", "d")
    relation_shape = ("e", "e")
    func = pkf.transd_interaction

    def __init__(self, p: int = 2, power_norm: bool = True):
        super().__init__(p=p, power_norm=power_norm)

    @staticmethod
    def _prepare_hrt_for_functional(
        h: Tuple[torch.FloatTensor, torch.FloatTensor],
        r: Tuple[torch.FloatTensor, torch.FloatTensor],
        t: Tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        h, h_p = h
        r, r_p = r
        t, t_p = t
        return dict(h=h, r=r, t=t, h_p=h_p, r_p=r_p, t_p=t_p)


class NTNInteraction(
    FunctionalInteraction[
        torch.FloatTensor,
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
        torch.FloatTensor,
    ],
):
    """A stateful module for the NTN interaction function.

    .. seealso:: :func:`pykeen.nn.functional.ntn_interaction`
    """

    relation_shape = ("kdd", "kd", "kd", "k", "k")
    func = pkf.ntn_interaction

    def __init__(self, non_linearity: Optional[nn.Module] = None):
        super().__init__()
        if non_linearity is None:
            non_linearity = nn.Tanh()
        self.non_linearity = non_linearity

    @staticmethod
    def _prepare_hrt_for_functional(
        h: torch.FloatTensor,
        r: Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
        t: torch.FloatTensor,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        w, vh, vt, b, u = r
        return dict(h=h, t=t, w=w, b=b, u=u, vh=vh, vt=vt)

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(activation=self.non_linearity)


class KG2EInteraction(
    FunctionalInteraction[
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
    ],
):
    """A stateful module for the KG2E interaction function.

    .. seealso:: :func:`pykeen.nn.functional.kg2e_interaction`
    """

    entity_shape = ("d", "d")
    relation_shape = ("d", "d")
    similarity: str
    exact: bool
    func = pkf.kg2e_interaction

    def __init__(self, similarity: Optional[str] = None, exact: bool = True):
        super().__init__()
        if similarity is None:
            similarity = 'KL'
        self.similarity = similarity
        self.exact = exact

    @staticmethod
    def _prepare_hrt_for_functional(
        h: Tuple[torch.FloatTensor, torch.FloatTensor],
        r: Tuple[torch.FloatTensor, torch.FloatTensor],
        t: Tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> MutableMapping[str, torch.FloatTensor]:
        h_mean, h_var = h
        r_mean, r_var = r
        t_mean, t_var = t
        return dict(
            h_mean=h_mean,
            h_var=h_var,
            r_mean=r_mean,
            r_var=r_var,
            t_mean=t_mean,
            t_var=t_var,
        )

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:
        return dict(
            similarity=self.similarity,
            exact=self.exact,
        )


class TransHInteraction(TranslationalInteraction[FloatTensor, Tuple[FloatTensor, FloatTensor], FloatTensor]):
    """A stateful module for the TransH interaction function.

    .. seealso:: :func:`pykeen.nn.functional.transh_interaction`
    """

    relation_shape = ("d", "d")
    func = pkf.transh_interaction

    @staticmethod
    def _prepare_hrt_for_functional(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(h=h, w_r=r[0], d_r=r[1], t=t)


class MuREInteraction(
    TranslationalInteraction[
        Tuple[FloatTensor, FloatTensor, FloatTensor],
        Tuple[FloatTensor, FloatTensor],
        Tuple[FloatTensor, FloatTensor, FloatTensor],
    ],
):
    """A stateful module for the MuRE interaction function from [balazevic2019b]_.

    .. seealso:: :func:`pykeen.nn.functional.mure_interaction`
    """

    # there are separate biases for entities in head and tail position
    entity_shape = ("d", "", "")
    relation_shape = ("d", "d")
    func = pkf.mure_interaction

    @staticmethod
    def _prepare_hrt_for_functional(
        h: Tuple[FloatTensor, FloatTensor, FloatTensor],
        r: Tuple[FloatTensor, FloatTensor],
        t: Tuple[FloatTensor, FloatTensor, FloatTensor],
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        h, b_h, _ = h
        t, _, b_t = t
        r_vec, r_mat = r
        return dict(h=h, b_h=b_h, r_vec=r_vec, r_mat=r_mat, t=t, b_t=b_t)


class SimplEInteraction(
    FunctionalInteraction[
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor],
    ],
):
    """A module wrapper for the SimplE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.simple_interaction`
    """

    func = pkf.simple_interaction
    entity_shape = ("d", "d")
    relation_shape = ("d", "d")

    def __init__(self, clamp_score: Union[None, float, Tuple[float, float]] = None):
        super().__init__()
        if isinstance(clamp_score, float):
            clamp_score = (-clamp_score, clamp_score)
        self.clamp_score = clamp_score

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(clamp=self.clamp_score)

    @staticmethod
    def _prepare_hrt_for_functional(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(h=h[0], h_inv=h[1], r=r[0], r_inv=r[1], t=t[0], t_inv=t[1])


class PairREInteraction(TranslationalInteraction[FloatTensor, Tuple[FloatTensor, FloatTensor], FloatTensor]):
    """A stateful module for the PairRE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.pair_re_interaction`
    """

    relation_shape = ("d", "d")
    func = pkf.pair_re_interaction

    @staticmethod
    def _prepare_hrt_for_functional(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(h=h, r_h=r[0], r_t=r[1], t=t)


class QuatEInteraction(
    FunctionalInteraction[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ],
):
    """A module wrapper for the QuatE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.quat_e_interaction`
    """

    func = pkf.quat_e_interaction


class MonotonicAffineTransformationInteraction(
    Interaction[
        HeadRepresentation,
        RelationRepresentation,
        TailRepresentation,
    ],
):
    r"""
    An adapter of interaction functions which adds a scalar (trainable) monotonic affine transformation of the score.

    .. math ::
        score(h, r, t) = \alpha \cdot score'(h, r, t) + \beta

    This adapter is useful for losses such as BCE, where there is a fixed decision threshold, or margin-based losses,
    where the margin is not be treated as hyper-parameter, but rather a trainable parameter. This is particularly
    useful, if the value range of the score function is not known in advance, and thus choosing an appropriate margin
    becomes difficult.

    Monotonicity is required to preserve the ordering of the original scoring function, and thus ensures that more
    plausible triples are still more plausible after the transformation.

    For example, we can add a bias to a distance-based interaction function to enable positive values:

    >>> base = TransEInteraction(p=2)
    >>> interaction = MonotonicAffineTransformationInteraction(base=base, trainable_bias=True, trainable_scale=False)

    When combined with BCE loss, we can geometrically think about predicting a (soft) sphere at :math:`h + r` with
    radius equal to the bias of the transformation. When we add a trainable scale, the model can control the "softness"
    of the decision boundary itself.
    """

    def __init__(
        self,
        base: Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation],
        initial_bias: float = 0.0,
        trainable_bias: bool = True,
        initial_scale: float = 1.0,
        trainable_scale: bool = True,
    ):
        """
        Initialize the interaction.

        :param base:
            The base interaction.
        :param initial_bias:
            The initial value for the bias.
        :param trainable_bias:
            Whether the bias should be trainable.
        :param initial_scale: >0
            The initial value for the scale. Must be strictly positive.
        :param trainable_scale:
            Whether the scale should be trainable.
        """
        super().__init__()

        # the base interaction
        self.base = base
        # forward entity/relation shapes
        self.entity_shape = base.entity_shape
        self.relation_shape = base.relation_shape
        self.tail_entity_shape = base.tail_entity_shape

        # The parameters of the affine transformation: bias
        self.bias = nn.Parameter(torch.empty(size=tuple()), requires_grad=trainable_bias)
        self.initial_bias = torch.as_tensor(data=[initial_bias], dtype=torch.get_default_dtype())

        # scale. We model this as log(scale) to ensure scale > 0, and thus monotonicity
        self.log_scale = nn.Parameter(torch.empty(size=tuple()), requires_grad=trainable_scale)
        self.initial_log_scale = torch.as_tensor(data=[math.log(initial_scale)], dtype=torch.get_default_dtype())

    def reset_parameters(self):  # noqa: D102
        self.bias.data = self.initial_bias.to(device=self.bias.device)
        self.log_scale.data = self.initial_log_scale.to(device=self.bias.device)

    def forward(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> torch.FloatTensor:  # noqa: D102
        return self.log_scale.exp() * self.base(h=h, r=r, t=t) + self.bias


class CrossEInteraction(FunctionalInteraction[FloatTensor, Tuple[FloatTensor, FloatTensor], FloatTensor]):
    """A module wrapper for the CrossE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.cross_e_interaction`
    """

    func = pkf.cross_e_interaction
    relation_shape = ("d", "d")

    def __init__(
        self,
        embedding_dim: int = 50,
        combination_activation: HintOrType[nn.Module] = nn.Tanh,
        combination_activation_kwargs: Optional[Mapping[str, Any]] = None,
        combination_dropout: Optional[float] = 0.5,
    ):
        """
        Instantiate the interaction module.

        :param embedding_dim:
            The embedding dimension.
        :param combination_activation:
            The combination activation function.
        :param combination_activation_kwargs:
            Additional keyword-based arguments passed to the constructor of the combination activation function (if
            not already instantiated).
        :param combination_dropout:
            An optional dropout applied to the combination.
        """
        super().__init__()
        self.combination_activation = activation_resolver.make(
            combination_activation,
            pos_kwargs=combination_activation_kwargs,
        )
        self.combination_bias = nn.Parameter(data=torch.zeros(1, 1, 1, 1, embedding_dim))
        self.combination_dropout = nn.Dropout(combination_dropout) if combination_dropout else None

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(
            bias=self.combination_bias,
            activation=self.combination_activation,
            dropout=self.combination_dropout,
        )

    @staticmethod
    def _prepare_hrt_for_functional(
        h: FloatTensor,
        r: Tuple[FloatTensor, FloatTensor],
        t: FloatTensor,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        r, c_r = r
        return dict(h=h, r=r, c_r=c_r, t=t)


interaction_resolver = Resolver.from_subclasses(
    Interaction,  # type: ignore
    skip={TranslationalInteraction, FunctionalInteraction, MonotonicAffineTransformationInteraction},
    suffix=Interaction.__name__,
)
