"""Stateful interaction functions."""

from __future__ import annotations

import itertools as itt
import logging
import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, MutableMapping, Sequence
from operator import itemgetter
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    cast,
)

import more_itertools
import numpy
import torch
from class_resolver import ClassResolver, Hint, OptionalKwargs
from class_resolver.contrib.torch import activation_resolver
from docdata import parse_docdata
from torch import FloatTensor, nn
from torch.nn.init import xavier_normal_

from . import functional as pkf
from .algebra import quaterion_multiplication_table
from .init import initializer_resolver
from ..metrics.utils import ValueRange
from ..typing import (
    HeadRepresentation,
    HintOrType,
    Initializer,
    RelationRepresentation,
    Representation,
    Sign,
    TailRepresentation,
)
from ..utils import (
    at_least_eps,
    einsum,
    ensure_complex,
    ensure_tuple,
    estimate_cost_of_sequence,
    negative_norm,
    unpack_singletons,
    upgrade_to_sequence,
)

# TODO: split file into multiple smaller ones?

__all__ = [
    "interaction_resolver",
    # Base Classes
    "Interaction",
    "FunctionalInteraction",
    "NormBasedInteraction",
    # Adapter classes
    "MonotonicAffineTransformationInteraction",
    # Concrete Classes
    "AutoSFInteraction",
    "BoxEInteraction",
    "ComplExInteraction",
    "ConvEInteraction",
    "ConvKBInteraction",
    "CPInteraction",
    "CrossEInteraction",
    "DistMAInteraction",
    "DistMultInteraction",
    "ERMLPEInteraction",
    "ERMLPInteraction",
    "HolEInteraction",
    "KG2EInteraction",
    "LineaREInteraction",
    "MultiLinearTuckerInteraction",
    "MuREInteraction",
    "NTNInteraction",
    "PairREInteraction",
    "ProjEInteraction",
    "QuatEInteraction",
    "RESCALInteraction",
    "RotatEInteraction",
    "SEInteraction",
    "SimplEInteraction",
    "TorusEInteraction",
    "TransDInteraction",
    "TransEInteraction",
    "TransFInteraction",
    "TransformerInteraction",
    "TransHInteraction",
    "TransRInteraction",
    "TripleREInteraction",
    "TuckerInteraction",
    "UMInteraction",
]

logger = logging.getLogger(__name__)


def parallel_slice_batches(
    *representations: Representation,
    split_size: int,
    dim: int,
) -> Iterable[Sequence[Representation]]:
    """
    Slice representations along the given dimension.

    :param representations:
        the representations to slice
    :param split_size:
        the slice size
    :param dim:
        the dimension along which to slice

    :yields: batches of sliced representations
    """
    # normalize input
    rs: Sequence[Sequence[torch.FloatTensor]] = ensure_tuple(*representations)
    # get number of head/relation/tail representations
    length = list(map(len, rs))
    splits = numpy.cumsum([0] + length)
    # flatten list
    rsl: Sequence[torch.FloatTensor] = sum(map(list, rs), [])
    # split tensors
    parts = [r.split(split_size, dim=dim) for r in rsl]
    # broadcasting
    n_parts = max(map(len, parts))
    parts = [r_parts if len(r_parts) == n_parts else r_parts * n_parts for r_parts in parts]
    # yield batches
    for batch in zip(*parts):
        # complex typing
        yield unpack_singletons(*(batch[start:stop] for start, stop in zip(splits, splits[1:])))  # type: ignore


def parallel_unsqueeze(x: Representation, dim: int) -> Representation:
    """Unsqueeze all representations along the given dimension."""
    xs: Sequence[torch.FloatTensor] = upgrade_to_sequence(x)
    xs = [xx.unsqueeze(dim=dim) for xx in xs]
    return cast(Representation, xs[0] if len(xs) == 1 else xs)


class Interaction(nn.Module, Generic[HeadRepresentation, RelationRepresentation, TailRepresentation], ABC):
    """Base class for interaction functions."""

    #: The symbolic shapes for entity representations
    entity_shape: Sequence[str] = ("d",)

    #: The symbolic shapes for entity representations for tail entities, if different.
    #: Otherwise, the entity_shape is used for head & tail entities
    _tail_entity_shape: Sequence[str] | None = None

    #: The symbolic shapes for relation representations
    relation_shape: Sequence[str] = ("d",)

    # if the interaction function's head parameter should only receive a subset of entity representations
    _head_indices: Sequence[int] | None = None

    # if the interaction function's tail parameter should only receive a subset of entity representations
    _tail_indices: Sequence[int] | None = None

    #: the interaction's value range (for unrestricted input)
    value_range: ClassVar[ValueRange] = ValueRange()

    # TODO: annotate modelling capabilities? cf., e.g., https://arxiv.org/abs/1902.10197, Table 2
    # TODO: annotate properties, e.g., symmetry, and use them for testing?
    # TODO: annotate complexity?
    #: whether the interaction is defined on complex input
    is_complex: ClassVar[bool] = False

    @property
    def tail_entity_shape(self) -> Sequence[str]:
        """Return the symbolic shape for tail entity representations."""
        if self._tail_entity_shape is None:
            return self.entity_shape
        return self._tail_entity_shape

    def head_indices(self) -> Sequence[int]:
        """Return the entity representation indices used for the head representations."""
        if self._head_indices is None:
            return list(range(len(self.entity_shape)))
        return self._head_indices

    def tail_indices(self) -> Sequence[int]:
        """Return the entity representation indices used for the tail representations."""
        if self._tail_indices is None:
            return list(range(len(self.tail_entity_shape)))
        return self._tail_indices

    def full_entity_shapes(self) -> Sequence[str]:
        """Return all entity shapes (head & tail)."""
        shapes: list[str | None] = [None] * (max(itt.chain(self.head_indices(), self.tail_indices())) + 1)
        for hi, hs in zip(self.head_indices(), self.entity_shape):
            shapes[hi] = hs
        for ti, ts in zip(self.tail_indices(), self.tail_entity_shape):
            if shapes[ti] is not None and ts != shapes[ti]:
                raise ValueError("Shape conflict.")
            shapes[ti] = ts
        if None in shapes:
            raise AssertionError("Unused shape.")
        return cast(list[str], shapes)

    @classmethod
    def get_dimensions(cls) -> set[str]:
        """Get all of the relevant dimension keys.

        This draws from :data:`Interaction.entity_shape`, :data:`Interaction.relation_shape`, and in the case of
        :class:`ConvEInteraction`, the :data:`Interaction.tail_entity_shape`.

        :returns: a set of strings representting the dimension keys.
        """
        # TODO: cannot cover dynamic shapes, e.g., AutoSF
        return set(itt.chain(cls.entity_shape, cls._tail_entity_shape or set(), cls.relation_shape))

    @abstractmethod
    def forward(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> torch.FloatTensor:
        """Compute broadcasted triple scores given broadcasted representations for head, relation and tails.

        :param h: shape: (`*batch_dims`, `*dims`)
            The head representations.
        :param r: shape: (`*batch_dims`, `*dims`)
            The relation representations.
        :param t: shape: (`*batch_dims`, `*dims`)
            The tail representations.

        :return: shape: batch_dims
            The scores.
        """

    def score(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
        slice_size: int | None = None,
        slice_dim: int = 1,
    ) -> torch.FloatTensor:
        """Compute broadcasted triple scores with optional slicing.

        .. note ::
            At most one of the slice sizes may be not None.

        # TODO: we could change that to slicing along multiple dimensions, if necessary

        :param h: shape: (`*batch_dims`, `*dims`)
            The head representations.
        :param r: shape: (`*batch_dims`, `*dims`)
            The relation representations.
        :param t: shape: (`*batch_dims`, `*dims`)
            The tail representations.
        :param slice_size:
            The slice size.
        :param slice_dim:
            The dimension along which to slice. From {0, ..., len(batch_dims)}

        :return: shape: batch_dims
            The scores.
        """
        if slice_size is None:
            return self(h=h, r=r, t=t)

        return torch.cat(
            [
                self(h=h_batch, r=r_batch, t=t_batch)
                for h_batch, r_batch, t_batch in parallel_slice_batches(h, r, t, split_size=slice_size, dim=slice_dim)
            ],
            dim=slice_dim,
        )

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
        return self.score(h=h, r=r, t=t).unsqueeze(dim=-1)

    def score_h(
        self,
        all_entities: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
        slice_size: int | None = None,
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
        return self.score(
            h=parallel_unsqueeze(all_entities, dim=0),
            r=parallel_unsqueeze(r, dim=1),
            t=parallel_unsqueeze(t, dim=1),
            slice_size=slice_size,
        )

    def score_r(
        self,
        h: HeadRepresentation,
        all_relations: RelationRepresentation,
        t: TailRepresentation,
        slice_size: int | None = None,
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
        return self.score(
            h=parallel_unsqueeze(h, dim=1),
            r=parallel_unsqueeze(all_relations, dim=0),
            t=parallel_unsqueeze(t, dim=1),
            slice_size=slice_size,
        )

    def score_t(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        all_entities: TailRepresentation,
        slice_size: int | None = None,
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
        return self.score(
            h=parallel_unsqueeze(h, dim=1),
            r=parallel_unsqueeze(r, dim=1),
            t=parallel_unsqueeze(all_entities, dim=0),
            slice_size=slice_size,
        )

    def reset_parameters(self):
        """Reset parameters the interaction function may have."""
        for mod in self.modules():
            if mod is self:
                continue
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()


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

        :param h: shape: (`*batch_dims`, `*dims`)
            The head representations.
        :param r: shape: (`*batch_dims`, `*dims`)
            The relation representations.
        :param t: shape: (`*batch_dims`, `*dims`)
            The tail representations.

        :return: shape: batch_dims
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

    # docstr-coverage: inherited
    @classmethod
    def _prepare_hrt_for_functional(
        cls,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        """Conversion utility to prepare the h/r/t representations for the functional form."""
        # TODO: we only allow single-tensor representations here, but could easily generalize
        assert all(torch.is_tensor(x) for x in (h, r, t))
        if cls.is_complex:
            h, r, t = ensure_complex(h, r, t)
        return dict(h=h, r=r, t=t)

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:
        """Conversion utility to prepare the state to be passed to the functional form."""
        return dict()


class NormBasedInteraction(
    FunctionalInteraction,
    Generic[HeadRepresentation, RelationRepresentation, TailRepresentation],
    ABC,
):
    """Norm-based interactions use a (powered) $p$-norm in their scoring function."""

    def __init__(self, p: int, power_norm: bool = False):
        """Initialize the norm-based interaction function.

        :param p:
            The norm used with :func:`torch.linalg.vector_norm`. Typically is 1 or 2.
        :param power_norm:
            Whether to use the p-th power of the $L_p$ norm. It has the advantage of being differentiable around 0,
            and numerically more stable.
        """
        super().__init__()
        self.p = p
        self.power_norm = power_norm

    # docstr-coverage: inherited
    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(p=self.p, power_norm=self.power_norm)


@parse_docdata
class TransEInteraction(NormBasedInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A stateful module for the TransE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.transe_interaction`

    ---
    citation:
        author: Bordes
        year: 2013
        link: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
    """

    func = pkf.transe_interaction


@parse_docdata
class TransFInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A stateless module for the TransF interaction function.

    .. seealso:: :func:`pykeen.nn.functional.transf_interaction`

    ---
    citation:
        author: Feng
        year: 2016
        link: https://www.aaai.org/ocs/index.php/KR/KR16/paper/view/12887
    """

    func = pkf.transf_interaction


@parse_docdata
class ComplExInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    r"""The ComplEx interaction proposed by [trouillon2016]_.

    ComplEx operates on complex-valued entity and relation representations, i.e.,
    $\textbf{e}_i, \textbf{r}_i \in \mathbb{C}^d$ and calculates the plausibility score via the Hadamard product:

    .. math::

        f(h,r,t) =  Re(\mathbf{e}_h\odot\mathbf{r}_r\odot\bar{\mathbf{e}}_t)

    Which expands to:

    .. math::

        f(h,r,t) = \left\langle Re(\mathbf{e}_h),Re(\mathbf{r}_r),Re(\mathbf{e}_t)\right\rangle
        + \left\langle Im(\mathbf{e}_h),Re(\mathbf{r}_r),Im(\mathbf{e}_t)\right\rangle
        + \left\langle Re(\mathbf{e}_h),Im(\mathbf{r}_r),Im(\mathbf{e}_t)\right\rangle
        - \left\langle Im(\mathbf{e}_h),Im(\mathbf{r}_r),Re(\mathbf{e}_t)\right\rangle

    where $Re(\textbf{x})$ and $Im(\textbf{x})$ denote the real and imaginary parts of the complex valued vector
    $\textbf{x}$. Because the Hadamard product is not commutative in the complex space, ComplEx can model
    anti-symmetric relations in contrast to DistMult.

    .. seealso ::

        Official implementation: https://github.com/ttrouill/complex/

    .. note::
        this method generally expects all tensors to be of complex datatype, i.e., `torch.is_complex(x)` to evaluate to
        `True`. However, for backwards compatibility and convenience in use, you can also pass real tensors whose shape
        is compliant with :func:`torch.view_as_complex`, cf. :func:`pykeen.utils.ensure_complex`.

    ---
    citation:
        arxiv: 1606.06357
        author: Trouillon
        github: ttrouill/complex
        link: https://arxiv.org/abs/1606.06357
        year: 2016
    """

    is_complex: ClassVar[bool] = True

    # TODO: update class docstring

    # TODO: give this a better name?
    @staticmethod
    def func(h: FloatTensor, r: FloatTensor, t: FloatTensor) -> FloatTensor:
        r"""Evaluate the interaction function.

        :param h: shape: (`*batch_dims`, dim)
            The complex head representations.
        :param r: shape: (`*batch_dims`, dim)
            The complex relation representations.
        :param t: shape: (`*batch_dims`, dim)
            The complex tail representations.

        :return: shape: batch_dims
            The scores.
        """
        return torch.real(einsum("...d, ...d, ...d -> ...", h, r, torch.conj(t)))


def _calculate_missing_shape_information(
    embedding_dim: int,
    input_channels: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> tuple[int, int, int]:
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
        raise ValueError(f"Could not resolve {original} to a valid factorization of {embedding_dim}.")

    return input_channels, width, height  # type: ignore


@parse_docdata
class ConvEInteraction(
    FunctionalInteraction[torch.FloatTensor, torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor]],
):
    """A stateful module for the ConvE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.conve_interaction`
    ---
    citation:
        author: Dettmers
        year: 2018
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17366
        github: TimDettmers/ConvE
        arxiv: 1707.01476
    """

    # vector & scalar offset
    tail_entity_shape = ("d", "")

    #: The head-relation encoder operating on 2D "images"
    hr2d: nn.Module

    #: The head-relation encoder operating on the 1D flattened version
    hr1d: nn.Module

    #: The interaction function
    func = pkf.conve_interaction

    def __init__(
        self,
        input_channels: int | None = None,
        output_channels: int = 32,
        embedding_height: int | None = None,
        embedding_width: int | None = None,
        kernel_width: int = 3,
        kernel_height: int | None = None,
        input_dropout: float = 0.2,
        feature_map_dropout: float = 0.2,
        output_dropout: float = 0.3,
        embedding_dim: int = 200,
        apply_batch_normalization: bool = True,
    ):
        """
        Initialize the interaction module.

        :param input_channels:
            the number of input channels for the convolution operation. Can be inferred from other parameters,
            cf. :func:`_calculate_missing_shape_information`.
        :param output_channels:
            the number of input channels for the convolution operation
        :param embedding_height:
            the height of the "image" after reshaping the concatenated head and relation embedding. Can be inferred
            from other parameters, cf. :func:`_calculate_missing_shape_information`.
        :param embedding_width:
            the width of the "image" after reshaping the concatenated head and relation embedding. Can be inferred
            from other parameters, cf. :func:`_calculate_missing_shape_information`.
        :param kernel_width:
            the width of the convolution kernel
        :param kernel_height:
            the height of the convolution kernel. Defaults to `kernel_width`
        :param input_dropout:
            the dropout applied *before* the convolution
        :param feature_map_dropout:
            the dropout applied *after* the convolution
        :param output_dropout:
            the dropout applied after the linear projection
        :param embedding_dim:
            the embedding dimension of entities and relations
        :param apply_batch_normalization:
            whether to apply batch normalization
        """
        super().__init__()

        # Automatic calculation of remaining dimensions
        logger.info(f"Resolving {input_channels} * {embedding_width} * {embedding_height} = {embedding_dim}.")
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
        logger.info(f"Resolved to {input_channels} * {embedding_width} * {embedding_height} = {embedding_dim}.")

        # normalize kernel height
        kernel_height = kernel_height or kernel_width

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
            output_channels * (2 * embedding_height - kernel_height + 1) * (embedding_width - kernel_width + 1)
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

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(h=h, r=r, t=t[0], t_bias=t[1])

    # docstr-coverage: inherited
    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(
            input_channels=self.input_channels,
            embedding_height=self.embedding_height,
            embedding_width=self.embedding_width,
            hr2d=self.hr2d,
            hr1d=self.hr1d,
        )


@parse_docdata
class ConvKBInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A stateful module for the ConvKB interaction function.

    .. seealso:: :func:`pykeen.nn.functional.convkb_interaction``

    ---
    citation:
        author: Nguyen
        year: 2018
        link: https://www.aclweb.org/anthology/N18-2053
        github: daiquocnguyen/ConvKB
        arxiv: 1712.02121
    """

    func = pkf.convkb_interaction

    def __init__(
        self,
        hidden_dropout_rate: float = 0.0,
        embedding_dim: int = 200,
        num_filters: int = 400,
    ):
        """
        Initialize the interaction module.

        :param hidden_dropout_rate:
            the dropout rate applied on the hidden layer
        :param embedding_dim:
            the entity and relation embedding dimension
        :param num_filters:
            the number of filters (=output channels) of the convolution
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters

        # The interaction model
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(1, 3), bias=True)
        self.activation = nn.ReLU()
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_rate)
        self.linear = nn.Linear(embedding_dim * num_filters, 1, bias=True)

    # docstr-coverage: inherited
    def reset_parameters(self):  # noqa: D102
        # Use Xavier initialization for weight; bias to zero
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.linear.bias)

        # Initialize all filters to [0.1, 0.1, -0.1],
        #  c.f. https://github.com/daiquocnguyen/ConvKB/blob/master/model.py#L34-L36
        nn.init.constant_(self.conv.weight[..., :2], 0.1)
        nn.init.constant_(self.conv.weight[..., 2], -0.1)
        nn.init.zeros_(self.conv.bias)

    # docstr-coverage: inherited
    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(
            conv=self.conv,
            activation=self.activation,
            hidden_dropout=self.hidden_dropout,
            linear=self.linear,
        )


@parse_docdata
class DistMultInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A module wrapper for the stateless DistMult interaction function.

    .. seealso:: :func:`pykeen.nn.functional.distmult_interaction`

    ---
    citation:
        author: Yang
        year: 2014
        link: https://arxiv.org/abs/1412.6575
        arxiv: 1412.6575
    """

    func = pkf.distmult_interaction


@parse_docdata
class DistMAInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A module wrapper for the stateless DistMA interaction function.

    .. seealso:: :func:`pykeen.nn.functional.dist_ma_interaction`

    ---
    citation:
        author: Shi
        year: 2019
        link: https://www.aclweb.org/anthology/D19-1075.pdf
    """

    func = pkf.dist_ma_interaction


@parse_docdata
class ERMLPInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A stateful module for the ER-MLP interaction.

    .. seealso:: :func:`pykeen.nn.functional.ermlp_interaction`

    .. math ::
        f(h, r, t) = W_2 ReLU(W_1 cat(h, r, t) + b_1) + b_2

    ---
    name: ER-MLP
    citation:
        author: Dong
        year: 2014
        link: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45634.pdf
    """

    func = pkf.ermlp_interaction

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int | None = None,
    ):
        """Initialize the interaction module.

        :param embedding_dim:
            The embedding vector dimension for entities and relations.
        :param hidden_dim:
            The hidden dimension of the MLP. Defaults to `embedding_dim`.
        """
        super().__init__()
        # normalize hidden_dim
        hidden_dim = hidden_dim or embedding_dim
        self.hidden = nn.Linear(in_features=3 * embedding_dim, out_features=hidden_dim, bias=True)
        self.activation = nn.ReLU()
        self.hidden_to_score = nn.Linear(in_features=hidden_dim, out_features=1, bias=True)

    # docstr-coverage: inherited
    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(
            hidden=self.hidden,
            activation=self.activation,
            final=self.hidden_to_score,
        )

    # docstr-coverage: inherited
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


@parse_docdata
class ERMLPEInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    r"""A stateful module for the ER-MLP (E) interaction function.

    This interaction uses a neural network-based approach similar to ER-MLP and with slight modifications.
    In ER-MLP, the interaction is:

    .. math::

        f(h, r, t) = \textbf{w}^{T} g(\textbf{W} [\textbf{h}; \textbf{r}; \textbf{t}])

    whereas in ER-MLP (E) the interaction is:

    .. math::

        f(h, r, t) = \textbf{t}^{T} f(\textbf{W} (g(\textbf{W} [\textbf{h}; \textbf{r}]))

    including dropouts and batch-norms between each two hidden layers. Thus, the ConvE interaction can be seen as a
    special case of ERMLP (E).

    .. seealso:: :func:`pykeen.nn.functional.ermlpe_interaction`

    ---
    name: ER-MLP (E)
    citation:
        author: Sharifzadeh
        year: 2019
        link: https://github.com/pykeen/pykeen
        github: pykeen/pykeen
    """

    func = pkf.ermlpe_interaction

    def __init__(
        self,
        embedding_dim: int = 256,
        input_dropout: float = 0.2,
        hidden_dim: int | None = None,
        hidden_dropout: float | None = None,
    ):
        """
        Initialize the interaction module.

        :param embedding_dim:
            the embedding dimension of entities and relations
        :param hidden_dim:
            the hidden dimension of the MLP. Defaults to `embedding_dim`.
        :param input_dropout:
            the dropout applied *before* the first layer
        :param hidden_dropout:
            the dropout applied *after* the first layer
        """
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim
        hidden_dropout = input_dropout if hidden_dropout is None else hidden_dropout
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

    # docstr-coverage: inherited
    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(mlp=self.mlp)


@parse_docdata
class TransRInteraction(
    NormBasedInteraction[
        torch.FloatTensor,
        tuple[torch.FloatTensor, torch.FloatTensor],
        torch.FloatTensor,
    ],
):
    """A stateful module for the TransR interaction function.

    .. seealso:: :func:`pykeen.nn.functional.transr_interaction`

    ---
    citation:
        author: Lin
        year: 2015
        link: http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/
    """

    relation_shape = ("e", "de")
    func = pkf.transr_interaction

    def __init__(self, p: int, power_norm: bool = True):
        """
        Initialize the interaction module.

        :param p:
            the $p$ value of the norm to use, cf. :meth:`NormBasedInteraction.__init__`
        :param power_norm:
            whether to use the $p$th power of the p-norm, cf. :meth:`NormBasedInteraction.__init__`.
        """
        super().__init__(p=p, power_norm=power_norm)

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(h=h, r=r[0], t=t, m_r=r[1])


@parse_docdata
class RotatEInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    r"""The RotatE interaction function proposed by [sun2019]_.

    RotatE operates on complex-valued entity and relation representations, i.e.,
    $\textbf{e}_i, \textbf{r}_i \in \mathbb{C}^d$.

    .. note::
        this method generally expects all tensors to be of complex datatype, i.e., `torch.is_complex(x)` to evaluate to
        `True`. However, for backwards compatibility and convenience in use, you can also pass real tensors whose shape
        is compliant with :func:`torch.view_as_complex`, cf. :func:`pykeen.utils.ensure_complex`.

    ---
    citation:
        arxiv: 1902.10197
        author: Sun
        github: DeepGraphLearning/KnowledgeGraphEmbedding
        link: https://arxiv.org/abs/1902.10197
        year: 2019
    """

    # TODO: update docstring

    is_complex: ClassVar[bool] = True

    # TODO: give this a better name?
    @staticmethod
    def func(h: FloatTensor, r: FloatTensor, t: FloatTensor) -> FloatTensor:
        """Evaluate the interaction function.

        .. note::
            this method expects all tensors to be of complex datatype, i.e., `torch.is_complex(x)` to evaluate to
            `True`.

        :param h: shape: (`*batch_dims`, dim)
            The head representations.
        :param r: shape: (`*batch_dims`, dim)
            The relation representations.
        :param t: shape: (`*batch_dims`, dim)
            The tail representations.

        :return: shape: batch_dims
            The scores.
        """
        if estimate_cost_of_sequence(h.shape, r.shape) < estimate_cost_of_sequence(r.shape, t.shape):
            # r expresses a rotation in complex plane.
            # rotate head by relation (=Hadamard product in complex space)
            h = h * r
        else:
            # rotate tail by inverse of relation
            # The inverse rotation is expressed by the complex conjugate of r.
            # The score is computed as the distance of the relation-rotated head to the tail.
            # Equivalently, we can rotate the tail by the inverse relation, and measure the distance to the head, i.e.
            # |h * r - t| = |h - conj(r) * t|
            t = t * torch.conj(r)

        return negative_norm(h - t, p=2, power_norm=False)


@parse_docdata
class HolEInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A module wrapper for the stateless HolE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.hole_interaction`

    ---
    citation:
        author: Nickel
        year: 2016
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828
        github: mnick/holographic-embeddings
    """

    func = pkf.hole_interaction


@parse_docdata
class ProjEInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A stateful module for the ProjE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.proje_interaction`

    ---
    citation:
        author: Shi
        year: 2017
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14279
        github: nddsg/ProjE
    """

    func = pkf.proje_interaction

    def __init__(
        self,
        embedding_dim: int = 50,
        inner_non_linearity: HintOrType[nn.Module] = None,
    ):
        """
        Initialize the interaction module.

        :param embedding_dim:
            the embedding dimension of entities and relations
        :param inner_non_linearity:
            the inner non-linearity, or a hint thereof. Defaults to :class:`nn.Tanh`.
            Disable by passing :class:`nn.Idenity`
        """
        super().__init__()

        # Global entity projection
        self.d_e = nn.Parameter(torch.empty(embedding_dim), requires_grad=True)

        # Global relation projection
        self.d_r = nn.Parameter(torch.empty(embedding_dim), requires_grad=True)

        # Global combination bias
        self.b_c = nn.Parameter(torch.empty(embedding_dim), requires_grad=True)

        # Global combination bias
        self.b_p = nn.Parameter(torch.empty(tuple()), requires_grad=True)

        if inner_non_linearity is None:
            inner_non_linearity = nn.Tanh
        self.inner_non_linearity = activation_resolver.make(inner_non_linearity)

    # docstr-coverage: inherited
    def reset_parameters(self):  # noqa: D102
        embedding_dim = self.d_e.shape[0]
        bound = math.sqrt(6) / embedding_dim
        for p in self.parameters():
            nn.init.uniform_(p, a=-bound, b=bound)

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:
        return dict(d_e=self.d_e, d_r=self.d_r, b_c=self.b_c, b_p=self.b_p, activation=self.inner_non_linearity)


@parse_docdata
class RESCALInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A module wrapper for the stateless RESCAL interaction function.

    .. seealso:: :func:`pykeen.nn.functional.rescal_interaction`

    ---
    citation:
        author: Nickel
        year: 2011
        link: https://icml.cc/2011/papers/438_icmlpaper.pdf
    """

    relation_shape = ("dd",)
    func = pkf.rescal_interaction


@parse_docdata
class SEInteraction(
    NormBasedInteraction[
        torch.FloatTensor,
        tuple[torch.FloatTensor, torch.FloatTensor],
        torch.FloatTensor,
    ],
):
    """A stateful module for the Structured Embedding (SE) interaction function.

    .. seealso:: :func:`pykeen.nn.functional.structured_embedding_interaction`

    ---
    name: Structured Embedding
    citation:
        author: Bordes
        year: 2011
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/download/3659/3898
    """

    relation_shape = ("dd", "dd")
    func = pkf.se_interaction

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(h=h, t=t, r_h=r[0], r_t=r[1])


@parse_docdata
class TuckerInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A stateful module for the stateless Tucker interaction function.

    .. seealso:: :func:`pykeen.nn.functional.tucker_interaction`

    ---
    citation:
        author: Balažević
        year: 2019
        arxiv: 1901.09590
        link: https://arxiv.org/abs/1901.09590
        github: ibalazevic/TuckER
    """

    func = pkf.tucker_interaction

    # default core tensor initialization
    # cf. https://github.com/ibalazevic/TuckER/blob/master/model.py#L12
    default_core_initializer: ClassVar[Initializer] = staticmethod(nn.init.uniform_)  # type: ignore
    default_core_initializer_kwargs: Mapping[str, Any] = {"a": -1.0, "b": 1.0}

    def __init__(
        self,
        embedding_dim: int = 200,
        relation_dim: int | None = None,
        head_dropout: float = 0.3,
        relation_dropout: float = 0.4,
        head_relation_dropout: float = 0.5,
        apply_batch_normalization: bool = True,
        core_initializer: Hint[Initializer] = None,
        core_initializer_kwargs: OptionalKwargs = None,
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
        :param core_initializer:
            the core tensor's initializer, or a hint thereof
        :param core_initializer_kwargs:
            additional keyword-based parameters for the initializer
        """
        super().__init__()

        # normalize initializer
        if core_initializer is None:
            core_initializer = self.default_core_initializer
        self.core_initializer = core_initializer
        if core_initializer is self.default_core_initializer and core_initializer_kwargs is None:
            core_initializer_kwargs = self.default_core_initializer_kwargs
        self.core_initializer_kwargs = core_initializer_kwargs

        # normalize relation dimension
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

        self.reset_parameters()

    # docstr-coverage: inherited
    def reset_parameters(self):  # noqa: D102
        # instantiate here to make module easily serializable
        core_initializer = initializer_resolver.make(self.core_initializer, pos_kwargs=self.core_initializer_kwargs)
        core_initializer(self.core_tensor)
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


@parse_docdata
class UMInteraction(
    NormBasedInteraction[torch.FloatTensor, None, torch.FloatTensor],
):
    """A stateful module for the UnstructuredModel interaction function.

    .. seealso:: :func:`pykeen.nn.functional.unstructured_model_interaction`

    ---
    name: Unstructured Model
    citation:
        author: Bordes
        year: 2014
        link: https://link.springer.com/content/pdf/10.1007%2Fs10994-013-5363-6.pdf
    """

    # shapes
    relation_shape: Sequence[str] = tuple()

    func = pkf.um_interaction

    def __init__(self, p: int, power_norm: bool = True):
        """
        Initialize the interaction module.

        :param p:
            the $p$ value of the norm to use, cf. :meth:`NormBasedInteraction.__init__`
        :param power_norm:
            whether to use the $p$th power of the p-norm, cf. :meth:`NormBasedInteraction.__init__`.
        """
        super().__init__(p=p, power_norm=power_norm)

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(h=h, t=t)


@parse_docdata
class TorusEInteraction(NormBasedInteraction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
    """A stateful module for the TorusE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.toruse_interaction`

    ---
    citation:
        author: Ebisu
        year: 2018
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16227
        arxiv: 1711.05435
        github: TakumaE/TorusE
    """

    func = pkf.toruse_interaction

    def __init__(self, p: int = 2, power_norm: bool = False):
        """
        Initialize the interaction module.

        :param p:
            the $p$ value of the norm to use, cf. :meth:`NormBasedInteraction.__init__`
        :param power_norm:
            whether to use the $p$th power of the p-norm, cf. :meth:`NormBasedInteraction.__init__`.
        """
        super().__init__(p=p, power_norm=power_norm)


@parse_docdata
class TransDInteraction(
    NormBasedInteraction[
        tuple[torch.FloatTensor, torch.FloatTensor],
        tuple[torch.FloatTensor, torch.FloatTensor],
        tuple[torch.FloatTensor, torch.FloatTensor],
    ],
):
    """A stateful module for the TransD interaction function.

    .. seealso:: :func:`pykeen.nn.functional.transd_interaction`

    ---
    citation:
        author: Ji
        year: 2015
        link: http://www.aclweb.org/anthology/P15-1067
    """

    entity_shape = ("d", "d")
    relation_shape = ("e", "e")
    func = pkf.transd_interaction

    def __init__(self, p: int = 2, power_norm: bool = True):
        """
        Initialize the interaction module.

        :param p:
            the $p$ value of the norm to use, cf. :meth:`NormBasedInteraction.__init__`
        :param power_norm:
            whether to use the $p$th power of the p-norm, cf. :meth:`NormBasedInteraction.__init__`.
        """
        super().__init__(p=p, power_norm=power_norm)

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: tuple[torch.FloatTensor, torch.FloatTensor],
        r: tuple[torch.FloatTensor, torch.FloatTensor],
        t: tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        h, h_p = h
        r, r_p = r
        t, t_p = t
        return dict(h=h, r=r, t=t, h_p=h_p, r_p=r_p, t_p=t_p)


@parse_docdata
class NTNInteraction(
    FunctionalInteraction[
        torch.FloatTensor,
        tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
        torch.FloatTensor,
    ],
):
    """A stateful module for the NTN interaction function.

    .. seealso:: :func:`pykeen.nn.functional.ntn_interaction`

    ---
    citation:
        author: Socher
        year: 2013
        link: https://proceedings.neurips.cc/paper/2013/file/b337e84de8752b27eda3a12363109e80-Paper.pdf
        github: khurram18/NeuralTensorNetworks
    """

    relation_shape = ("kdd", "kd", "kd", "k", "k")
    func = pkf.ntn_interaction

    def __init__(
        self,
        activation: HintOrType[nn.Module] = None,
        activation_kwargs: Mapping[str, Any] | None = None,
    ):
        """Initialize NTN with the given non-linear activation function.

        :param activation: A non-linear activation function. Defaults to the hyperbolic
            tangent :class:`torch.nn.Tanh` if None, otherwise uses the :data:`pykeen.utils.activation_resolver`
            for lookup.
        :param activation_kwargs: If the ``activation`` is passed as a class, these keyword arguments
            are used during its instantiation.
        """
        super().__init__()
        if activation is None:
            activation = nn.Tanh()
        self.non_linearity = activation_resolver.make(activation, activation_kwargs)

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: torch.FloatTensor,
        r: tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
        t: torch.FloatTensor,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        w, vh, vt, b, u = r
        return dict(h=h, t=t, w=w, b=b, u=u, vh=vh, vt=vt)

    # docstr-coverage: inherited
    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(activation=self.non_linearity)


@parse_docdata
class KG2EInteraction(
    FunctionalInteraction[
        tuple[torch.FloatTensor, torch.FloatTensor],
        tuple[torch.FloatTensor, torch.FloatTensor],
        tuple[torch.FloatTensor, torch.FloatTensor],
    ],
):
    """A stateful module for the KG2E interaction function.

    .. seealso:: :func:`pykeen.nn.functional.kg2e_interaction`

    ---
    citation:
        author: He
        year: 2015
        link: https://dl.acm.org/doi/10.1145/2806416.2806502
    """

    entity_shape = ("d", "d")
    relation_shape = ("d", "d")
    similarity: str
    exact: bool
    func = pkf.kg2e_interaction

    def __init__(self, similarity: str | None = None, exact: bool = True):
        """
        Initialize the interaction module.

        :param similarity:
            the distribution similarity to use. Defaults to KL divergence.
        :param exact:
            whether to compute the exact similarity, or leave out constant terms
        """
        super().__init__()
        if similarity is None:
            similarity = "KL"
        self.similarity = similarity
        self.exact = exact

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: tuple[torch.FloatTensor, torch.FloatTensor],
        r: tuple[torch.FloatTensor, torch.FloatTensor],
        t: tuple[torch.FloatTensor, torch.FloatTensor],
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


@parse_docdata
class TransHInteraction(NormBasedInteraction[FloatTensor, tuple[FloatTensor, FloatTensor], FloatTensor]):
    """A stateful module for the TransH interaction function.

    .. seealso:: :func:`pykeen.nn.functional.transh_interaction`

    ---
    citation:
        author: Wang
        year: 2014
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546
    """

    relation_shape = ("d", "d")
    func = pkf.transh_interaction

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(h=h, w_r=r[1], d_r=r[0], t=t)


@parse_docdata
class MuREInteraction(
    NormBasedInteraction[
        tuple[FloatTensor, FloatTensor, FloatTensor],
        tuple[FloatTensor, FloatTensor],
        tuple[FloatTensor, FloatTensor, FloatTensor],
    ],
):
    """A stateful module for the MuRE interaction function from [balazevic2019b]_.

    .. seealso:: :func:`pykeen.nn.functional.mure_interaction`

    ---
    citation:
        author: Balažević
        year: 2019
        link: https://arxiv.org/abs/1905.09791
        arxiv: 1905.09791
    """

    # there are separate biases for entities in head and tail position
    entity_shape = ("d", "", "")
    relation_shape = ("d", "d")
    func = pkf.mure_interaction

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: tuple[FloatTensor, FloatTensor, FloatTensor],
        r: tuple[FloatTensor, FloatTensor],
        t: tuple[FloatTensor, FloatTensor, FloatTensor],
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        h, b_h, _ = h
        t, _, b_t = t
        r_vec, r_mat = r
        return dict(h=h, b_h=b_h, r_vec=r_vec, r_mat=r_mat, t=t, b_t=b_t)


@parse_docdata
class SimplEInteraction(
    FunctionalInteraction[
        tuple[torch.FloatTensor, torch.FloatTensor],
        tuple[torch.FloatTensor, torch.FloatTensor],
        tuple[torch.FloatTensor, torch.FloatTensor],
    ],
):
    """A module wrapper for the SimplE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.simple_interaction`

    ---
    citation:
        author: Kazemi
        year: 2018
        link: https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs
        github: Mehran-k/SimplE
    """

    func = pkf.simple_interaction
    entity_shape = ("d", "d")
    relation_shape = ("d", "d")

    def __init__(self, clamp_score: None | float | tuple[float, float] = None):
        """
        Initialize the interaction module.

        :param clamp_score:
            whether to clamp scores into a fixed interval
        """
        super().__init__()
        if isinstance(clamp_score, float):
            clamp_score = (-clamp_score, clamp_score)
        self.clamp_score = clamp_score

    # docstr-coverage: inherited
    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(clamp=self.clamp_score)

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(h=h[0], h_inv=h[1], r=r[0], r_inv=r[1], t=t[0], t_inv=t[1])


@parse_docdata
class PairREInteraction(NormBasedInteraction[FloatTensor, tuple[FloatTensor, FloatTensor], FloatTensor]):
    """A stateful module for the PairRE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.pair_re_interaction`

    ---
    citation:
        author: Chao
        year: 2020
        link: http://arxiv.org/abs/2011.03798
        arxiv: 2011.03798
        github: alipay/KnowledgeGraphEmbeddingsViaPairedRelationVectors_PairRE
    """

    relation_shape = ("d", "d")
    func = pkf.pair_re_interaction

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(h=h, r_h=r[0], r_t=r[1], t=t)


@parse_docdata
class QuatEInteraction(
    FunctionalInteraction[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ],
):
    """A module wrapper for the QuatE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.quat_e_interaction`

    ---
    citation:
        author: Zhang
        year: 2019
        arxiv: 1904.10281
        link: https://arxiv.org/abs/1904.10281
        github: cheungdaven/quate
    """

    # with k=4
    entity_shape: Sequence[str] = ("dk",)
    relation_shape: Sequence[str] = ("dk",)
    func = pkf.quat_e_interaction

    def __init__(self) -> None:
        """Initialize the interaction module."""
        super().__init__()
        self.register_buffer(name="table", tensor=quaterion_multiplication_table())

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:
        return dict(table=self.table)


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
        self._tail_entity_shape = base._tail_entity_shape

        # The parameters of the affine transformation: bias
        self.bias = nn.Parameter(torch.empty(size=tuple()), requires_grad=trainable_bias)
        self.initial_bias = torch.as_tensor(data=[initial_bias], dtype=torch.get_default_dtype()).squeeze()

        # scale. We model this as log(scale) to ensure scale > 0, and thus monotonicity
        self.log_scale = nn.Parameter(torch.empty(size=tuple()), requires_grad=trainable_scale)
        self.initial_log_scale = torch.as_tensor(
            data=[math.log(initial_scale)],
            dtype=torch.get_default_dtype(),
        ).squeeze()

    # docstr-coverage: inherited
    def reset_parameters(self):  # noqa: D102
        self.bias.data = self.initial_bias.to(device=self.bias.device)
        self.log_scale.data = self.initial_log_scale.to(device=self.bias.device)

    # docstr-coverage: inherited
    def forward(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> torch.FloatTensor:  # noqa: D102
        return self.log_scale.exp() * self.base(h=h, r=r, t=t) + self.bias


@parse_docdata
class CrossEInteraction(FunctionalInteraction[FloatTensor, tuple[FloatTensor, FloatTensor], FloatTensor]):
    """A module wrapper for the CrossE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.cross_e_interaction`

    ---
    citation:
        author: Zhang
        year: 2019
        link: https://arxiv.org/abs/1903.04750
        arxiv: 1903.04750
    """

    func = pkf.cross_e_interaction
    relation_shape = ("d", "d")

    def __init__(
        self,
        embedding_dim: int = 50,
        combination_activation: HintOrType[nn.Module] = nn.Tanh,
        combination_activation_kwargs: Mapping[str, Any] | None = None,
        combination_dropout: float | None = 0.5,
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
        self.combination_bias = nn.Parameter(data=torch.zeros(embedding_dim))
        self.combination_dropout = nn.Dropout(combination_dropout) if combination_dropout else None

    # docstr-coverage: inherited
    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(
            bias=self.combination_bias,
            activation=self.combination_activation,
            dropout=self.combination_dropout,
        )

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: FloatTensor,
        r: tuple[FloatTensor, FloatTensor],
        t: FloatTensor,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        r, c_r = r
        return dict(h=h, r=r, c_r=c_r, t=t)


@parse_docdata
class BoxEInteraction(
    NormBasedInteraction[
        tuple[FloatTensor, FloatTensor],
        tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor],
        tuple[FloatTensor, FloatTensor],
    ]
):
    """
    The BoxE interaction from [abboud2020]_.

    Entities are represented by two $d$-dimensional vectors describing the *base position* as well
    as the translational bump, which translates all the entities co-occuring in a fact with this entity
    from their base positions to their final embeddings, called "bumping".

    Relations are represented as a fixed number of hyper-rectangles corresponding to the relation's arity.
    Since we are only considering single-hop link predition here, the arity is always two, i.e., one box
    for the head position and another one for the tail position. There are different possibilities to
    parametrize a hyper-rectangle, where the most common may be its description as the coordinate of to
    opposing vertices. BoxE suggests a different parametrization:

    - each box has a base position given by its center
    - each box has an extent in each dimension. This size is further factored in

      - a scalar global scaling factor
      - a normalized extent in each dimension, i.e., the extents sum to one

    ---
    citation:
        author: Abboud
        year: 2020
        link: https://arxiv.org/abs/2007.06267
        github: ralphabb/BoxE
    """

    relation_shape = ("d", "d", "s", "d", "d", "s")  # Boxes are 2xd (size) each, x 2 sets of boxes: head and tail
    entity_shape = ("d", "d")  # Base position and bump

    def __init__(self, tanh_map: bool = True, p: int = 2, power_norm: bool = False):
        r"""
        Instantiate the interaction module.

        :param tanh_map:
            Should the hyperbolic tangent be applied to all representations prior to model scoring?
        :param p:
            the order of the norm
        :param power_norm:
            whether to use the p-th power of the norm instead
        """
        super().__init__(p=p, power_norm=power_norm)
        self.tanh_map = tanh_map

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: tuple[FloatTensor, FloatTensor],
        r: tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor],
        t: tuple[FloatTensor, FloatTensor],
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        rh_base, rh_delta, rh_size, rt_base, rt_delta, rt_size = r
        h_pos, h_bump = h
        t_pos, t_bump = t
        return dict(
            # head position and bump
            h_pos=h_pos,
            h_bump=h_bump,
            # relation box: head
            rh_base=rh_base,
            rh_delta=rh_delta,
            rh_size=rh_size,
            # relation box: tail
            rt_base=rt_base,
            rt_delta=rt_delta,
            rt_size=rt_size,
            # tail position and bump
            t_pos=t_pos,
            t_bump=t_bump,
        )

    # docstr-coverage: inherited
    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        state = super()._prepare_state_for_functional()
        state["tanh_map"] = self.tanh_map
        return state

    @staticmethod
    def product_normalize(x: torch.FloatTensor, dim: int = -1) -> torch.FloatTensor:
        r"""Normalize a tensor along a given dimension so that the geometric mean is 1.0.

        :param x: shape: s
            An input tensor
        :param dim:
            the dimension along which to normalize the tensor

        :return: shape: s
            An output tensor where the given dimension is normalized to have a geometric mean of 1.0.
        """
        return x / at_least_eps(at_least_eps(x.abs()).log().mean(dim=dim, keepdim=True).exp())

    @staticmethod
    def point_to_box_distance(
        points: torch.FloatTensor,
        box_lows: torch.FloatTensor,
        box_highs: torch.FloatTensor,
    ) -> torch.FloatTensor:
        r"""Compute the point to box distance function proposed by [abboud2020]_ in an element-wise fashion.

        :param points: shape: ``(*, d)``
            the positions of the points being scored against boxes
        :param box_lows: shape: ``(*, d)``
            the lower corners of the boxes
        :param box_highs: shape: ``(*, d)``
            the upper corners of the boxes

        :returns:
            Element-wise distance function scores as per the definition above

            Given points $p$, box_lows $l$, and box_highs $h$, the following quantities are
            defined:

            - Width $w$ is the difference between the upper and lower box bound: $w = h - l$
            - Box centers $c$ are the mean of the box bounds: $c = (h + l) / 2$

            Finally, the point to box distance $dist(p,l,h)$ is defined as
            the following piecewise function:

            .. math::

                dist(p,l,h) = \begin{cases}
                    |p-c|/(w+1) & l <= p <+ h \\
                    |p-c|*(w+1) - 0.5*w*((w+1)-1/(w+1)) & otherwise \\
                \end{cases}
        """
        widths = box_highs - box_lows

        # compute width plus 1
        widths_p1 = widths + 1

        # compute box midpoints
        # TODO: we already had this before, as `base`
        centres = 0.5 * (box_lows + box_highs)

        return torch.where(
            # inside box?
            torch.logical_and(points >= box_lows, points <= box_highs),
            # yes: |p - c| / (w + 1)
            torch.abs(points - centres) / widths_p1,
            # no: (w + 1) * |p - c| - 0.5 * w * (w - 1/(w + 1))
            widths_p1 * torch.abs(points - centres) - (0.5 * widths) * (widths_p1 - 1 / widths_p1),
        )

    @classmethod
    def boxe_kg_arity_position_score(
        cls,
        entity_pos: torch.FloatTensor,
        other_entity_bump: torch.FloatTensor,
        relation_box: tuple[torch.FloatTensor, torch.FloatTensor],
        tanh_map: bool,
        p: int,
        power_norm: bool,
    ) -> torch.FloatTensor:
        r"""Perform the BoxE computation at a single arity position.

        .. note::
            this computation is parallelizable across all positions

        .. note ::
            `entity_pos`, `other_entity_bump`, `relation_box_low` and `relation_box_high` have to be in broadcastable
            shape.

        :param entity_pos: shape: ``(*s_p, d)``
            This is the base entity position of the entity appearing in the target position. For example,
            for a fact $r(h, t)$ and the head arity position, `entity_pos` is the base position of $h$.
        :param other_entity_bump: shape: ``(*s_b, d)``
            This is the bump of the entity at the other position in the fact. For example, given a
            fact $r(h, t)$ and the head arity position, `other_entity_bump` is the bump of $t$.
        :param relation_box: shape: ``(*s_r, d)``
            The lower/upper corner of the relation box at the target arity position.
        :param tanh_map:
            whether to apply the tanh map regularizer
        :param p:
            The norm order to apply across dimensions to compute overall position score.
        :param power_norm:
            whether to use the powered norm instead

        :return: shape: ``*s``
            Arity-position score for the entity relative to the target relation box. Larger is better. The shape is the
            broadcasted shape from position, bump and box, where the last dimension has been removed.
        """
        # Step 1: Apply the other entity bump
        bumped_representation = entity_pos + other_entity_bump

        relation_box_low, relation_box_high = relation_box

        # Step 2: Apply tanh if tanh_map is set to True.
        if tanh_map:
            relation_box_low = torch.tanh(relation_box_low)
            relation_box_high = torch.tanh(relation_box_high)
            bumped_representation = torch.tanh(bumped_representation)

        # Compute the distance function output element-wise
        element_wise_distance = cls.point_to_box_distance(
            points=bumped_representation,
            box_lows=relation_box_low,
            box_highs=relation_box_high,
        )

        # Finally, compute the norm
        return negative_norm(element_wise_distance, p=p, power_norm=power_norm)

    @classmethod
    def compute_box(
        cls,
        base: torch.FloatTensor,
        delta: torch.FloatTensor,
        size: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        r"""Compute the lower and upper corners of a resulting box.

        :param base: shape: ``(*, d)``
            the base position (box center) of the input relation embeddings
        :param delta:  shape: ``(*, d)``
            the base shape of the input relation embeddings
        :param size: shape: ``(*, d)``
            the size scalar vectors of the input relation embeddings

        :return: shape: ``(*, d)`` each
            lower and upper bounds of the box whose embeddings are provided as input.
        """
        # Enforce that sizes are strictly positive by passing through ELU
        size_pos = torch.nn.functional.elu(size) + 1

        # Shape vector is normalized using the above helper function
        delta_norm = cls.product_normalize(delta)

        # Size is learned separately and applied to normalized shape
        delta_final = size_pos * delta_norm

        # Compute potential boundaries by applying the shape in substraction
        first_bound = base - 0.5 * delta_final

        # and in addition
        second_bound = base + 0.5 * delta_final

        # Compute box upper bounds using min and max respectively
        box_low = torch.minimum(first_bound, second_bound)
        box_high = torch.maximum(first_bound, second_bound)

        return box_low, box_high

    @staticmethod
    def func(
        # head
        h_pos: torch.FloatTensor,
        h_bump: torch.FloatTensor,
        # relation box: head
        rh_base: torch.FloatTensor,
        rh_delta: torch.FloatTensor,
        rh_size: torch.FloatTensor,
        # relation box: tail
        rt_base: torch.FloatTensor,
        rt_delta: torch.FloatTensor,
        rt_size: torch.FloatTensor,
        # tail
        t_pos: torch.FloatTensor,
        t_bump: torch.FloatTensor,
        # power norm
        tanh_map: bool = True,
        p: int = 2,
        power_norm: bool = False,
    ) -> FloatTensor:
        """
        Evaluate the BoxE interaction function from [abboud2020]_.

        :param h_pos: shape: (`*batch_dims`, d)
            the head entity position
        :param h_bump: shape: (`*batch_dims`, d)
            the head entity bump

        :param rh_base: shape: (`*batch_dims`, d)
            the relation-specific head box base position
        :param rh_delta: shape: (`*batch_dims`, d)
            # the relation-specific head box base shape (normalized to have a volume of 1):
        :param rh_size: shape: (`*batch_dims`, 1)
            the relation-specific head box size (a scalar)

        :param rt_base: shape: (`*batch_dims`, d)
            the relation-specific tail box base position
        :param rt_delta: shape: (`*batch_dims`, d)
            # the relation-specific tail box base shape (normalized to have a volume of 1):
        :param rt_size: shape: (`*batch_dims`, d)
            the relation-specific tail box size

        :param t_pos: shape: (`*batch_dims`, d)
            the tail entity position
        :param t_bump: shape: (`*batch_dims`, d)
            the tail entity bump

        :param tanh_map:
            whether to apply the tanh mapping
        :param p:
            the order of the norm to apply
        :param power_norm:
            whether to use the p-th power of the p-norm instead

        :return: shape: batch_dims
            The scores.
        """
        return sum(
            BoxEInteraction.boxe_kg_arity_position_score(
                entity_pos=entity_pos,
                other_entity_bump=other_entity_pos,
                relation_box=BoxEInteraction.compute_box(base=base, delta=delta, size=size),
                tanh_map=tanh_map,
                p=p,
                power_norm=power_norm,
            )
            for entity_pos, other_entity_pos, base, delta, size in (
                (h_pos, t_bump, rh_base, rh_delta, rh_size),
                (t_pos, h_bump, rt_base, rt_delta, rt_size),
            )
        )


@parse_docdata
class CPInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """
    An implementation of the CP interaction as described [lacroix2018]_ (originally from [hitchcock1927]_).

    .. note ::
        For $k=1$, this interaction is the same as DistMult (but consider the note below).

    .. note ::
        For equivalence to CP, entities should have different representations for head & tail role. This is different
        to DistMult.

    ---
    name: Canonical Tensor Decomposition
    citation:
        author: Lacroix
        year: 2018
        arxiv: 1806.07297
        link: https://arxiv.org/abs/1806.07297
        github: facebookresearch/kbc
    """

    func = pkf.cp_interaction
    entity_shape = ("kd",)
    relation_shape = ("kd",)
    _head_indices = (0,)
    _tail_indices = (1,)


@parse_docdata
class MultiLinearTuckerInteraction(
    FunctionalInteraction[tuple[FloatTensor, FloatTensor], FloatTensor, tuple[FloatTensor, FloatTensor]]
):
    """
    An implementation of the original (multi-linear) TuckER interaction as described [tucker1966]_.

    .. note ::
        For small tensors, there are more efficient algorithms to compute the decomposition, e.g.,
        http://tensorly.org/stable/modules/generated/tensorly.decomposition.Tucker.html

    ---
    name: MultiLinearTucker
    citation:
        author: Tucker
        year: 1966
        link: https://dx.doi.org/10.1007/BF02289464
    """

    func = pkf.multilinear_tucker_interaction
    entity_shape = ("d", "f")
    relation_shape = ("e",)

    def __init__(
        self,
        head_dim: int = 64,
        relation_dim: int | None = None,
        tail_dim: int | None = None,
    ):
        """
        Initialize the Tucker interaction function.

        :param head_dim:
            The head entity embedding dimension.
        :param relation_dim:
            The relation embedding dimension. Defaults to `head_dim`.
        :param tail_dim:
            The tail entity embedding dimension. Defaults to `head_dim`.
        """
        super().__init__()

        # input normalization
        relation_dim = relation_dim or head_dim
        tail_dim = tail_dim or head_dim

        # Core tensor
        self.core_tensor = nn.Parameter(
            torch.empty(head_dim, relation_dim, tail_dim),
            requires_grad=True,
        )

    # docstr-coverage: inherited
    def reset_parameters(self):  # noqa: D102
        # initialize core tensor
        nn.init.normal_(
            self.core_tensor,
            mean=0,
            std=numpy.sqrt(numpy.prod(numpy.reciprocal(numpy.asarray(self.core_tensor.shape)))),
        )

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: tuple[FloatTensor, FloatTensor],
        r: FloatTensor,
        t: tuple[FloatTensor, FloatTensor],
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(h=h[0], r=r, t=t[1])

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:
        return dict(core_tensor=self.core_tensor)


@parse_docdata
class TransformerInteraction(FunctionalInteraction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
    """Transformer-based interaction, as described in [galkin2020]_.

    ---
    name: Transformer
    citation:
        author: Galkin
        year: 2020
        link: https://doi.org/10.18653/v1/2020.emnlp-main.596
    """

    func = pkf.transformer_interaction

    def __init__(
        self,
        input_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        position_initializer: HintOrType[Initializer] = xavier_normal_,
    ):
        """
        Initialize the module.

        :param input_dim: >0
            the input dimension
        :param num_layers: >0
            the number of Transformer layers, cf. :class:`nn.TransformerEncoder`.
        :param num_heads: >0
            the number of self-attention heads inside each transformer encoder layer,
            cf. :class:`nn.TransformerEncoderLayer`
        :param dropout:
            the dropout rate on each transformer encoder layer, cf. :class:`nn.TransformerEncoderLayer`
        :param dim_feedforward:
            the hidden dimension of the feed-forward layers of the transformer encoder layer,
            cf. :class:`nn.TransformerEncoderLayer`
        :param position_initializer:
            the initializer to use for positional embeddings
        """
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            ),
            num_layers=num_layers,
        )
        self.position_embeddings = nn.Parameter(position_initializer(torch.empty(2, input_dim)))
        self.final = nn.Linear(input_dim, input_dim, bias=True)

    # docstr-coverage: inherited
    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(
            transformer=self.transformer,
            position_embeddings=self.position_embeddings,
            final=self.final,
        )


@parse_docdata
class TripleREInteraction(
    NormBasedInteraction[
        FloatTensor,
        tuple[FloatTensor, FloatTensor, FloatTensor],
        FloatTensor,
    ]
):
    """A stateful module for the TripleRE interaction function from [yu2021]_.

    .. math ::
        score(h, (r_h, r, r_t), t) = h * (r_h + u) - t * (r_t + u) + r

    .. note ::

        For equivalence to the paper version, `h` and `t` should be normalized to unit
        Euclidean length, and `p` and `power_norm` be kept at their default values.

    .. seealso:: :func:`pykeen.nn.functional.triple_re_interaction`

    .. seealso:: https://github.com/LongYu-360/TripleRE-Add-NodePiece

    .. note ::
        this interaction is equivalent to :class:`LineaREInteraction` except the `u` term
    ---
    name: TripleRE
    citation:
        author: Yu
        year: 2021
        link: https://vixra.org/abs/2112.0095
    """

    # r_head, r_mid, r_tail
    relation_shape = ("d", "d", "d")

    func = pkf.triple_re_interaction

    def __init__(self, u: float | None = 1.0, p: int = 1, power_norm: bool = False):
        """
        Initialize the module.

        :param u:
            the relation factor offset. can be set to None to disable it.
        :param p:
            The norm used with :func:`torch.linalg.vector_norm`. Defaults to 1 for TripleRE.
        :param power_norm:
            Whether to use the p-th power of the $L_p$ norm. It has the advantage of being differentiable around 0,
            and numerically more stable. Defaults to False for TripleRE.
        """
        super().__init__(p=p, power_norm=power_norm)
        self.u = u

    # docstr-coverage: inherited
    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._prepare_state_for_functional()
        kwargs["u"] = self.u
        return kwargs

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: FloatTensor,
        r: tuple[FloatTensor, FloatTensor, FloatTensor],
        t: FloatTensor,
    ) -> MutableMapping[str, FloatTensor]:  # noqa: D102
        r_head, r_mid, r_tail = r
        return dict(
            h=h,
            r_head=r_head,
            r_mid=r_mid,
            r_tail=r_tail,
            t=t,
        )


# type alias for AutoSF block description
# head_index, relation_index, tail_index, sign
AutoSFBlock = tuple[int, int, int, Sign]


@parse_docdata
class AutoSFInteraction(FunctionalInteraction[HeadRepresentation, RelationRepresentation, TailRepresentation]):
    r"""
    The AutoSF interaction as described by [zhang2020]_.

    This interaction function is a parametrized way to express bi-linear models
    with block structure. It divides the entity and relation representations into blocks,
    and expresses the interaction as a sequence of 4-tuples $(i_h, i_r, i_t, s)$,
    where $i_h, i_r, i_t$ index a _block_ of the head, relation, or tail representation,
    and $s \in {-1, 1}$ is the sign.

    The interaction function is then given as

    .. math::
        \sum_{(i_h, i_r, i_t, s) \in \mathcal{C}} s \cdot \langle h[i_h], r[i_r], t[i_t] \rangle

    where $\langle \cdot, \cdot, \cdot \rangle$ denotes the tri-linear dot product.

    This parametrization allows to express several well-known interaction functions, e.g.

    - :class:`pykeen.nn.DistMultInteraction`:
        one block, $\mathcal{C} = \{(0, 0, 0, 1)\}$
    - :class:`pykeen.nn.ComplExInteraction`:
        two blocks, $\mathcal{C} = \{(0, 0, 0, 1), (0, 1, 1, 1), (1, 0, 1, -1), (1, 0, 1, 1)\}$
    - :class:`pykeen.nn.SimplEInteraction`:
        two blocks: $\mathcal{C} = \{(0, 0, 1, 1), (1, 1, 0, 1)\}$

    While in theory, we can have up to `num_blocks**3` unique triples, usually, a smaller number is preferable to have
    some sparsity.

    ---
    citation:
        author: Zhang
        year: 2020
        arxiv: 1904.11682
        link: https://arxiv.org/abs/1904.11682
        github: AutoML-Research/AutoSF
    """

    #: a description of the block structure
    coefficients: tuple[AutoSFBlock, ...]

    @staticmethod
    def _check_coefficients(
        coefficients: Collection[AutoSFBlock], num_entity_representations: int, num_relation_representations: int
    ):
        """Check coefficients.

        :param coefficients:
            the block description
        :param num_entity_representations:
            the number of entity representations / blocks
        :param num_relation_representations:
            the number of relation representations / blocks

        :raises ValueError:
            if there are duplicate coefficients
        """
        counter = Counter(coef[:3] for coef in coefficients)
        duplicates = {k for k, v in counter.items() if v > 1}
        if duplicates:
            raise ValueError(f"Cannot have duplicates in coefficients! Duplicate entries for {duplicates}")
        for entities, num_blocks in ((True, num_entity_representations), (False, num_relation_representations)):
            missing_ids = set(range(num_blocks)).difference(
                AutoSFInteraction._iter_ids(coefficients, entities=entities)
            )
            if missing_ids:
                label = "entity" if entities else "relation"
                logger.warning(f"Unused {label} blocks: {missing_ids}. This may indicate an error.")

    @staticmethod
    def _iter_ids(coefficients: Collection[AutoSFBlock], entities: bool) -> Iterable[int]:
        """Iterate over selected parts of the blocks.

        :param coefficients:
            the block coefficients
        :param entities:
            whether to select entity or relation ids, i.e., components `(0, 2)` for entities, or `(1,)` for relations.

        :yields: the used indices
        """
        indices = (0, 2) if entities else (1,)
        yield from itt.chain.from_iterable(map(itemgetter(i), coefficients) for i in indices)

    @staticmethod
    def _infer_number(coefficients: Collection[AutoSFBlock], entities: bool) -> int:
        """Infer the number of blocks from the given coefficients.

        :param coefficients:
            the block coefficients
        :param entities:
            whether to select entity or relation ids, i.e., components `(0, 2)` for entities, or `(1,)` for relations.

        :return:
            the inferred number of blocks
        """
        return 1 + max(AutoSFInteraction._iter_ids(coefficients, entities=entities))

    def __init__(
        self,
        coefficients: Iterable[AutoSFBlock],
        *,
        num_blocks: int | None = None,
        num_entity_representations: int | None = None,
        num_relation_representations: int | None = None,
    ) -> None:
        """
        Initialize the interaction function.

        :param coefficients:
            the coefficients for the individual blocks, cf. :class:`pykeen.nn.AutoSFInteraction`

        :param num_blocks:
            the number of blocks. If given, will be used for both, entity and relation representations.
        :param num_entity_representations:
            an explicit number of entity representations / blocks. Only used if `num_blocks` is `None`.
            If `num_entity_representations` is `None`, too, this number if inferred from `coefficients`.
        :param num_relation_representations:
            an explicit number of relation representations / blocks. Only used if `num_blocks` is `None`.
            If `num_relation_representations` is `None`, too, this number if inferred from `coefficients`.
        """
        super().__init__()

        # convert to tuple
        coefficients = tuple(coefficients)

        # infer the number of entity and relation representations
        num_entity_representations = (
            num_blocks or num_entity_representations or self._infer_number(coefficients, entities=True)
        )
        num_relation_representations = (
            num_blocks or num_relation_representations or self._infer_number(coefficients, entities=False)
        )

        # verify coefficients
        self._check_coefficients(
            coefficients=coefficients,
            num_entity_representations=num_entity_representations,
            num_relation_representations=num_relation_representations,
        )

        self.coefficients = coefficients
        # dynamic entity / relation shapes
        self.entity_shape = tuple(["d"] * num_entity_representations)
        self.relation_shape = tuple(["d"] * num_relation_representations)

    @classmethod
    def from_searched_sf(cls, coefficients: Sequence[int], **kwargs) -> AutoSFInteraction:
        """
        Instantiate AutoSF interaction from the "official" serialization format.

        > The first 4 values (a,b,c,d) represent h_1 * r_1 * t_a + h_2 * r_2 * t_b + h_3 * r_3 * t_c + h_4 * r_4 * t_d.
        > For the others, every 4 values represent one adding block: index of r, index of h, index of t, the sign s.

        :param coefficients:
            the coefficients in the "official" serialization format.
        :param kwargs:
            additional keyword-based parameters passed to :meth:`pykeen.nn.AutoSFInteraction.__init__`

        :return:
            An AutoSF interaction module

        .. seealso::
            https://github.com/AutoML-Research/AutoSF/blob/07b7243ccf15e579176943c47d6e65392cd57af3/searched_SFs.txt
        """
        return cls(
            coefficients=[(i, ri, i, 1) for i, ri in enumerate(coefficients[:4])]
            + [(hi, ri, ti, s) for ri, hi, ti, s in more_itertools.chunked(coefficients[4:], 4)],
            **kwargs,
        )

    @staticmethod
    def func(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
        coefficients: Collection[AutoSFBlock],
    ) -> FloatTensor:
        r"""Evaluate an AutoSF-style interaction function as described by [zhang2020]_.

        :param h: each shape: (`*batch_dims`, dim)
            The list of head representations.
        :param r: each shape: (`*batch_dims`, dim)
            The list of relation representations.
        :param t: each shape: (`*batch_dims`, dim)
            The list of tail representations.
        :param coefficients:
            the coefficients, cf. :class:`pykeen.nn.AutoSFInteraction`

        :return: shape: `batch_dims`
            The scores
        """
        return sum(sign * (h[hi] * r[ri] * t[ti]).sum(dim=-1) for hi, ri, ti, sign in coefficients)

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:
        return dict(coefficients=self.coefficients)

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        return dict(zip("hrt", ensure_tuple(h, r, t)))

    def extend(self, *new_coefficients: tuple[int, int, int, Sign]) -> AutoSFInteraction:
        """Extend AutoSF function, as described in the greedy search algorithm in the paper."""
        return AutoSFInteraction(coefficients=self.coefficients + tuple(new_coefficients))

    def latex_visualize(self) -> str:
        """Create the LaTeX + tikz visualization as shown in the paper."""
        n = len(self.entity_shape)
        return "\n".join(
            [
                r"\begin{tikzpicture}[yscale=-1]",
                rf"\draw (0, 0) grid ({n}, {n});",
            ]
            + [
                rf"\draw ({ti}.5, {hi}.5) node {{${'-' if s < 0 else ''}D^r_{{{ri + 1}}}$}};"
                for hi, ri, ti, s in self.coefficients
            ]
            + [
                r"\end{tikzpicture}",
            ],
        )


@parse_docdata
class LineaREInteraction(NormBasedInteraction):
    r"""
    The LineaRE interaction described by [peng2020]_.

    The interaction function is given as

    .. math ::

        \| \mathbf{w}_{r}^{h} \odot \mathbf{x}_{h} + \mathbf{b}_r - \mathbf{w}_{r}^{t} \odot \mathbf{x}_{t} \|

    where $\mathbf{w}_{r}^{h}, \mathbf{b}_r, \mathbf{w}_{r}^{t} \in \mathbb{R}^d$ are relation-specific terms,
    and $\mathbf{x}_{h}, \mathbf{x}_{t} \in \mathbb{R}$ the head and tail entity representation.

    .. note ::
        the original paper only describes the interaction for $L_1$ norm, but we extend it to the general $L_p$
        norm as well as its powered variant.

    .. note ::
        this interaction is equivalent to :class:`TripleREInteraction` without the `u` term

    ---
    name: LineaRE
    citation:
        author: Peng
        year: 2020
        arxiv: 2004.10037
        github: pengyanhui/LineaRE
        link: https://arxiv.org/abs/2004.10037
    """

    # r_head, r_bias, r_tail
    relation_shape = ("d", "d", "d")

    func = pkf.linea_re_interaction

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: FloatTensor,
        r: tuple[FloatTensor, FloatTensor, FloatTensor],
        t: FloatTensor,
    ) -> MutableMapping[str, FloatTensor]:  # noqa: D102
        r_head, r_mid, r_tail = r
        return dict(h=h, r_head=r_head, r_mid=r_mid, r_tail=r_tail, t=t)


interaction_resolver: ClassResolver[Interaction] = ClassResolver.from_subclasses(
    Interaction,
    skip={NormBasedInteraction, FunctionalInteraction, MonotonicAffineTransformationInteraction},
    default=TransEInteraction,
)
