"""Stateful interaction functions."""

from __future__ import annotations

import dataclasses
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
from class_resolver import ClassResolver, Hint, OptionalKwargs, ResolverKey, update_docstring_with_resolver_keys
from class_resolver.contrib.torch import activation_resolver
from docdata import parse_docdata
from torch import nn
from torch.nn.init import xavier_normal_
from typing_extensions import Self

from . import functional as pkf
from .algebra import quaterion_multiplication_table
from .compute_kernel import batched_dot
from .init import initializer_resolver
from .sim import KG2ESimilarity, kg2e_similarity_resolver
from ..metrics.utils import ValueRange
from ..typing import (
    FloatTensor,
    GaussianDistribution,
    HeadRepresentation,
    HintOrType,
    Initializer,
    RelationRepresentation,
    Representation,
    Sign,
    TailRepresentation,
)
from ..utils import (
    add_cudnn_error_hint,
    at_least_eps,
    clamp_norm,
    einsum,
    ensure_complex,
    ensure_tuple,
    estimate_cost_of_sequence,
    make_ones_like,
    negative_norm,
    negative_norm_of_sum,
    project_entity,
    tensor_product,
    tensor_sum,
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
    rs: Sequence[Sequence[FloatTensor]] = ensure_tuple(*representations)
    # get number of head/relation/tail representations
    length = list(map(len, rs))
    splits = numpy.cumsum([0] + length)
    # flatten list
    rsl: Sequence[FloatTensor] = sum(map(list, rs), [])
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
    xs: Sequence[FloatTensor] = upgrade_to_sequence(x)
    xs = [xx.unsqueeze(dim=dim) for xx in xs]
    return xs[0] if len(xs) == 1 else xs


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
    ) -> FloatTensor:
        """Compute broadcasted triple scores given broadcasted representations for head, relation and tails.

        In general, each interaction function (class) expects a certain format for each of head, relation and
        tail representations. This format is composed of the *number* and the shape of the representations.

        Many simple interaction functions such as :class:`~pykeen.nn.modules.TransEInteraction`
        operate on a single representation, however there are also interactions such as
        :class:`~pykeen.nn.modules.TransDInteraction`, which requires two representations for each slot, or
        :class:`~pykeen.nn.modules.PairREInteraction`, which requires two relation representations, but only a single
        representation for head and tail entity respectively.

        Each individual representation has a *shape*. This can be a simple $d$-dimensional vector, but also comprise
        matrices, or even high-order tensors.

        This method supports the general batched calculation, i.e., each of the representations can have a
        preceding batch dimensions. Those batch dimensions do not necessarily need to be exactly the same, but they
        need to be broadcastable. A good explanation of broadcasting rules can be found in
        `NumPy's documentation <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.

        .. seealso::
            - :ref:`representations` for an overview about different ways how to obtain individual representations.

        :param h: shape: ``(*batch_dims, *dims)``
            The head representations.

        :param r: shape: ``(*batch_dims, *dims)``
            The relation representations.

        :param t: shape: ``(*batch_dims, *dims)``
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
    ) -> FloatTensor:
        """Compute broadcasted triple scores with optional slicing.

        .. note ::
            At most one of the slice sizes may be not None.

        .. todo::
            we could change that to slicing along multiple dimensions, if necessary

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
    ) -> FloatTensor:
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
    ) -> FloatTensor:
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
    ) -> FloatTensor:
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
    ) -> FloatTensor:
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
    func: Callable[..., FloatTensor]

    def forward(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> FloatTensor:
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
    ) -> Mapping[str, FloatTensor]:
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
    ) -> MutableMapping[str, FloatTensor]:  # noqa: D102
        """Conversion utility to prepare the h/r/t representations for the functional form."""
        # TODO: we only allow single-tensor representations here, but could easily generalize
        assert all(torch.is_tensor(x) for x in (h, r, t))
        if cls.is_complex:
            h, r, t = ensure_complex(h, r, t)
        return dict(h=h, r=r, t=t)

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:
        """Conversion utility to prepare the state to be passed to the functional form."""
        return dict()


class NormBasedInteraction(Interaction, Generic[HeadRepresentation, RelationRepresentation, TailRepresentation], ABC):
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
    r"""The state-less norm-based TransE interaction function.

    TransE models relations as a translation from head to tail entities in :math:`\textbf{e}`:

    .. math::

        \textbf{e}_h + \textbf{e}_r \approx \textbf{e}_t

    This equation is rearranged and the :math:`l_p` norm is applied to create the TransE interaction function.

    .. math::

        f(h, r, t) = - \|\textbf{e}_h + \textbf{e}_r - \textbf{e}_t\|_{p}

    While this formulation is computationally efficient, it inherently cannot model one-to-many, many-to-one, and
    many-to-many relationships. For triples :math:`(h,r,t_1), (h,r,t_2) \in \mathcal{K}` where :math:`t_1 \neq t_2`,
    the model adapts the embeddings in order to ensure :math:`\textbf{e}_h + \textbf{e}_r \approx \textbf{e}_{t_1}`
    and :math:`\textbf{e}_h + \textbf{e}_r \approx \textbf{e}_{t_2}` which results in
    :math:`\textbf{e}_{t_1} \approx \textbf{e}_{t_2}`.

    ---
    citation:
        author: Bordes
        year: 2013
        link: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
    """

    def forward(self, h: FloatTensor, r: FloatTensor, t: FloatTensor) -> FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims, d)``
            The head representations.
        :param r: shape: ``(*batch_dims, d)``
            The relation representations.
        :param t: shape: ``(*batch_dims, d)``
            The tail representations.

        :return: shape: ``batch_dims``
            The scores.
        """
        return pkf.negative_norm_of_sum(h, r, -t, p=self.p, power_norm=self.power_norm)


@parse_docdata
class TransFInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    r"""The state-less norm-based TransF interaction function.

    It is given by

    .. math ::
        f(\mathbf{h}, \mathbf{r}, \mathbf{t}) =
            (\mathbf{h} + \mathbf{r})^T \mathbf{t} + \mathbf{h}^T (\mathbf{r} - \mathbf{t})

    for head entity, relation, and tail entity representations $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$.
    The interaction function can be simplified as

    .. math ::
        f(\mathbf{h}, \mathbf{r}, \mathbf{t}) &=&
            (\mathbf{h} + \mathbf{r})^T \mathbf{t} + \mathbf{h}^T (\mathbf{t} - \mathbf{r}) \\
            &=&
            \langle \mathbf{h}, \mathbf{t}\rangle
            + \langle \mathbf{r}, \mathbf{t}\rangle
            + \langle \mathbf{h}, \mathbf{t}\rangle
            - \langle \mathbf{h}, \mathbf{r}\rangle \\
            &=&
            2 \cdot \langle \mathbf{h}, \mathbf{t}\rangle
            + \langle \mathbf{r}, \mathbf{t}\rangle
            - \langle \mathbf{h}, \mathbf{r}\rangle

    .. note ::
        This is the *balanced* variant from the paper.

    .. todo ::
        Implement the unbalanced version, too:
        $f(\mathbf{h}, \mathbf{r}, \mathbf{t}) = (\mathbf{h} + \mathbf{r})^T \mathbf{t}$

    ---
    citation:
        author: Feng
        year: 2016
        link: https://www.aaai.org/ocs/index.php/KR/KR16/paper/view/12887
        arxiv: 1505.05253
    """

    # TODO: implement the unbalanced variant from the paper: f(h, r, t) = (h + r)^T t

    def forward(self, h: FloatTensor, r: FloatTensor, t: FloatTensor) -> FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims, d)``
            The head representations.
        :param r: shape: ``(*batch_dims, d)``
            The relation representations.
        :param t: shape: ``(*batch_dims, d)``
            The tail representations.

        :return: shape: ``batch_dims``
            The scores.
        """
        return tensor_sum(2 * batched_dot(h, t), batched_dot(r, t), -batched_dot(h, r))


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


@dataclasses.dataclass
class ConvEResolvedImageShape:
    """The resolved shape of the ConvE 'image'."""

    dim: int
    width: int
    height: int
    channels: int

    @property
    def is_valid(self) -> bool:
        """Determine whether the given shape is a valid factorization of the embedding dimension."""
        return self.channels * self.width * self.height == self.dim

    @classmethod
    def make(cls, channels: int | None, dim: int | None, height: int | None, width: int | None) -> Self:
        """
        Automatically calculates missing dimensions for ConvE.

        The dimensions need to fulfil $channels * height * width = dim$.

        :param channels:
            the number of input channels
        :param dim:
            the embedding dimension
        :param height:
            the "image" height
        :param width:
            the "image" width

        :return:
            a resolve shape information.

        :raises ValueError:
            when the constraints cannot be satisfied.
        """
        if dim is None:
            if channels is None or width is None or height is None:
                raise ValueError(
                    f"When {dim=} none of the other dimensions may be None, "
                    f"but {channels=}, {width=}, and {height=}"
                )
            dim = channels * width * height

        # All are None -> try and make closest to square
        if channels is None and width is None and height is None:
            result_sqrt = math.floor(math.sqrt(dim))
            height = max(factor for factor in range(1, result_sqrt + 1) if dim % factor == 0)
            width = dim // height
            return cls(dim=dim, width=width, height=height, channels=1)

        # Only input channels is None
        if channels is None and width is not None and height is not None:
            return cls(dim=dim, width=width, height=height, channels=dim // (width * height))

        # Only width is None
        if channels is not None and width is None and height is not None:
            return cls(dim=dim, width=dim // (height * channels), height=height, channels=channels)

        # Only height is none
        if height is None and width is not None and channels is not None:
            return cls(dim=dim, width=width, height=dim // (width * channels), channels=channels)

        # Height and input_channels are None -> set input_channels to 1 and calculage height
        if channels is None and height is None and width is not None:
            return cls(dim=dim, width=width, height=dim // width, channels=1)

        # Width and input channels are None -> set input channels to 1 and calculate width
        if channels is None and height is not None and width is None:
            return cls(dim=dim, width=dim // height, height=height, channels=1)

        raise ValueError(f"Could not resolve {channels=}, {height=}, {width=} = {dim=}.")


@dataclasses.dataclass
class ConvEShapeInformation:
    """Resolved ConvE shape information."""

    #: the embedding dimension
    embedding_dim: int

    #: the number of input channels of the convolution
    input_channels: int

    #: the embedding "image" height
    image_height: int

    #: the embedding "image" width
    image_width: int

    #: the number of output channels of the convolution
    output_channels: int

    #: the convolution kernel height
    kernel_height: int

    #: the convolution kernel width
    kernel_width: int

    @property
    def num_in_features(self) -> int:
        """The number of input features to the linear layer."""
        return (
            self.output_channels
            * (2 * self.image_height - self.kernel_height + 1)
            * (self.image_width - self.kernel_width + 1)
        )

    @classmethod
    def make(
        cls,
        embedding_dim: int | None,
        image_width: int | None = None,
        image_height: int | None = None,
        input_channels: int | None = None,
        output_channels: int = 32,
        kernel_width: int = 3,
        kernel_height: int | None = None,
    ) -> Self:
        """Automatically calculates missing dimensions for ConvE.

        :param embedding_dim:
            The embedding dimension.
        :param image_width:
            The width of the embedding "image".
        :param image_height:
            The height of the embedding "image".
        :param input_channels:
            The number of input channels for the convolution.
        :param output_channels:
            The number of output channels for the convolution.
        :param kernel_width:
            The width of the convolution kernel.
        :param kernel_height:
            The height of the convolution kernel.

        :return: Fully resolve shapes.

        :raises ValueError:
            If no factorization could be found.
        """
        # resolve image shape
        logger.info(f"Resolving {input_channels} * {image_width} * {image_height} = {embedding_dim}.")
        # Store initial input for error message
        original = (input_channels, image_width, image_height)
        # infer open dimensions from the remainder
        image_shape = ConvEResolvedImageShape.make(
            dim=embedding_dim,
            height=image_height,
            width=image_width,
            channels=input_channels,
        )
        if not image_shape.is_valid:
            raise ValueError(f"Could not resolve {original} to a valid factorization of {embedding_dim}.")
        # resolve kernel size defaults
        kernel_height = kernel_height or kernel_width
        return cls(
            embedding_dim=image_shape.dim,
            input_channels=image_shape.channels,
            image_width=image_shape.width,
            image_height=image_shape.height,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            output_channels=output_channels,
        )


@parse_docdata
class ConvEInteraction(Interaction[FloatTensor, FloatTensor, tuple[FloatTensor, FloatTensor]]):
    r"""The stateful ConvE interaction function.

    ConvE is a CNN-based approach. For input representations $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$,
    it first combines $\mathbf{h}$ and $\mathbf{r}$ into a matrix matrix $\mathbf{A} \in \mathbb{R}^{2 \times d}$,
    where the first row of $\mathbf{A}$ represents $\mathbf{h}$ and the second row represents $\mathbf{r}$.
    $\mathbf{A}$ is reshaped to a matrix $\mathbf{B} \in \mathbb{R}^{m \times n}$
    where the first $m/2$ half rows represent $\mathbf{h}$ and the remaining $m/2$ half rows represent $\mathbf{r}$.
    In the convolution layer, a set of *2-dimensional* convolutional filters
    $\Omega = \{\omega_i \mid \omega_i \in \mathbb{R}^{r \times c}\}$ are applied on $\mathbf{B}$
    that capture interactions between $\mathbf{h}$ and $\mathbf{r}$.
    The resulting feature maps are reshaped and concatenated in order to create a feature vector
    $\mathbf{v} \in \mathbb{R}^{|\Omega|rc}$.
    In the next step, $\mathbf{v}$ is mapped into the entity space using a linear transformation
    $\mathbf{W} \in \mathbb{R}^{|\Omega|rc \times d}$, that is $\mathbf{e}_{h,r} = \mathbf{v}^{T} \mathbf{W}$.
    The  score is then obtained by:

    .. math::

        f(\mathbf{h}, \mathbf{r}, \mathbf{t}) = \mathbf{e}_{h,r} \mathbf{t}

    Since the interaction model can be decomposed into
    $f(\mathbf{h}, \mathbf{r}, \mathbf{t}) = \left\langle f'(\mathbf{h}, \mathbf{r}), \mathbf{t} \right\rangle$
    the model is particularly designed to 1-N scoring, i.e. efficient computation of scores for
    $(h,r,t)$ for fixed $h,r$ and many different $t$.

    The default setting uses batch normalization. Batch normalization normalizes the output of the activation functions,
    in order to ensure that the weights of the NN don't become imbalanced and to speed up training.
    However, batch normalization is not the only way to achieve more robust and effective training [santurkar2018]_.
    Therefore, we added the flag ``apply_batch_normalization`` to turn batch normalization on/off (it's turned on as
    default).

    ---
    citation:
        author: Dettmers
        year: 2018
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17366
        github: TimDettmers/ConvE
        arxiv: 1707.01476
    """

    # vector & scalar offset
    _tail_entity_shape = ("d", "")

    #: The head-relation encoder operating on 2D "images"
    hr2d: nn.Module

    #: The head-relation encoder operating on the 1D flattened version
    hr1d: nn.Module

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

        # Parameter need to fulfil:
        #   input_channels * embedding_height * embedding_width = embedding_dim
        self.shape_info = ConvEShapeInformation.make(
            embedding_dim=embedding_dim,
            input_channels=input_channels,
            image_width=embedding_width,
            image_height=embedding_height,
            kernel_width=kernel_width,
            kernel_height=kernel_height,
            output_channels=output_channels,
        )

        # encoders
        # 1: 2D encoder: BN?, DO, Conv, BN?, Act, DO
        hr2d_layers = [
            nn.BatchNorm2d(self.shape_info.input_channels) if apply_batch_normalization else None,
            nn.Dropout(input_dropout),
            nn.Conv2d(
                in_channels=self.shape_info.input_channels,
                out_channels=self.shape_info.output_channels,
                kernel_size=(self.shape_info.kernel_height, self.shape_info.kernel_width),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(self.shape_info.output_channels) if apply_batch_normalization else None,
            nn.ReLU(),
            nn.Dropout2d(feature_map_dropout),
        ]
        self.hr2d = nn.Sequential(*(layer for layer in hr2d_layers if layer is not None))

        # 2: 1D encoder: FC, DO, BN?, Act
        hr1d_layers = [
            nn.Linear(self.shape_info.num_in_features, self.shape_info.embedding_dim),
            nn.Dropout(output_dropout),
            nn.BatchNorm1d(self.shape_info.embedding_dim) if apply_batch_normalization else None,
            nn.ReLU(),
        ]
        self.hr1d = nn.Sequential(*(layer for layer in hr1d_layers if layer is not None))

    @add_cudnn_error_hint
    def forward(
        self,
        h: FloatTensor,
        r: FloatTensor,
        t: tuple[FloatTensor, FloatTensor],
    ) -> FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims, d)``
            The head representations.
        :param r: shape: ``(*batch_dims, d)``
            The relation representations.
        :param t: two vectors of shape: ``(*batch_dims, d)`` and ``batch_dims``
            The tail representations, comprising the tail entity embedding and bias.

        :return: shape: ``batch_dims``
            The scores.
        """
        t_emb, t_bias = t

        # repeat if necessary, and concat head and relation
        # shape: -1, num_input_channels, 2*height, width
        x = torch.cat(
            torch.broadcast_tensors(
                h.view(
                    *h.shape[:-1],
                    self.shape_info.input_channels,
                    self.shape_info.image_height,
                    self.shape_info.image_width,
                ),
                r.view(
                    *r.shape[:-1],
                    self.shape_info.input_channels,
                    self.shape_info.image_height,
                    self.shape_info.image_width,
                ),
            ),
            dim=-2,
        )
        prefix_shape = x.shape[:-3]
        x = x.view(-1, self.shape_info.input_channels, 2 * self.shape_info.image_height, self.shape_info.image_width)

        # shape: -1, num_input_channels, 2*height, width
        x = self.hr2d(x)

        # -1, num_output_channels * (2 * height - kernel_height + 1) * (width - kernel_width + 1)
        x = x.view(-1, self.shape_info.num_in_features)
        x = self.hr1d(x)

        # reshape: (-1, dim) -> (*batch_dims, dim)
        x = x.view(*prefix_shape, h.shape[-1])

        # For efficient calculation, each of the convolved [h, r] rows has only to be multiplied with one t row
        # output_shape: batch_dims
        x = einsum("...d, ...d -> ...", x, t_emb)

        # add bias term
        return x + t_bias


@parse_docdata
class ConvKBInteraction(Interaction[FloatTensor, FloatTensor, FloatTensor]):
    r"""The stateful ConvKB interaction function.

    ConvKB uses a convolutional neural network (CNN) whose feature maps capture global interactions of the input.

    For given input representations for head entity, relation and tail entity, denoted by
    $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$, it first combines them to a matrix
    $\mathbf{A} = [\mathbf{h}; \mathbf{r}; \mathbf{t}] \in \mathbb{R}^{d \times 3}$.

    In the convolution layer, a set of convolutional filters
    $\omega_i \in \mathbb{R}^{1 \times 3}$, $i=1, \dots, \tau,$ are applied on the input in order to compute for
    each dimension global interactions of the embedded triple. Each $\omega_i$ is applied on every row of
    $\mathbf{A}$ creating a feature map $\mathbf{v}_i = [v_{i,1},...,v_{i,d}] \in \mathbb{R}^d$:

    .. math::

        \mathbf{v}_i = g(\omega_j \mathbf{A} + \mathbf{b})

    where $\mathbf{b} \in \mathbb{R}$ denotes a bias term and $g$ an activation function which is employed element-wise.
    Based on the resulting feature maps $\mathbf{v}_1, \dots, \mathbf{v}_{\tau}$, the plausibility score of a triple
    is given by:

    .. math::

        f(h,r,t) = [\mathbf{v}_i; \ldots ;\mathbf{v}_\tau] \cdot \mathbf{w}

    where $[\mathbf{v}_i; \ldots ;\mathbf{v}_\tau] \in \mathbb{R}^{\tau d \times 1}$ and
    $\mathbf{w} \in \mathbb{R}^{\tau d \times 1}$ is a shared weight vector.

    ConvKB may be seen as a restriction of :class:`~pykeen.nn.modules.ERMLPInteraction` with a certain weight sharing
    pattern in the first layer.

    ---
    citation:
        author: Nguyen
        year: 2018
        link: https://www.aclweb.org/anthology/N18-2053
        github: daiquocnguyen/ConvKB
        arxiv: 1712.02121
    """

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

    def forward(self, h: FloatTensor, r: FloatTensor, t: FloatTensor) -> FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims, d)``
            The head representations.
        :param r: shape: ``(*batch_dims, d)``
            The relation representations.
        :param t: shape: ``(*batch_dims, d)``
            The tail representations.

        :return: shape: ``batch_dims``
            The scores.
        """
        # cat into shape (..., 1, d, 3)
        x = torch.stack(torch.broadcast_tensors(h, r, t), dim=-1).unsqueeze(dim=-3)
        s = x.shape
        x = x.view(-1, *s[-3:])
        x = self.conv(x)
        x = x.view(*s[:-3], -1)
        x = self.activation(x)

        # Apply dropout, cf. https://github.com/daiquocnguyen/ConvKB/blob/master/model.py#L54-L56
        x = self.hidden_dropout(x)

        # Linear layer for final scores; use flattened representations, shape: (*batch_dims, d * f)
        x = self.linear(x)
        return x.squeeze(dim=-1)


@parse_docdata
class DistMultInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    r"""The stateless DistMult interaction function.

    This interaction is given by

    .. math::

        f(\mathbf{h}, \mathbf{r}, \mathbf{t}) = \sum \limits_{i} \mathbf{h}_i \cdot \mathbf{r}_{i} \cdot \mathbf{t}_i

    where $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^{d}$ are the representations for the head entity,
    the relation, and the tail entity.

    For a single triple of $d$-dimensional vectors, the computational complexity is given as $\mathcal{O}(d)$.

    The interaction function is symmetric in the entities, i.e.,

    .. math::

        f(h, r, t) = f(t, r, h)

    ---
    citation:
        author: Yang
        year: 2014
        link: https://arxiv.org/abs/1412.6575
        arxiv: 1412.6575
    """

    @staticmethod
    def func(h: FloatTensor, r: FloatTensor, t: FloatTensor) -> FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims, d)``
            The head representations.
        :param r: shape: ``(*batch_dims, d)``
            The relation representations.
        :param t: shape: ``(*batch_dims, d)``
            The tail representations.

        :return: shape: ``batch_dims``
            The scores.
        """
        return tensor_product(h, r, t).sum(dim=-1)


@parse_docdata
class DistMAInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    r"""The stateless DistMA interaction function from [shi2019]_.

    For head entity, relation, and tail representations $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$,
    the interaction functions is given by

    .. math ::
          \langle \mathbf{h}, \mathbf{r}\rangle
        + \langle \mathbf{r}, \mathbf{t}\rangle
        + \langle \mathbf{h}, \mathbf{t}\rangle

    .. note ::
        This interaction function is the symmetric part $E_1$ from the respective paper, and not the combination
        with :class:`~pykeen.nn.modules.ComplExInteraction`.

    ---
    citation:
        author: Shi
        year: 2019
        link: https://www.aclweb.org/anthology/D19-1075.pdf
    """

    @staticmethod
    def func(h: FloatTensor, r: FloatTensor, t: FloatTensor) -> FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims, d)``
            The head representations.
        :param r: shape: ``(*batch_dims, d)``
            The relation representations.
        :param t: shape: ``(*batch_dims, d)``
            The tail representations.

        :return: shape: ``batch_dims``
            The scores.
        """
        return batched_dot(h, r) + batched_dot(r, t) + batched_dot(h, t)


@parse_docdata
class ERMLPInteraction(Interaction[FloatTensor, FloatTensor, FloatTensor]):
    r"""The ER-MLP stateful interaction function.

    ER-MLP uses a multi-layer perceptron based approach with a single hidden layer.
    The $d$-dimensional representations of head entity, relation, and tail entity are concatenated
    and passed to the hidden layer. The output-layer consists of a single neuron that computes the plausibility score:

    .. math::

        f(\mathbf{h}, \mathbf{r}, \mathbf{t}) = \mathbf{w}^{T} g(\mathbf{W} [\mathbf{h}; \mathbf{r}; \mathbf{t}]),

    where $\textbf{W} \in \mathbb{R}^{k \times 3d}$ represents the weight matrix of the hidden layer,
    $\textbf{w} \in \mathbb{R}^{k}$, the weights of the output layer, and $g$ denotes an activation function such
    as the hyperbolic tangent.

    ---
    name: ER-MLP
    citation:
        author: Dong
        year: 2014
        link: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45634.pdf
    """

    @update_docstring_with_resolver_keys(
        ResolverKey(name="activation", resolver="class_resolver.contrib.torch.activation_resolver")
    )
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int | None = None,
        activation: HintOrType[nn.Module] = nn.ReLU,
        activation_kwargs: OptionalKwargs = None,
    ):
        """Initialize the interaction module.

        :param embedding_dim:
            The embedding vector dimension for entities and relations.
        :param hidden_dim:
            The hidden dimension of the MLP. Defaults to `embedding_dim`.
        :param activation:
            The activation function or a hint thereof.
        :param activation_kwargs:
            Additional keyword-based parameters passed to the activation's constructor, if the activation is not
            pre-instantiated.
        """
        super().__init__()
        # normalize hidden_dim
        hidden_dim = hidden_dim or embedding_dim
        self.hidden = nn.Linear(in_features=3 * embedding_dim, out_features=hidden_dim, bias=True)
        self.activation = activation_resolver.make(activation, activation_kwargs)
        self.hidden_to_score = nn.Linear(in_features=hidden_dim, out_features=1, bias=True)

    def forward(self, h: FloatTensor, r: FloatTensor, t: FloatTensor) -> FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims, d)``
            The head representations.
        :param r: shape: ``(*batch_dims, d)``
            The relation representations.
        :param t: shape: ``(*batch_dims, d)``
            The tail representations.

        :return: shape: ``batch_dims``
            The scores.
        """
        # shortcut for same shape
        if h.shape == r.shape and h.shape == t.shape:
            x = self.hidden(torch.cat([h, r, t], dim=-1))
        else:
            # split weight into head-/relation-/tail-specific sub-matrices
            *prefix, dim = h.shape
            x = tensor_sum(
                self.hidden.bias.view(*make_ones_like(prefix), -1),
                *(
                    einsum("...i, ji -> ...j", xx, weight)
                    for xx, weight in zip([h, r, t], self.hidden.weight.split(split_size=dim, dim=-1))
                ),
            )
        return self.hidden_to_score(self.activation(x)).squeeze(dim=-1)

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
class ERMLPEInteraction(Interaction[FloatTensor, FloatTensor, FloatTensor]):
    r"""The stateful ER-MLP (E) interaction function.

    This interaction uses a neural network-based approach similar to ER-MLP and with slight modifications.
    In :class:`~pykeen.nn.modules.ERMLPInteraction`, the interaction is:

    .. math::

        f(h, r, t) = \textbf{w}^{T} g(\textbf{W} [\textbf{h}; \textbf{r}; \textbf{t}])

    whereas here it is:

    .. math::

        f(h, r, t) = \textbf{t}^{T} f(\textbf{W} (g(\textbf{W} [\textbf{h}; \textbf{r}]))

    including dropouts and batch-norms between each two hidden layers. Thus,
    :class:`~pykeen.nn.modules.ConvEInteraction` can be seen as a special case of ERMLP (E).

    ---
    name: ER-MLP (E)
    citation:
        author: Sharifzadeh
        year: 2019
        link: https://github.com/pykeen/pykeen
        github: pykeen/pykeen
    """

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

    def forward(self, h: FloatTensor, r: FloatTensor, t: FloatTensor) -> FloatTensor:
        """Compute broadcasted triple scores given broadcasted representations for head, relation and tails.

        :param h: shape: ``(*batch_dims, d)``
            The head representations.
        :param r: shape: ``(*batch_dims, d)``
            The relation representations.
        :param t: shape: ``(*batch_dims, d)``
            The tail representations.

        :return: shape: ``batch_dims``
            The scores.
        """
        # repeat if necessary, and concat head and relation, (*batch_dims, 2 * embedding_dim)
        x = torch.cat(torch.broadcast_tensors(h, r), dim=-1)

        # Predict t embedding, shape: (*batch_dims, d)
        *batch_dims, dim = x.shape
        x = self.mlp(x.view(-1, dim)).view(*batch_dims, -1)

        # dot product
        return einsum("...d,...d->...", x, t)


@parse_docdata
class TransRInteraction(NormBasedInteraction[FloatTensor, tuple[FloatTensor, FloatTensor], FloatTensor]):
    r"""The state-less norm-based TransR interaction function.

    It is given by

    .. math ::

        -\|c(\mathbf{M}_{r}\mathbf{h}) + \mathbf{r} - c(\mathbf{M}_{r}\mathbf{t})\|_{2}^2

    for head and tail entity representations $\mathbf{h}, \mathbf{t} \in \mathbb{R}^d$,
    relation representation $\mathbf{r} \in \mathbb{R}^k$,
    and a relation-specific projection matrix $\mathbf{M}_r \in \mathbb{R}^{k \times d}$.
    $c$ enforces the constraint $\|\cdot\| \leq 1$, cf. :func:`pykeen.utils.clamp_norm`.

    .. note ::
        :class:`pykeen.models.TransR` additionally also enforces $\|\cdot\| \leq 1$ on all embeddings.

    ---
    citation:
        author: Lin
        year: 2015
        link: https://aaai.org/papers/9491-learning-entity-and-relation-embeddings-for-knowledge-graph-completion/
    """

    relation_shape = ("e", "de")

    def __init__(self, p: int, power_norm: bool = True, max_projection_norm: float = 1.0):
        """
        Initialize the interaction module.

        :param p:
            The $p$ value of the norm to use.
        :param power_norm:
            Whether to use the $p$-th power of the $p$-norm.
        :param max_projection_norm:
            The maximum norm to be clamped after projection.

        ``p`` and ``power_norm`` are passed to :class:`~pykeen.nn.modules.NormBasedInteraction`.
        """
        super().__init__(p=p, power_norm=power_norm)
        self.max_projection_norm = max_projection_norm

    def forward(self, h: FloatTensor, r: tuple[FloatTensor, FloatTensor], t: FloatTensor) -> FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims, d)``
            The head representations.
        :param r: shape: ``(*batch_dims, k)`` and ``(*batch_dims, d, k)``
            The relation representations.
        :param t: shape: ``(*batch_dims, d)``
            The tail representations.

        :return: shape: ``batch_dims``
            The scores.
        """
        r, m_r = r
        # project to relation specific subspace
        h_bot = einsum("...e, ...er -> ...r", h, m_r)
        t_bot = einsum("...e, ...er -> ...r", t, m_r)
        # ensure constraints
        h_bot = clamp_norm(h_bot, p=self.p, dim=-1, maxnorm=self.max_projection_norm)
        t_bot = clamp_norm(t_bot, p=self.p, dim=-1, maxnorm=self.max_projection_norm)
        return pkf.negative_norm_of_sum(h_bot, r, -t_bot, p=self.p, power_norm=self.power_norm)


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
    r"""The stateless HolE interaction function.

    Holographic embeddings (HolE) make use of the circular correlation operator to compute interactions between
    latent features of entities and relations:

    .. math::

        f(h,r,t) = \textbf{r}^{T}(\textbf{h} \star \textbf{t})

    where the circular correlation $\star: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}^d$ is defined as:

    .. math::

        [\textbf{a} \star \textbf{b}]_i = \sum_{k=0}^{d-1} \textbf{a}_{k} * \textbf{b}_{(i+k)\ mod \ d}

    By using the correlation operator each component $[\textbf{h} \star \textbf{t}]_i$ represents a sum over a
    fixed partition over pairwise interactions. This enables the model to put semantic similar interactions into the
    same partition and share weights through $\textbf{r}$. Similarly irrelevant interactions of features could also
    be placed into the same partition which could be assigned a small weight in $\textbf{r}$.

    ---
    citation:
        author: Nickel
        year: 2016
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828
        github: mnick/holographic-embeddings
        arxiv: 1510.04935
    """

    @staticmethod
    def func(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims, d)``
            The head representations.
        :param r: shape: ``(*batch_dims, d)``
            The relation representations.
        :param t: shape: ``(*batch_dims, d)``
            The tail representations.

        :return: shape: ``batch_dims``
            The scores.
        """
        # composite: (*batch_dims, d)
        composite = pkf.circular_correlation(h, t)

        # inner product with relation embedding
        return (r * composite).sum(dim=-1)


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
class SEInteraction(NormBasedInteraction[FloatTensor, tuple[FloatTensor, FloatTensor], FloatTensor]):
    r"""The Structured Embedding (SE) interaction function.

    SE applies role- and relation-specific projection matrices
    $\textbf{M}_{r}^{h}, \textbf{M}_{r}^{t} \in \mathbb{R}^{d \times d}$ to the head and tail
    entities' representations $\mathbf{h}, \mathbf{t} \in \mathbb{R}^d$ before computing their distance.

    .. math::

        f(\textbf{h}, (\textbf{M}_{r}^{h}, \textbf{M}_{r}^{t}), \textbf{t})
            = -\|\textbf{M}_{r}^{h} \textbf{h}  - \textbf{M}_{r}^{t} \textbf{t}\|_p

    ---
    name: Structured Embedding
    citation:
        author: Bordes
        year: 2011
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/download/3659/3898
    """

    relation_shape = ("dd", "dd")

    def forward(self, h: FloatTensor, r: FloatTensor, t: FloatTensor) -> FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims, d)``
            The head representations.
        :param r: shape: ``(*batch_dims, d, d)`` and ``(*batch_dims, d, d)``.
            The relation representations.
        :param t: shape: ``(*batch_dims, d)``
            The tail representations.

        :return: shape: ``batch_dims``
            The scores.
        """
        r_h, r_t = r
        # projections
        p_h = einsum("...rd,...d->...r", r_h, h)
        p_t = einsum("...rd,...d->...r", r_t, t)
        return negative_norm(p_h - p_t, p=self.p, power_norm=self.power_norm)


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
    NormBasedInteraction[FloatTensor, None, FloatTensor],
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
    ) -> MutableMapping[str, FloatTensor]:  # noqa: D102
        return dict(h=h, t=t)


@parse_docdata
class TorusEInteraction(NormBasedInteraction[FloatTensor, FloatTensor, FloatTensor]):
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
        tuple[FloatTensor, FloatTensor], tuple[FloatTensor, FloatTensor], tuple[FloatTensor, FloatTensor]
    ]
):
    r"""The TransD interaction function.

    TransD is an extension of :class:`~pykeen.nn.modules.TransRInteraction` that, like TransR,
    considers entities and relations as objects living in different vector spaces.
    However, instead of performing the same relation-specific projection for all entity embeddings,
    entity-relation-specific projection matrices
    $\mathbf{M}_{r, h}, \mathbf{M}_{t, h} \in \mathbb{R}^{k \times d}$
    are constructed.

    To do so, all head entities, tail entities, and relations are represented by two vectors,
    $\mathbf{h}_v, \mathbf{h}_p, \mathbf{t}_v, \mathbf{t}_p \in \mathbb{R}^d$
    and $\mathbf{r}_v, \mathbf{r}_v \in \mathbb{R}^k$, respectively.

    The first set of representations is used for calculating the entity-relation-specific projection matrices:

    .. math::

        \mathbf{M}_{r, h} &=& \mathbf{r}_p \mathbf{h}_p^{T} + \tilde{\mathbf{I}}

        \mathbf{M}_{r, t} &=& \mathbf{r}_p \mathbf{t}_p^{T} + \tilde{\mathbf{I}}

    where $\tilde{\textbf{I}} \in \mathbb{R}^{k \times d}$ is a $k \times d$ matrix with ones on the diagonal and
    zeros elsewhere. Next, $\mathbf{h}_v$ and $\mathbf{t}_v$ are projected into the relation space by means of the
    constructed projection matrices, before calculating a distance similar to
    :class:`~pykeen.nn.modules.TransEInteraction`:

    .. math::

        -\|c(\mathbf{M}_{r, h} \mathbf{h}_v) + \mathbf{r}_v - c(\mathbf{M}_{r, t} \mathbf{t}_v)\|_{2}^2

    where $c$ enforces the constraint $\|\cdot\| \leq 1$.

    .. note ::
        :class:`~pykeen.models.TransD` additionally enforces $\|\mathbf{h}\|, \|\mathbf{r}\|, \|\mathbf{t}\| \leq 1$.

    ---
    citation:
        author: Ji
        year: 2015
        link: http://www.aclweb.org/anthology/P15-1067
    """

    entity_shape = ("d", "d")
    relation_shape = ("e", "e")

    def __init__(self, p: int = 2, power_norm: bool = True):
        """
        Initialize the interaction module.

        .. seealso::

            The parameters ``p`` and ``power_norm`` are directly passed to
            :class:`~pykeen.nn.modules.NormBasedInteraction`

        :param p:
            The $p$ value of the norm to use.
        :param power_norm:
            Whether to use the $p$-th power of the $p$-norm.
        """
        super().__init__(p=p, power_norm=power_norm)

    def forward(
        self, h: tuple[FloatTensor, FloatTensor], r: tuple[FloatTensor, FloatTensor], t: tuple[FloatTensor, FloatTensor]
    ) -> FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims, d)`` and ``(*batch_dims, d)``
            The head representations.
        :param r: shape: ``(*batch_dims, e)`` and ``(*batch_dims, e)``
            The relation representations.
        :param t: shape: ``(*batch_dims, d)`` and ``(*batch_dims, d)``
            The tail representations.

        :return: shape: ``batch_dims``
            The scores.
        """
        h_emb, h_p = h
        r_emb, r_p = r
        t_emb, t_p = t
        # Project entities
        h_bot = project_entity(e=h_emb, e_p=h_p, r_p=r_p)
        t_bot = project_entity(e=t_emb, e_p=t_p, r_p=r_p)
        return negative_norm_of_sum(h_bot, r_emb, -t_bot, p=self.p, power_norm=self.power_norm)


@parse_docdata
class NTNInteraction(
    FunctionalInteraction[
        FloatTensor,
        tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor],
        FloatTensor,
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
        h: FloatTensor,
        r: tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor],
        t: FloatTensor,
    ) -> MutableMapping[str, FloatTensor]:  # noqa: D102
        w, vh, vt, b, u = r
        return dict(h=h, t=t, w=w, b=b, u=u, vh=vh, vt=vt)

    # docstr-coverage: inherited
    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:  # noqa: D102
        return dict(activation=self.non_linearity)


@parse_docdata
class KG2EInteraction(
    Interaction[tuple[FloatTensor, FloatTensor], tuple[FloatTensor, FloatTensor], tuple[FloatTensor, FloatTensor]]
):
    r"""The stateless KG2E interaction function.

    Inspired by :class:`~pykeen.nn.modules.TransEInteraction`, relations are modeled as transformations
    from head to tail entities $\mathcal{H} - \mathcal{T} \approx \mathcal{R}$, where

    .. math ::

        \mathcal{H} \sim \mathcal{N}(\mu_h, \Sigma_h)\\
        \mathcal{T} \sim \mathcal{N}(\mu_t, \Sigma_t)\\
        \mathcal{R} \sim \mathcal{N}(\mu_r, \Sigma_r)

    and thus, since head and tail entities are considered independent with respect to the relations,

    .. math ::

        \mathcal{P}_e = \mathcal{H} - \mathcal{T} \sim \mathcal{N}(\mu_h - \mu_t, \Sigma_h + \Sigma_t)

    To obtain scores, the interaction measures the similarity between $\mathcal{P}_e$ and
    $\mathcal{P}_r = \mathcal{N}(\mu_r, \Sigma_r)$, either by means of the (asymmetric)
    :class:`~pykeen.nn.sim.NegativeKullbackLeiblerDivergence`, or a symmetric variant with
    :class:`~pykeen.nn.sim.ExpectedLikelihood`.

    .. note ::
        This interaction module does *not* sub-class from :class:`~pykeen.nn.modules.FunctionalInteraction`
        just for the technical reason that the choice of the similarity represents some "state". However, it
        does not contain any trainable parameters.

    ---
    citation:
        author: He
        year: 2015
        link: https://dl.acm.org/doi/10.1145/2806416.2806502
    """

    entity_shape = ("d", "d")
    relation_shape = ("d", "d")
    similarity: KG2ESimilarity

    @update_docstring_with_resolver_keys(
        ResolverKey(name="similarity", resolver="pykeen.nn.sim.kg2e_similarity_resolver")
    )
    def __init__(self, similarity: HintOrType[KG2ESimilarity] | None = None, similarity_kwargs: OptionalKwargs = None):
        """
        Initialize the interaction module.

        :param similarity:
            The similarity measures for gaussian distributions. Defaults to
            :class:`~pykeen.nn.sim.NegativeKullbackLeiblerDivergence`.
        :param similarity_kwargs:
            Additional keyword-based parameters used to instantiate the similarity.
        """
        super().__init__()
        self.similarity = kg2e_similarity_resolver.make(similarity, similarity_kwargs)

    def forward(
        self, h: tuple[FloatTensor, FloatTensor], r: tuple[FloatTensor, FloatTensor], t: tuple[FloatTensor, FloatTensor]
    ) -> FloatTensor:
        """Evaluate the interaction function.

        :param h: both shape: (`*batch_dims`, `d`)
            The head representations, mean and (diagonal) variance.
        :param r: shape: (`*batch_dims`, `d`)
            The relation representations, mean and (diagonal) variance.
        :param t: shape: (`*batch_dims`, `d`)
            The tail representations, mean and (diagonal) variance.

        :return: shape: batch_dims
            The scores.
        """
        h_mean, h_var = h
        r_mean, r_var = r
        t_mean, t_var = t
        return self.similarity(
            h=GaussianDistribution(mean=h_mean, diagonal_covariance=h_var),
            r=GaussianDistribution(mean=r_mean, diagonal_covariance=r_var),
            t=GaussianDistribution(mean=t_mean, diagonal_covariance=t_var),
        )


@parse_docdata
class TransHInteraction(NormBasedInteraction[FloatTensor, tuple[FloatTensor, FloatTensor], FloatTensor]):
    r"""The norm-based TransH interaction function.

    This model extends :class:`~pykeen.models.TransEInteraction` by applying the translation from head to tail entity
    in a relation-specific hyperplane in order to address its inability to model one-to-many, many-to-one, and
    many-to-many relations.

    In TransH, each relation is represented by a hyperplane, or more specifically a normal vector of this hyperplane
    $\mathbf{r}_{w} \in \mathbb{R}^d$ and a vector $\mathbf{r}_{d} \in \mathbb{R}^d$ that lies in the hyperplane.
    To obtain a plausibility score, the head representation $\mathbf{h} \in \mathbb{R}^d$,
    and the tail embedding $\mathbf{t} \in \mathbb{R}^d$ are first projected onto the relation-specific hyperplane:

    .. math::

        \mathbf{h}_{r} = \mathbf{h} - \mathbf{r}_{w}^T \mathbf{h} \mathbf{r}_w

        \mathbf{t}_{r} = \mathbf{t} - \mathbf{r}_{w}^T \mathbf{t} \mathbf{r}_w

    Then, the projected representations are used to compute the score as in
    :class:`~pykeen.nn.modules.TransEInteraction`:

    .. math::

        -\|\textbf{h}_{r} + \textbf{r}_d - \textbf{t}_{r}\|_{p}^2

    ---
    citation:
        author: Wang
        year: 2014
        link: https://aaai.org/papers/8870-knowledge-graph-embedding-by-translating-on-hyperplanes/
    """

    relation_shape = ("d", "d")

    def forward(self, h: FloatTensor, r: tuple[FloatTensor, FloatTensor], t: FloatTensor) -> FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims, d)``
            The head representations.
        :param r: shape: ``(*batch_dims, d)``
            The relation representations.
        :param t: shape: ``(*batch_dims, d)``
            The tail representations.

        :return: shape: ``batch_dims``
            The scores.
        """
        d_r, w_r = r
        return negative_norm_of_sum(
            # h projection to hyperplane
            h,
            -einsum("...i, ...i, ...j -> ...j", h, w_r, w_r),
            # r
            d_r,
            # -t projection to hyperplane
            -t,
            einsum("...i, ...i, ...j -> ...j", t, w_r, w_r),
            p=self.p,
            power_norm=self.power_norm,
        )


@parse_docdata
class MuREInteraction(
    NormBasedInteraction[
        tuple[FloatTensor, FloatTensor], tuple[FloatTensor, FloatTensor], tuple[FloatTensor, FloatTensor]
    ],
):
    r"""The norm-based MuRE interaction function from [balazevic2019b]_.

    For $\mathbf{h}, \mathbf{r}, \mathbf{R}, \mathbf{t} \in \mathbb{R}^d$, and $b_h, b_t \in \mathbb{R}$, it is given
    by

    .. math ::
        -\|\mathbf{R} \odot \mathbf{h} + \mathbf{r} - \mathbf{t}\| + b_h + b_t

    where $\mathbf{h}, \mathbf{r}, \mathbf{t}$ are head entity, relation, and tail entity embedding vectors,
    $\mathbf{R}$ is a diagonal relation matrix, and $b_h, b_t$ are head and tail entity biases.

    .. note::
        This module implements a slightly more generic function, where the norm $\| \cdot \|_p$ can be chosen,
        as well as a variant which uses $\| \cdot \|_p^p$, cf. :class:`~pykeen.nn.modules.NormBasedInteraction`.

    ---
    citation:
        author: Balažević
        year: 2019
        link: https://arxiv.org/abs/1905.09791
        arxiv: 1905.09791
    """

    # there are separate biases for entities in head and tail position
    entity_shape = ("d", "", "")
    _head_indices = (0, 1)
    _tail_indices = (0, 2)

    relation_shape = ("d", "d")

    def forward(
        self, h: tuple[FloatTensor, FloatTensor], r: tuple[FloatTensor, FloatTensor], t: tuple[FloatTensor, FloatTensor]
    ) -> FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims, d)`` and ``(*batch_dims)``
            The head representations.
        :param r: shape: ``(*batch_dims, d)`` and ``(*batch_dims, d)``
            The relation representations.
        :param t: shape: ``(*batch_dims, d)`` and ``(*batch_dims)``
            The tail representations.

        :return: shape: ``batch_dims``
            The scores.
        """
        h_emb, h_bias = h
        t_emb, t_bias = t
        r_vec, r_mat = r
        return (
            negative_norm_of_sum(h_emb * r_mat, r_vec, -t_emb, p=self.p, power_norm=self.power_norm) + h_bias + t_bias
        )


@parse_docdata
class SimplEInteraction(
    FunctionalInteraction[
        tuple[FloatTensor, FloatTensor],
        tuple[FloatTensor, FloatTensor],
        tuple[FloatTensor, FloatTensor],
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
    ) -> MutableMapping[str, FloatTensor]:  # noqa: D102
        return dict(h=h[0], h_inv=h[1], r=r[0], r_inv=r[1], t=t[0], t_inv=t[1])


@parse_docdata
class PairREInteraction(NormBasedInteraction[FloatTensor, tuple[FloatTensor, FloatTensor], FloatTensor]):
    r"""The state-less norm-based PairRE interaction function.

    It is given by

    .. math ::
        -\|\mathbf{h} \odot \mathbf{r}_h - \mathbf{t} \odot \mathbf{r}_t \|

    where $\mathbf{h}, \mathbf{r}_h, \mathbf{r}_t, \mathbf{t} \in \mathbb{R}$ are representations for head entity,
    relation-specific head projection, relation-specific tail projection, and tail entity, respectively.

    .. note ::
        :class:`pykeen.models.PairRE` additionally enforces $\|\mathbf{h}\| = \|\mathbf{t}\| = 1$.

    ---
    citation:
        author: Chao
        year: 2020
        link: http://arxiv.org/abs/2011.03798
        arxiv: 2011.03798
        github: alipay/KnowledgeGraphEmbeddingsViaPairedRelationVectors_PairRE
    """

    relation_shape = ("d", "d")

    def forward(self, h: FloatTensor, r: tuple[FloatTensor, FloatTensor], t: FloatTensor) -> FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims, d)``
            The head representations.
        :param r: shape: ``(*batch_dims, d)`` and ``(*batch_dims, d)``
            The relation representations.
        :param t: shape: ``(*batch_dims, d)``
            The tail representations.

        :return: shape: ``batch_dims``
            The scores.
        """
        r_h, r_t = r
        return negative_norm_of_sum(
            h * r_h,
            -t * r_t,
            p=self.p,
            power_norm=self.power_norm,
        )


@parse_docdata
class QuatEInteraction(
    FunctionalInteraction[
        FloatTensor,
        FloatTensor,
        FloatTensor,
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
    ) -> FloatTensor:  # noqa: D102
        return self.log_scale.exp() * self.base(h=h, r=r, t=t) + self.bias


@parse_docdata
class CrossEInteraction(Interaction[FloatTensor, tuple[FloatTensor, FloatTensor], FloatTensor]):
    r"""The stateful interaction function of CrossE.

    The interaction function is given by

    .. math ::

        \textit{drop}(
            \textit{act}(
                \mathbf{c}_r \odot \mathbf{h} + \mathbf{c}_r \odot \mathbf{h} \odot \mathbf{r} + \mathbf{b})
            )
        )^T
        \mathbf{t}

    where $\mathbf{h}, \mathbf{c}_r, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$ is the head embedding, the relation
    interaction vector, the relation embedding, and the tail embedding, respectively.
    $\mathbf{b} \in \mathbb{R}^d$ is a global bias vector (which makes this interaction function stateful).
    $\textit{drop}$ denotes dropout, and $\textit{act}$ is the activation function.

    .. note ::
        The CrossE paper describes an additional sigmoid activation as part of the interaction function. Since using a
        log-likelihood loss can cause numerical problems (due to explicitly calling sigmoid before log), we do not use
        it in our implementation, but opt for the numerically stable variant. However, the model itself has an option
        ``predict_with_sigmoid``, which can be used to force the use of sigmoid during inference. This can also affect
        rank-based scoring, since limited numerical precision can lead to exactly equal scores for multiple choices.
        The definition of a rank is not clear in this case, and there are several competing ways to break ties.
        See :ref:`understanding-evaluation` for more information.

    ---
    citation:
        author: Zhang
        year: 2019
        link: https://arxiv.org/abs/1903.04750
        arxiv: 1903.04750
        github: https://github.com/wencolani/CrossE
    """

    relation_shape = ("d", "d")

    @update_docstring_with_resolver_keys(
        ResolverKey("combination_activation", "class_resolver.contrib.torch.activation_resolver")
    )
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
            An optional dropout applied after the combination and before the dot product similarity.
        """
        super().__init__()
        self.activation = activation_resolver.make(
            combination_activation,
            pos_kwargs=combination_activation_kwargs,
        )
        # TODO: expose initialization?
        self.bias = nn.Parameter(data=torch.zeros(embedding_dim))
        self.dropout = nn.Dropout(combination_dropout) if combination_dropout else None

    def forward(
        self,
        h: FloatTensor,
        r: tuple[FloatTensor, FloatTensor],
        t: FloatTensor,
    ) -> FloatTensor:
        r"""
        Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: (`*batch_dims`, dim)
            The head representations.
        :param r: shape: (`*batch_dims`, dim)
            The relation representations and relation-specific interaction vector.
        :param t: shape: (`*batch_dims`, dim)
            The tail representations.

        :return: shape: batch_dims
            The scores.
        """
        r_emb, c_r = r
        # head interaction
        h = c_r * h
        # relation interaction (notice that h has been updated)
        r_emb = h * r_emb
        # combination
        x = self.activation(self.bias.view(*make_ones_like(h.shape[:-1]), -1) + h + r_emb)
        if self.dropout is not None:
            x = self.dropout(x)
        # similarity
        return batched_dot(x, t)


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
    ) -> MutableMapping[str, FloatTensor]:  # noqa: D102
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
    def product_normalize(x: FloatTensor, dim: int = -1) -> FloatTensor:
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
        points: FloatTensor,
        box_lows: FloatTensor,
        box_highs: FloatTensor,
    ) -> FloatTensor:
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
        entity_pos: FloatTensor,
        other_entity_bump: FloatTensor,
        relation_box: tuple[FloatTensor, FloatTensor],
        tanh_map: bool,
        p: int,
        power_norm: bool,
    ) -> FloatTensor:
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
        base: FloatTensor,
        delta: FloatTensor,
        size: FloatTensor,
    ) -> tuple[FloatTensor, FloatTensor]:
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
        h_pos: FloatTensor,
        h_bump: FloatTensor,
        # relation box: head
        rh_base: FloatTensor,
        rh_delta: FloatTensor,
        rh_size: FloatTensor,
        # relation box: tail
        rt_base: FloatTensor,
        rt_delta: FloatTensor,
        rt_size: FloatTensor,
        # tail
        t_pos: FloatTensor,
        t_bump: FloatTensor,
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
    The Canonical Tensor Decomposition interaction as described [lacroix2018]_ (originally from [hitchcock1927]_).

    .. note ::
        For $k=1$, this interaction is the same as :class:`~pykeen.nn.modules.DistMultInteraction`.
        However, in contrast to :class:`~pykeen.models.DistMult`, entities should have different representations for the
        head and the tail role.

    ---
    name: Canonical Tensor Decomposition
    citation:
        author: Lacroix
        year: 2018
        arxiv: 1806.07297
        link: https://arxiv.org/abs/1806.07297
        github: facebookresearch/kbc
    """

    entity_shape = ("kd",)
    relation_shape = ("kd",)
    _head_indices = (0,)
    _tail_indices = (1,)

    @staticmethod
    def func(h: FloatTensor, r: FloatTensor, t: FloatTensor) -> FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims`, rank, dim)``
            The head representations.
        :param r: shape: ``(*batch_dims`, rank, dim)``
            The relation representations.
        :param t: shape: ``(*batch_dims`, rank, dim)``
            The tail representations.

        :return: shape: batch_dims
            The scores.
        """
        return (h * r * t).sum(dim=(-2, -1))


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
    ) -> MutableMapping[str, FloatTensor]:  # noqa: D102
        return dict(h=h[0], r=r, t=t[1])

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:
        return dict(core_tensor=self.core_tensor)


@parse_docdata
class TransformerInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
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
class TripleREInteraction(NormBasedInteraction[FloatTensor, tuple[FloatTensor, FloatTensor, FloatTensor], FloatTensor]):
    r"""The TripleRE interaction function from [yu2021]_.

    It is given by

    .. math ::
         \mathbf{h} \odot (\mathbf{r}_h + u) - \mathbf{t} \odot (\mathbf{r}_t + u) + \mathbf{r}

    with head entity, relation and tail entity representations $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$,
    relation specific head and tail multipliers $\mathbf{r}_h, \mathbf{r}_t \in \mathbb{R}^d$,
    and a scalar relation factor offset $u \in \mathbb{R}$.

    .. note ::
        This interaction is equivalent to :class:`~pykeen.nn.modules.LineaREInteraction` except the $u$ term.
        The $u$ is only non-zero for the version 2 from the paper.

    .. note ::

        For equivalence to the paper version, `h` and `t` should be normalized to unit
        Euclidean length, and `p` and `power_norm` be kept at their default values.

    .. seealso:: https://github.com/LongYu-360/TripleRE-Add-NodePiece

    ---
    name: TripleRE
    citation:
        author: Yu
        year: 2021
        link: https://vixra.org/abs/2112.0095
    """

    # r_head, r_mid, r_tail
    relation_shape = ("d", "d", "d")

    def __init__(self, u: float | None = 1.0, p: int = 1, power_norm: bool = False):
        """
        Initialize the module.

        .. seealso::
            The parameter ``p`` and ``power_norm`` are directly passed to
            :class:`~pykeen.nn.modules.NormBasedInteraction`.

        :param u:
            Rhe relation factor offset. Can be set to `None` (or 0) to disable it.
        :param p:
            The norm used with :func:`torch.linalg.vector_norm`.
        :param power_norm:
            Whether to use the $p$-th power of the $L_p$ norm. It has the advantage of being differentiable around 0,
            and numerically more stable.
        """
        super().__init__(p=p, power_norm=power_norm)
        self.u = u

    def forward(self, h: FloatTensor, r: tuple[FloatTensor, FloatTensor, FloatTensor], t: FloatTensor) -> FloatTensor:
        """Evaluate the interaction function.

        .. seealso::
            :meth:`Interaction.forward <pykeen.nn.modules.Interaction.forward>` for a detailed description about
            the generic batched form of the interaction function.

        :param h: shape: ``(*batch_dims, d)``
            The head representations.
        :param r: shape: ``(*batch_dims, d)``
            The relation representations.
        :param t: shape: ``(*batch_dims, d)``
            The tail representations.

        :return: shape: ``batch_dims``
            The scores.
        """
        u = self.u
        r_head, r_mid, r_tail = r
        # note: normalization should be done from the representations
        # cf. https://github.com/LongYu-360/TripleRE-Add-NodePiece/blob/994216dcb1d718318384368dd0135477f852c6a4/TripleRE%2BNodepiece/ogb_wikikg2/model.py#L317-L328  # noqa: E501
        # version 2
        if u is not None:
            # r_head = r_head + u * torch.ones_like(r_head)
            # r_tail = r_tail + u * torch.ones_like(r_tail)
            r_head = r_head + u
            r_tail = r_tail + u

        return negative_norm_of_sum(h * r_head, -t * r_tail, r_mid, p=self.p, power_norm=self.power_norm)


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
    ) -> MutableMapping[str, FloatTensor]:  # noqa: D102
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
