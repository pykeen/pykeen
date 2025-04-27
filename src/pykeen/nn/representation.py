"""Representation modules."""

from __future__ import annotations

import dataclasses
import itertools
import logging
import math
import string
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar, Literal, cast

import more_itertools
import numpy
import numpy as np
import torch
import torch.nn
from class_resolver import (
    FunctionResolver,
    HintOrType,
    OneOrManyHintOrType,
    OneOrManyOptionalKwargs,
    OptionalKwargs,
    ResolverKey,
    update_docstring_with_resolver_keys,
)
from class_resolver.contrib.torch import activation_resolver
from docdata import parse_docdata
from torch import nn
from torch.nn import functional
from typing_extensions import Self

from .combination import Combination, combination_resolver
from .compositions import CompositionModule, composition_resolver
from .init import PretrainedInitializer, initializer_resolver
from .text.cache import PyOBOTextCache, TextCache, WikidataTextCache
from .text.encoder import TextEncoder, text_encoder_resolver
from .utils import BaseShapeError, ShapeError
from .weighting import EdgeWeighting, SymmetricEdgeWeighting, edge_weight_resolver
from ..datasets import Dataset
from ..regularizers import Regularizer, regularizer_resolver
from ..triples import CoreTriplesFactory, TriplesFactory
from ..triples.triples_factory import Labeling
from ..typing import (
    Constrainer,
    FloatTensor,
    Hint,
    HintType,
    Initializer,
    LongTensor,
    Normalizer,
    OneOrSequence,
)
from ..utils import (
    Bias,
    ExtraReprMixin,
    broadcast_upgrade_to_sequences,
    clamp_norm,
    complex_normalize,
    einsum,
    get_edge_index,
    get_preferred_device,
    merge_kwargs,
    upgrade_to_sequence,
)

__all__ = [
    "Representation",
    "Embedding",
    "LowRankRepresentation",
    "CompGCNLayer",
    "CombinedCompGCNRepresentations",
    "PartitionRepresentation",
    "BackfillRepresentation",
    "SingleCompGCNRepresentation",
    "SubsetRepresentation",
    "CombinedRepresentation",
    "TensorTrainRepresentation",
    "MultiBackfillRepresentation",
    "TransformedRepresentation",
    "TextRepresentation",
    "CachedTextRepresentation",
    "WikidataTextRepresentation",
    "BiomedicalCURIERepresentation",
    "EmbeddingBagRepresentation",
    # Utils
    "constrainer_resolver",
    "normalizer_resolver",
]

logger = logging.getLogger(__name__)


#: A resolver for constrainers.
#:
#: - :func:`torch.nn.functional.normalize`
#: - :func:`complex_normalize`
#: - :func:`torch.clamp`
#: - :func:`clamp_norm`
constrainer_resolver: FunctionResolver[[FloatTensor], FloatTensor] = FunctionResolver(
    [functional.normalize, complex_normalize, torch.clamp, clamp_norm],
    location="pykeen.nn.representation.constrainer_resolver",
)

#: A resolver for normalizers.
#:
#: - :func:`torch.nn.functional.normalize`
#: - :func:`torch.nn.functional.softmax`
normalizer_resolver: FunctionResolver[[FloatTensor], FloatTensor] = FunctionResolver(
    [functional.normalize, functional.softmax],
    location="pykeen.nn.representation.normalizer_resolver",
)


class MaxIDMismatchError(ValueError):
    """Raised when the maximum ID of a representation is inconsistent."""


class Representation(nn.Module, ExtraReprMixin, ABC):
    """
    A base class for obtaining representations for entities/relations.

    A representation module maps integer IDs to representations, which are tensors of floats.

    ``max_id`` defines the upper bound of indices we are allowed to request (exclusively). For simple embeddings this is
    equivalent to num_embeddings, but more a more appropriate word for general non-embedding representations, where the
    representations could come from somewhere else, e.g. a GNN encoder.

    ``shape`` describes the shape of a single representation. In case of a vector embedding, this is just a single
    dimension. For others, e.g. :class:`~pykeen.models.RESCAL`, we have 2-d representations, and in general it can be
    any fixed shape.

    We can look at all representations as a tensor of shape ``(max_id, *shape)``, and this is exactly the result of
    passing ``indices=None`` to the forward method.

    We can also pass multi-dimensional ``indices`` to the forward method, in which case the indices' shape becomes the
    prefix of the result shape: ``(*indices.shape, *self.shape)``.
    """

    #: the maximum ID (exclusively)
    max_id: int

    #: the shape of an individual representation
    shape: tuple[int, ...]

    #: a normalizer for individual representations
    normalizer: Normalizer | None

    #: a regularizer for individual representations
    regularizer: Regularizer | None

    #: dropout
    dropout: nn.Dropout | None

    @update_docstring_with_resolver_keys(
        ResolverKey("normalizer", normalizer_resolver),
        ResolverKey("regularizer", regularizer_resolver),
    )
    def __init__(
        self,
        max_id: int,
        shape: OneOrSequence[int] = 64,
        normalizer: HintOrType[Normalizer] = None,
        normalizer_kwargs: OptionalKwargs = None,
        regularizer: HintOrType[Regularizer] = None,
        regularizer_kwargs: OptionalKwargs = None,
        dropout: float | None = None,
        unique: bool | None = None,
    ):
        """Initialize the representation module.

        :param max_id:
            The maximum ID (exclusively). Valid Ids reach from ``0`` to ``max_id-1``.
        :param shape:
            The shape of an individual representation.

        :param normalizer:
            A normalization function, which is applied to the selected representations in every forward pass.
        :param normalizer_kwargs:
            Additional keyword arguments passed to the normalizer.

        :param regularizer:
            An output regularizer, which is applied to the selected representations in forward pass.
        :param regularizer_kwargs:
            Additional keyword arguments passed to the regularizer.

        :param dropout:
            The optional dropout probability.
        :param unique:
            Whether to optimize for calculating representations for same indices only once. This is only useful if the
            calculation of representations is significantly more expensive than an index-based lookup and
            duplicate indices are expected, e.g., when using negative sampling and large batch sizes.

            .. warning ::
                When using this optimization you may encounter unexpected results for stochastic operations, e.g.,
                :class:`torch.nn.Dropout`.
        """
        super().__init__()
        self.max_id = max_id
        self.shape = tuple(upgrade_to_sequence(shape))
        self.normalizer = normalizer_resolver.make_safe(normalizer, normalizer_kwargs)
        self.regularizer = regularizer_resolver.make_safe(regularizer, regularizer_kwargs)
        self.dropout = None if dropout is None else nn.Dropout(dropout)
        if unique is None:
            # heuristic
            unique = not isinstance(self, Embedding) and not dropout
            logger.info(f"Inferred {unique=} for {self}")
        self.unique = unique

    @abstractmethod
    def _plain_forward(
        self,
        indices: LongTensor | None = None,
    ) -> FloatTensor:
        """Get representations for indices, without applying normalization, regularization or output dropout."""
        raise NotImplementedError

    def forward(
        self,
        indices: LongTensor | None = None,
    ) -> FloatTensor:
        """Get representations for indices.

        :param indices: shape: ``s``
            The indices, or ``None``. If ``None``, this is interpreted as ``torch.arange(self.max_id)``
            (although implemented more efficiently).

        :return: shape: ``(*s, *self.shape)``
            The representations.
        """
        inverse = None
        if indices is not None and self.unique:
            indices, inverse = indices.unique(return_inverse=True)
        x = self._plain_forward(indices=indices)
        # normalize *before* repeating
        if self.normalizer is not None:
            x = self.normalizer(x)
        # repeat if necessary
        if inverse is not None:
            x = x[inverse]
        # regularize *after* repeating
        if self.regularizer is not None:
            self.regularizer.update(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def reset_parameters(self) -> None:
        """Reset the module's parameters."""

    def post_parameter_update(self):
        """Apply constraints which should not be included in gradients."""

    def iter_extra_repr(self) -> Iterable[str]:
        """Iterate over components for :meth:`extra_repr`."""
        yield from super().iter_extra_repr()
        yield f"max_id={self.max_id}"
        yield f"shape={self.shape}"
        yield f"unique={self.unique}"
        if self.normalizer is not None:
            yield f"normalizer={self.normalizer}"
        # dropout & regularizer will appear automatically, since it is a nn.Module

    @property
    def device(self) -> torch.device:
        """Return the device."""
        return get_preferred_device(module=self, allow_ambiguity=True)


@parse_docdata
class SubsetRepresentation(Representation):
    """A representation module, which only exposes a subset of representations of its base.

    .. note ::
        At runtime, no index verification is made. Thus the only effect is based on the adjusted ``max_id``.

    ---
    name: Subset Representation
    """

    @update_docstring_with_resolver_keys(ResolverKey(name="base", resolver="pykeen.nn.representation_resolver"))
    def __init__(
        self,
        max_id: int,
        shape: OneOrSequence[int] | None = None,
        base: HintOrType[Representation] = None,
        base_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        """
        Initialize the representations.

        :param max_id:
            The number of representations.
        :param shape:
            The shape of an individual representation.

        :param base:
            The base representation. Has to have a sufficient number of representations, i.e., at least ``max_id``.
        :param base_kwargs:
            Additional keyword arguments for the base representation.

        :param kwargs:
            Additional keyword-based parameters passed to :class:`~pykeen.nn.representation.Representation`.

        :raises ValueError: if ``max_id`` is larger than the base representation's ``max_id``
        """
        # has to be imported here to avoid cyclic import
        from . import representation_resolver

        base = representation_resolver.make(base, pos_kwargs=base_kwargs)
        if max_id > base.max_id:
            raise ValueError(
                f"Base representations comprise only {base.max_id} representations, "
                f"but at least {max_id} are required.",
            )
        super().__init__(max_id=max_id, shape=ShapeError.verify(shape=base.shape, reference=shape), **kwargs)
        self.base = base

    # docstr-coverage: inherited
    def _plain_forward(
        self,
        indices: LongTensor | None = None,
    ) -> FloatTensor:  # noqa: D102
        if indices is None:
            indices = torch.arange(self.max_id, device=self.device)
        return self.base._plain_forward(indices=indices)


@parse_docdata
class Embedding(Representation):
    """Trainable embeddings.

    This class provides the same interface as :class:`torch.nn.Embedding` and
    can be used throughout PyKEEN as a more complete drop-in replacement.

    It extends it by adding additional options to normalize, constrain, or apply drop-out.

    .. note ::
        A discussion about the differences between normalizers and constrainers can be found
        in :ref:`normalizer_constrainer_regularizer`.

    The optional *dropout* can also be used as a regularization technique.
    It also allows uncertainty estimates to be obtained using techniques such as
    `Monte-Carlo dropout <https://arxiv.org/abs/1506.02142>`_.
    The following simple example shows how to obtain different scores for a single triple from an (untrained) model.
    These scores can be viewed as samples from a distribution over the scores.

    .. literalinclude:: ../examples/nn/representation/monte_carlo_embedding.py

    ---
    name: Embedding
    """

    normalizer: Normalizer | None
    constrainer: Constrainer | None
    regularizer: Regularizer | None
    dropout: nn.Dropout | None

    @update_docstring_with_resolver_keys(
        ResolverKey("initializer", initializer_resolver),
        ResolverKey("constrainer", constrainer_resolver),
    )
    def __init__(
        self,
        max_id: int | None = None,
        num_embeddings: int | None = None,
        embedding_dim: int | None = None,
        shape: None | int | Sequence[int] = None,
        initializer: Hint[Initializer] = None,
        initializer_kwargs: Mapping[str, Any] | None = None,
        constrainer: Hint[Constrainer] = None,
        constrainer_kwargs: Mapping[str, Any] | None = None,
        trainable: bool = True,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        """Instantiate an embedding with extended functionality.

        :param max_id: >0
            The number of embeddings, cf. :class:`~pykeen.nn.representation.Representation`.
        :param num_embeddings: >0
            The number of embeddings.

            .. note::
                This argument is kept for backwards compatibility. New code should use ``max_id`` instead.

        :param embedding_dim: >0
            The embedding dimensionality.
        :param shape:
            The shape of an individual representation, cf. :class:`~pykeen.nn.representation.Representation`.

            .. note::
                You can pass exactly only one of ``embedding_dim`` and ``shape``.
                ``shape`` is generally preferred because it is the more generic parameter also used in
                :class:`~pykeen.nn.representation.Representation`,
                but the term ``embedding_dim`` is so ubiquitous that it is available as well.

        :param initializer:
            An optional initializer, which takes an uninitialized ``(max_id, *shape)`` tensor as input,
            and returns an initialized tensor of same shape and dtype (which may be the same, i.e. the
            initialization may be in-place). Can be passed as a function, or as string, cf. resolver note.
        :param initializer_kwargs:
            Additional keyword arguments passed to the initializer
        :param constrainer:
            A function which is applied to the weights after each parameter update, without tracking gradients.
            It may be used to enforce model constraints outside gradient-based training. The function does not need
            to be in-place, but the weight tensor is modified in-place. Can be passed as a function, or as a string,
            cf. resolver note.
        :param constrainer_kwargs:
            Additional keyword arguments passed to the constrainer
        :param trainable:
            Should the wrapped embeddings be marked to require gradient.
        :param dtype:
            The datatype (otherwise uses :func:`torch.get_default_dtype` to look up).
        :param kwargs:
            Additional keyword-based parameters passed to :class:`~pykeen.nn.representation.Representation`
        """
        # normalize num_embeddings vs. max_id
        max_id = process_max_id(max_id, num_embeddings)

        # normalize embedding_dim vs. shape
        _embedding_dim, shape = process_shape(embedding_dim, shape)

        if dtype is None:
            dtype = torch.get_default_dtype()

        # work-around until full complex support (torch==1.10 still does not work)
        # TODO: verify that this is our understanding of complex!
        self.is_complex = dtype.is_complex
        _shape = shape
        if self.is_complex:
            _shape = tuple(shape[:-1]) + (shape[-1], 2)
            _embedding_dim = _embedding_dim * 2
            # note: this seems to work, as finfo returns the datatype of the underlying floating
            # point dtype, rather than the combined complex one
            dtype = getattr(torch, torch.finfo(dtype).dtype)
        self._shape = _shape

        super().__init__(max_id=max_id, shape=shape, **kwargs)

        # use make for initializer since there's a default, and make_safe
        # for the others to pass through None values
        self.initializer = initializer_resolver.make(initializer, initializer_kwargs)
        self.constrainer = constrainer_resolver.make_safe(constrainer, constrainer_kwargs)
        self._embeddings = torch.nn.Embedding(num_embeddings=max_id, embedding_dim=_embedding_dim, dtype=dtype)
        self._embeddings.requires_grad_(trainable)

    @classmethod
    def from_pretrained(
        cls, tensor: FloatTensor | PretrainedInitializer, *, trainable: bool = False, **kwargs: Any
    ) -> Self:
        """Construct an embedding from a pre-trained tensor.

        :param tensor:
            the tensor of pretrained embeddings, or pretrained initializer that wraps a tensor
        :param trainable:
            should the embedding be trainable? defaults to false, since this
            constructor is typically used for making a static embedding.
        :param kwargs: Remaining keyword arguments to pass to the :class:`pykeen.nn.Embedding` constructor
        :returns: An embedding representation
        """
        if not isinstance(tensor, PretrainedInitializer):
            tensor = PretrainedInitializer(tensor)
        max_id, *shape = tensor.tensor.shape
        return cls(max_id=max_id, shape=shape, initializer=tensor, trainable=trainable, **kwargs)

    # docstr-coverage: inherited
    def reset_parameters(self) -> None:  # noqa: D102
        # initialize weights in-place
        self._embeddings.weight.data = self.initializer(
            self._embeddings.weight.data.view(self.max_id, *self._shape),
        ).view(*self._embeddings.weight.data.shape)

    # docstr-coverage: inherited
    def post_parameter_update(self):  # noqa: D102
        # apply constraints in-place
        if self.constrainer is not None:
            x = self._plain_forward()
            x = self.constrainer(x)
            # fixme: work-around until nn.Embedding supports complex
            if self.is_complex:
                x = torch.view_as_real(x)
            self._embeddings.weight.data = x.view(*self._embeddings.weight.data.shape)

    # docstr-coverage: inherited
    def _plain_forward(
        self,
        indices: LongTensor | None = None,
    ) -> FloatTensor:  # noqa: D102
        if indices is None:
            prefix_shape = (self.max_id,)
            x = self._embeddings.weight
        else:
            prefix_shape = indices.shape
            x = self._embeddings(indices.to(self.device))
        x = x.view(*prefix_shape, *self._shape)
        # fixme: work-around until nn.Embedding supports complex
        if self.is_complex:
            x = torch.view_as_complex(x)
        # verify that contiguity is preserved
        assert x.is_contiguous()
        return x


@parse_docdata
class LowRankRepresentation(Representation):
    r"""
    Low-rank embedding factorization.

    This representation reduces the number of trainable parameters by not learning independent weights for each index,
    but rather having shared bases for all indices and learning only the weights of the linear combination.

    .. math ::
        E[i] = \sum_k B[i, k] \cdot W[k]

    This representation implements the generalized form, where both, $B$ and $W$ are arbitrary representations
    themselves.

    Example usage:

    .. literalinclude:: ../examples/nn/representation/low_rank_mixture.py

    ---
    name: Low Rank Embedding
    """

    @update_docstring_with_resolver_keys(
        ResolverKey("base", resolver="pykeen.nn.representation_resolver"),
        ResolverKey("weight", resolver="pykeen.nn.representation_resolver"),
    )
    def __init__(
        self,
        *,
        max_id: int | None = None,
        shape: Sequence[int] | int | None = None,
        num_bases: int | None = 3,
        # base representation
        base: HintOrType[Representation] = None,
        base_kwargs: OptionalKwargs = None,
        # weight representation
        weight: HintOrType[Representation] = None,
        weight_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        """
        Initialize the representations.

        :param max_id:
            The maximum ID (exclusively). Valid Ids reach from ``0`` to ``max_id-1``.
            If None, a pre-instantiated weight representation needs to be provided.
        :param shape:
            The shape of an individual representation.
            If None, a pre-instantiated base representation has to be provided.
        :param num_bases:
            The number of bases. More bases increase expressivity, but also increase the number of trainable parameters.
            If None, a pre-instantiated base representation has to be provided.

        :param weight:
            The weight representation, or a hint thereof.
        :param weight_kwargs:
            Additional keyword based arguments used to instantiate the weight representation.

        :param base:
            The base representation, or a hint thereof.
        :param base_kwargs:
            Additional keyword based arguments used to instantiate the weight representation.

        :param kwargs:
            Additional keyword based arguments passed to :class:`~pykeen.nn.representation.Representation`.

        :raises MaxIDMismatchError:
            if the ``max_id`` was given explicitly and does not match the ``max_id`` of the weight
            representation
        """
        # has to be imported here to avoid cyclic import
        from . import representation_resolver

        base = representation_resolver.make(base, pos_kwargs=base_kwargs, max_id=num_bases, shape=shape)
        weight = representation_resolver.make(
            weight, pos_kwargs=merge_kwargs(weight_kwargs, max_id=max_id), shape=num_bases
        )

        # Verification
        if max_id is None:
            max_id = weight.max_id
        elif max_id != weight.max_id:
            raise MaxIDMismatchError(f"Explicitly provided {max_id=:_} does not match {weight.max_id=:_}")
        if num_bases is not None and base.max_id != num_bases:
            logger.warning(
                f"The explicitly provided {num_bases=:_} does not match {base.max_id=:_} and has been ignored."
            )
        super().__init__(max_id=max_id, shape=ShapeError.verify(base.shape, shape), **kwargs)

        # assign *after* super init
        self.base = base
        self.weight = weight

    @classmethod
    def approximate(cls, other: Representation, num_bases: int = 3, **kwargs) -> Self:
        """
        Construct a low-rank approximation of another representation.

        .. note ::

            While this method tries to find a good approximation of the base representation, you may lose any (useful)
            inductive biases you had with the original one, e.g., from shared tokens in
            :class:`~pykeen.nn.node_piece.NodePieceRepresentation`.

        :param other:
            The representation to approximate.
        :param num_bases:
            The number of bases. More bases increase expressivity, but also increase the number of trainable parameters.
        :param kwargs:
            Additional keyword-based parameters passed to :meth:`__init__`. Must not contain
            ``max_id`` nor ``shape``, which are determined by ``other``.

        :return:
            A low-rank approximation obtained via (truncated) SVD, cf. :func:`torch.svd_lowrank`.
        """
        # get base representations, shape: (n, *ds)
        x = other(indices=None)
        # calculate SVD, U.shape: (n, k), s.shape: (k,), u.shape: (k, prod(ds))
        u, s, vh = torch.svd_lowrank(x.view(x.shape[0], -1), q=num_bases)
        # setup weight & base representation
        weight = Embedding(
            max_id=num_bases,
            shape=num_bases,
            initializer=PretrainedInitializer(tensor=torch.einsum("nk, k -> nk", u, s)),
        )
        base = Embedding(max_id=num_bases, shape=other.shape, initializer=PretrainedInitializer(tensor=vh))
        # create low-rank approximation object
        return cls(base=base, weight=weight, **kwargs)

    @property
    def num_bases(self) -> int:
        """Return the number of bases."""
        return self.base.max_id

    # docstr-coverage: inherited
    def _plain_forward(
        self,
        indices: LongTensor | None = None,
    ) -> FloatTensor:  # noqa: D102
        # get all base representations, shape: (num_bases, *shape)
        bases = self.base(indices=None)
        # get base weights, shape: (*batch_dims, num_bases)
        weight = self.weight(indices=indices)
        # weighted linear combination of bases, shape: (*batch_dims, *shape)
        return torch.tensordot(weight, bases, dims=([-1], [0]))


def process_shape(
    dim: int | None,
    shape: None | int | Sequence[int],
) -> tuple[int, Sequence[int]]:
    """Make a shape pack."""
    if shape is None and dim is None:
        raise ValueError("Missing both, shape and embedding_dim")
    elif shape is not None and dim is not None:
        raise ValueError("Provided both, shape and embedding_dim")
    elif shape is None and dim is not None:
        shape = (dim,)
    elif isinstance(shape, int) and dim is None:
        dim = shape
        shape = (shape,)
    elif isinstance(shape, Sequence) and dim is None:
        shape = tuple(shape)
        dim = int(np.prod(shape))
    else:
        raise TypeError(f"Invalid type for shape: ({type(shape)}) {shape}")
    return dim, shape


def process_max_id(max_id: int | None, num_embeddings: int | None) -> int:
    """Normalize max_id."""
    if max_id is None:
        if num_embeddings is None:
            raise ValueError("Must provide max_id")
        warnings.warn("prefer using 'max_id' over 'num_embeddings'", DeprecationWarning, stacklevel=2)
        max_id = num_embeddings
    elif num_embeddings is not None and num_embeddings != max_id:
        raise ValueError("Cannot provide both, 'max_id' over 'num_embeddings'")
    return max_id


class CompGCNLayer(nn.Module):
    """A single CompGCN layer."""

    @update_docstring_with_resolver_keys(
        ResolverKey("composition", composition_resolver),
        ResolverKey("edge_weighting", edge_weight_resolver),
        ResolverKey("activation", activation_resolver),
    )
    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        *,
        dropout: float = 0.0,
        use_bias: bool = True,
        use_relation_bias: bool = False,
        composition: Hint[CompositionModule] = None,
        composition_kwargs: OptionalKwargs = None,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        activation: HintOrType[nn.Module] = nn.Identity,
        activation_kwargs: Mapping[str, Any] | None = None,
        edge_weighting: HintType[EdgeWeighting] = SymmetricEdgeWeighting,
        edge_weighting_kwargs: OptionalKwargs = None,
    ):
        """
        Initialize the module.

        :param input_dim:
            The input dimension.
        :param output_dim:
            The output dimension. If ``None``, equals the input dimension.
        :param dropout:
            The dropout to use for forward and backward edges.
        :param use_bias:  # TODO: do we really need this? it comes before a mandatory batch norm layer
            Whether to use bias.
        :param use_relation_bias:
            Whether to use a bias for the relation transformation.
        :param composition:
            The composition function.
        :param composition_kwargs:
            Additional keyword based arguments passed to the composition.
        :param attention_heads:
            Number of attention heads when using the attention weighting
        :param attention_dropout:
            Dropout for the attention message weighting
        :param activation:
            The activation to use.
        :param activation_kwargs:
            Additional key-word based arguments passed to the activation.
        :param edge_weighting:
            A pre-instantiated :class:`EdgeWeighting`, a class, or name to look
            up with :class:`class_resolver`.
        :param edge_weighting_kwargs:
            Additional keyword based arguments passed to the edge weighting.
            Note that the following keyword arguments for :class:`CompGCNLayer` are automatically
            shuttled in here:

            - ``output_dim`` (or ``input_dim``, if output dimension is not given) is passed to ``message_dim``
            - ``attention_dropout`` is passed to ``dropout``
            - ``attention_heads`` is passed to ``num_heads``
        """
        super().__init__()

        # normalize output dimension
        output_dim = output_dim or input_dim

        # entity-relation composition
        self.composition = composition_resolver.make(composition, composition_kwargs)

        # edge weighting
        self.edge_weighting: EdgeWeighting = edge_weight_resolver.make(
            edge_weighting,
            edge_weighting_kwargs,
            message_dim=output_dim,
            dropout=attention_dropout,
            num_heads=attention_heads,
        )

        # message passing weights
        self.w_loop = nn.Parameter(data=torch.empty(input_dim, output_dim))
        self.w_fwd = nn.Parameter(data=torch.empty(input_dim, output_dim))
        self.w_bwd = nn.Parameter(data=torch.empty(input_dim, output_dim))

        # linear relation transformation
        self.w_rel = nn.Linear(in_features=input_dim, out_features=output_dim, bias=use_relation_bias)

        # layer-specific self-loop relation representation
        self.self_loop = nn.Parameter(data=torch.empty(1, input_dim))

        # other components
        self.drop = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(output_dim)
        self.bias = Bias(output_dim) if use_bias else None
        self.activation = activation_resolver.make(query=activation, pos_kwargs=activation_kwargs)

        # initialize
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the model's parameters."""
        for w in (
            self.w_loop,
            self.w_fwd,
            self.w_bwd,
            self.self_loop,
        ):
            nn.init.xavier_uniform_(w)
        self.bias.reset_parameters()
        self.w_rel.reset_parameters()

    def message(
        self,
        x_e: FloatTensor,
        x_r: FloatTensor,
        edge_index: LongTensor,
        edge_type: LongTensor,
        weight: nn.Parameter,
    ) -> FloatTensor:
        """
        Perform message passing.

        :param x_e: shape: ``(num_entities, input_dim)``
            The entity representations.
        :param x_r: shape: ``(2 * num_relations, input_dim)``
            The relation representations (including inverse relations).
        :param edge_index: shape: ``(2, num_edges)``
            The edge index, pairs of source and target entity for each triple.
        :param edge_type: shape ``(num_edges,)``
            The edge type, i.e., relation ID, for each triple.
        :param weight:
            The transformation weight.

        :return:
            The updated entity representations.
        """
        # split
        source, target = edge_index

        # compose
        m = self.composition(x_e[source], x_r[edge_type])

        # transform
        m = m @ weight

        # normalization
        m = self.edge_weighting(source=source, target=target, message=m, x_e=x_e)

        # aggregate by sum
        x_e = x_e.new_zeros(x_e.shape[0], m.shape[1]).index_add(dim=0, index=target, source=m)

        # dropout
        x_e = self.drop(x_e)

        return x_e

    def forward(
        self,
        x_e: FloatTensor,
        x_r: FloatTensor,
        edge_index: LongTensor,
        edge_type: LongTensor,
    ) -> tuple[FloatTensor, FloatTensor]:
        r"""
        Update entity and relation representations.

        .. math ::
            X_E'[e] = \frac{1}{3} \left(
                X_E W_s
                + \left( \sum_{h,r,e \in T} \alpha(h, e) \phi(X_E[h], X_R[r]) W_f \right)
                + \left( \sum_{e,r,t \in T} \alpha(e, t) \phi(X_E[t], X_R[r^{-1}]) W_b \right)
            \right)

        :param x_e: shape: ``(num_entities, input_dim)``
            The entity representations.
        :param x_r: shape: ``(2 * num_relations, input_dim)``
            The relation representations (including inverse relations).
        :param edge_index: shape: ``(2, num_edges)``
            The edge index, pairs of source and target entity for each triple.
        :param edge_type: shape ``(num_edges,)``
            The edge type, i.e., relation ID, for each triple.

        :return: shape: ``(num_entities, output_dim)`` / ``(2 * num_relations, output_dim)``
            The updated entity and relation representations.
        """
        # prepare for inverse relations
        edge_type = 2 * edge_type
        # update entity representations: mean over self-loops / forward edges / backward edges
        x_e = (
            self.composition(x_e, self.self_loop) @ self.w_loop
            + self.message(x_e=x_e, x_r=x_r, edge_index=edge_index, edge_type=edge_type, weight=self.w_fwd)
            + self.message(x_e=x_e, x_r=x_r, edge_index=edge_index.flip(0), edge_type=edge_type + 1, weight=self.w_bwd)
        ) / 3

        if self.bias:
            x_e = self.bias(x_e)
        x_e = self.bn(x_e)
        x_e = self.activation(x_e)

        # Relation transformation
        x_r = self.w_rel(x_r)
        return x_e, x_r


def build_representation(
    max_id: int,
    representation: HintOrType[Representation],
    representation_kwargs: OptionalKwargs,
) -> Representation:
    """Build representations and check maximum ID."""
    # has to be imported here to avoid cyclic imports
    from . import representation_resolver

    representation = representation_resolver.make(
        representation, pos_kwargs=merge_kwargs(representation_kwargs, max_id=max_id)
    )
    if representation.max_id != max_id:
        raise MaxIDMismatchError(
            f"Representations should provide {max_id} representations, but have {representation.max_id}",
        )
    return representation


@parse_docdata
class CombinedCompGCNRepresentations(nn.Module):
    """A sequence of CompGCN layers.

    .. seealso::
        :class:`pykeen.nn.representation.CompGCNLayer`

    ---
    name: CompGCN (combine)
    citation:
        author: Vashishth
        year: 2020
        link: https://arxiv.org/pdf/1911.03082
        github: malllabiisc/CompGCN
    """

    # TODO: extract adapter for cached representations; cf. RGCN
    # Buffered enriched entity and relation representations
    enriched_representations: tuple[FloatTensor, FloatTensor] | None

    @update_docstring_with_resolver_keys(
        ResolverKey("entity_representations", resolver="pykeen.nn.representation_resolver"),
        ResolverKey("relation_representations", resolver="pykeen.nn.representation_resolver"),
    )
    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        entity_representations: HintOrType[Representation] = None,
        entity_representations_kwargs: OptionalKwargs = None,
        relation_representations: HintOrType[Representation] = None,
        relation_representations_kwargs: OptionalKwargs = None,
        num_layers: int | None = 1,
        dims: None | int | Sequence[int] = None,
        layer_kwargs: Mapping[str, Any] | None = None,
    ):
        """
        Initialize the combined entity and relation representation module.

        :param triples_factory:
            The triples factory containing the training triples.

        :param entity_representations:
            The base entity representations
        :param entity_representations_kwargs:
            Additional keyword parameters for the base entity representations.

        :param relation_representations:
            The base relation representations.
        :param relation_representations_kwargs:
            Additional keyword parameters for the base relation representations.

        :param num_layers:
            The number of message passing layers to use. If None, will be inferred by len(dims), i.e., requires dims to
            be a sequence / list.
        :param dims:
            The hidden dimensions to use. If None, defaults to the embedding dimension of the base representations.
            If an integer, is the same for all layers. The last dimension is equal to the output dimension.
        :param layer_kwargs:
            Additional key-word based parameters passed to the individual layers;
            cf. :class:`~pykeen.nn.representation.CompGCNLayer`.
        :raises ValueError: For several invalid combinations of arguments:
            1. If the dimensions were given as an integer but no number of layers were given
            2. If the dimensions were given as a ist but it does not match the number of layers that were given
        """
        super().__init__()
        # TODO: Check
        assert triples_factory.create_inverse_triples
        self.entity_representations = build_representation(
            max_id=triples_factory.num_entities,
            representation=entity_representations,
            representation_kwargs=entity_representations_kwargs,
        )
        self.relation_representations = build_representation(
            max_id=2 * triples_factory.real_num_relations,
            representation=relation_representations,
            representation_kwargs=relation_representations_kwargs,
        )
        if len(self.entity_representations.shape) > 1:
            raise ValueError(f"{self.__class__.__name__} requires vector base entity representations.")
        input_dim = self.entity_representations.shape[0]
        # TODO: might not be true for all compositions
        if self.relation_representations.shape != self.entity_representations.shape:
            raise ValueError(
                f"{self.__class__.__name__} requires entity and relation representations of the same shape."
            )

        # hidden dimension normalization
        if dims is None:
            dims = input_dim
        if isinstance(dims, int):
            if num_layers is None:
                raise ValueError
            else:
                dims = [dims] * num_layers
        if len(dims) != num_layers:
            raise ValueError(
                f"The number of provided dimensions ({len(dims)}) must equal the number of layers ({num_layers}).",
            )
        self.output_dim = dims[-1]

        # Create message passing layers
        layers = []
        for input_dim_, output_dim in zip(itertools.chain([input_dim], dims), dims, strict=False):
            layers.append(
                CompGCNLayer(
                    input_dim=input_dim_,
                    output_dim=output_dim,
                    **(layer_kwargs or {}),
                )
            )
        self.layers = nn.ModuleList(layers)

        # register buffers for adjacency matrix; we use the same format as PyTorch Geometric
        # TODO: This always uses all training triples for message passing
        self.register_buffer(name="edge_index", tensor=get_edge_index(triples_factory=triples_factory))
        self.register_buffer(name="edge_type", tensor=triples_factory.mapped_triples[:, 1])

        # initialize buffer of enriched representations
        self.enriched_representations = None

    # docstr-coverage: inherited
    def post_parameter_update(self) -> None:  # noqa: D102
        # invalidate enriched embeddings
        self.enriched_representations = None

    # docstr-coverage: inherited
    def train(self, mode: bool = True):  # noqa: D102
        # when changing from evaluation to training mode, the buffered representations have been computed without
        # gradient tracking. hence, we need to invalidate them.
        # note: this occurs in practice when continuing training after evaluation.
        if mode and not self.training:
            self.enriched_representations = None
        return super().train(mode=mode)

    def forward(
        self,
    ) -> tuple[FloatTensor, FloatTensor]:
        """Compute enriched representations."""
        if self.enriched_representations is None:
            x_e = self.entity_representations()
            x_r = self.relation_representations()
            # enrich
            for layer in self.layers:
                x_e, x_r = layer(x_e=x_e, x_r=x_r, edge_index=self.edge_index, edge_type=self.edge_type)
            self.enriched_representations = (x_e, x_r)
        return self.enriched_representations

    def split(self) -> tuple[SingleCompGCNRepresentation, SingleCompGCNRepresentation]:
        """Return the separated representations."""
        return (
            SingleCompGCNRepresentation(self, position="entity"),
            SingleCompGCNRepresentation(self, position="relation"),
        )


@parse_docdata
class SingleCompGCNRepresentation(Representation):
    """A wrapper around the combined representation module.

    .. seealso::
        :class:`pykeen.nn.representation.CombinedCompGCNRepresentations`

    ---
    name: CompGCN
    citation:
        author: Vashishth
        year: 2020
        link: https://arxiv.org/pdf/1911.03082
        github: malllabiisc/CompGCN
    """

    position: int

    def __init__(
        self,
        combined: CombinedCompGCNRepresentations,
        position: Literal["entity", "relation"] = "entity",
        shape: OneOrSequence[int] | None = None,
        **kwargs,
    ):
        """
        Initialize the module.

        :param combined:
            The combined representations.
        :param position:
            The position.
        :param shape:
            The shape of an individual representation.
        :param kwargs:
            Additional keyword-based parameters passed to :class:`pykeen.nn.representation.Representation`.

        :raises ValueError:
            If an invalid value is given for the position.
        """
        if position == "entity":
            max_id = combined.entity_representations.max_id
            shape_ = (combined.output_dim,)
            position_index = 0
        elif position == "relation":
            max_id = combined.relation_representations.max_id
            shape_ = (combined.output_dim,)
            position_index = 1
        else:
            raise ValueError(f"invalid position: {position}")
        super().__init__(max_id=max_id, shape=ShapeError.verify(shape=shape_, reference=shape), **kwargs)
        self.combined = combined
        self.position = position_index
        self.reset_parameters()

    # docstr-coverage: inherited
    def _plain_forward(
        self,
        indices: LongTensor | None = None,
    ) -> FloatTensor:  # noqa: D102
        x = self.combined()[self.position]
        if indices is not None:
            x = x[indices.to(self.device)]
        return x


def _clean_labels(labels: Sequence[str | None], missing_action: Literal["error", "blank"]) -> Sequence[str]:
    if missing_action == "error":
        idx = [i for i, label in enumerate(labels) if label is None]
        if idx:
            raise ValueError(
                f"The labels at the following indexes were none. Consider an alternate `missing_action` policy.\n{idx}",
            )
        return cast(Sequence[str], labels)
    elif missing_action == "blank":
        return [label or "" for label in labels]
    else:
        raise ValueError(f"Invalid `missing_action` policy: {missing_action}")


@parse_docdata
class TextRepresentation(Representation):
    """
    Textual representations using a text encoder on labels.

    Example Usage:

    Entity representations are obtained by encoding the labels with a Transformer model. The transformer
    model becomes part of the KGE model, and its parameters are trained jointly.

    .. literalinclude:: ../examples/nn/representation/text_based.py

    ---
    name: Text Encoding
    """

    labels: list[str]

    @update_docstring_with_resolver_keys(ResolverKey("encoder", resolver="pykeen.nn.text.text_encoder_resolver"))
    def __init__(
        self,
        labels: Sequence[str | None],
        max_id: int | None = None,
        shape: OneOrSequence[int] | None = None,
        encoder: HintOrType[TextEncoder] = None,
        encoder_kwargs: OptionalKwargs = None,
        missing_action: Literal["blank", "error"] = "error",
        **kwargs: Any,
    ):
        """
        Initialize the representation.

        :param labels:
            An ordered, finite collection of labels.
        :param max_id:
            The number of representations. If provided, has to match the number of labels.
        :param shape:
            The shape of an individual representation.

        :param encoder:
            The text encoder, or a hint thereof.
        :param encoder_kwargs:
            Keyword-based parameters used to instantiate the text encoder.

        :param missing_action:
            Which policy for handling nones in the given labels. If "error", raises an error
            on any nones. If "blank", replaces nones with an empty string.
        :param kwargs:
            Additional keyword-based parameters passed to :class:`pykeen.nn.representation.Representation`

        :raises MaxIDMismatchError:
            if the ``max_id`` was given explicitly and does not match the length of the labels
        """
        encoder = text_encoder_resolver.make(encoder, encoder_kwargs)

        if max_id is None:
            max_id = len(labels)
        elif max_id != len(labels):
            raise MaxIDMismatchError(f"max_id={max_id} does not match len(labels)={len(labels)}")

        labels = _clean_labels(labels, missing_action)
        # infer shape
        shape = ShapeError.verify(shape=encoder.encode_all(labels[0:1]).shape[1:], reference=shape)
        super().__init__(max_id=max_id, shape=shape, **kwargs)
        self.labels = list(labels)
        # assign after super, since they should be properly registered as submodules
        self.encoder = encoder

    @classmethod
    def from_triples_factory(
        cls,
        triples_factory: TriplesFactory,
        for_entities: bool = True,
        **kwargs,
    ) -> TextRepresentation:
        """
        Prepare a text representations with labels from a triples factory.

        :param triples_factory:
            The triples factory.
        :param for_entities:
            Whether to create the initializer for entities (or relations).
        :param kwargs:
            Additional keyword-based arguments passed to :class:`pykeen.nn.representation.TextRepresentation`

        :returns:
            a text representation from the triples factory
        """
        labeling: Labeling = triples_factory.entity_labeling if for_entities else triples_factory.relation_labeling
        return cls(labels=labeling.all_labels(), **kwargs)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        for_entities: bool = True,
        **kwargs,
    ) -> TextRepresentation:
        """Prepare text representation with labels from a dataset.

        :param dataset:
            The dataset.
        :param for_entities:
            Whether to create the initializer for entities (or relations).
        :param kwargs:
            Additional keyword-based arguments passed to :class:`pykeen.nn.representation.TextRepresentation`

        :return:
            A text representation from the dataset.

        :raises TypeError:
            If the dataset's triples factory does not provide labels.
        """
        if not isinstance(dataset.training, TriplesFactory):
            raise TypeError(f"{cls.__name__} requires access to labels, but dataset.training does not provide such.")
        return cls.from_triples_factory(triples_factory=dataset.training, **kwargs)

    # docstr-coverage: inherited
    def _plain_forward(
        self,
        indices: LongTensor | None = None,
    ) -> FloatTensor:  # noqa: D102
        if indices is None:
            labels = self.labels
        else:
            labels = [self.labels[i] for i in indices.tolist()]
        return self.encoder(labels=labels)


@parse_docdata
class CombinedRepresentation(Representation):
    """Combined representation.

    It has a sequence of base representations, each providing a representation for each index.
    A combination is used to combine the multiple representations for the same index into a single one.

    Example usage:

    .. literalinclude:: ../examples/nn/representation/text_wikidata.py

    ---
    name: Combined
    """

    #: the base representations
    base: Sequence[Representation]

    #: the combination module
    combination: Combination

    @update_docstring_with_resolver_keys(
        ResolverKey("combination", combination_resolver),
        ResolverKey(name="base", resolver="pykeen.nn.representation_resolver"),
    )
    def __init__(
        self,
        max_id: int | None,
        shape: OneOrSequence[int] | None = None,
        unique: bool | None = None,
        base: OneOrManyHintOrType[Representation] = None,
        base_kwargs: OneOrManyOptionalKwargs = None,
        # TODO: we could relax that to Callable[[Sequence[Representation]], Representation]
        #   to make it easier to create ad-hoc combinations
        combination: HintOrType[Combination] = None,
        combination_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        """
        Initialize the representation.

        :param max_id:
            The number of representations. If `None`, it will be inferred from the base representations.
        :param shape:
            The shape of an individual representation.
        :param unique:
            Whether to optimize for calculating representations for same indices only once. This is only useful if the
            calculation of representations is significantly more expensive than an index-based lookup and
            duplicate indices are expected, e.g., when using negative sampling and large batch sizes.
            If `None` it is inferred from the base representations.

            .. warning ::
                When using this optimization you may encounter unexpected results for stochastic operations, e.g.,
                :class:`torch.nn.Dropout`.

        :param base:
            The base representations, or hints thereof.
        :param base_kwargs:
            Keyword-based parameters for the instantiation of base representations.

        :param combination:
            The combination, or a hint thereof.
        :param combination_kwargs:
            Additional keyword-based parameters used to instantiate the combination.

        :param kwargs:
            Additional keyword-based parameters passed to :class:`pykeen.nn.representation.Representation`.

        :raises ValueError:
            If the `max_id` of the base representations are not all the same
        :raises MaxIDMismatchError:
            if the ``max_id`` was given explicitly and does not match the bases' ``max_id``
        """
        # input normalization
        combination = combination_resolver.make(combination, combination_kwargs)

        # has to be imported here to avoid cyclic import
        from . import representation_resolver

        # create base representations
        base = representation_resolver.make_many(base, kwargs=merge_kwargs(base_kwargs, max_id=max_id))

        # verify same ID range
        max_ids = sorted(set(b.max_id for b in base))
        if len(max_ids) != 1:
            # note: we could also relax the requirement, and set max_id = min(max_ids)
            raise ValueError(
                f"Maximum number of IDs are not the same in all base representations. Unique max_ids={max_ids}"
            )

        if max_id is None:
            max_id = max_ids[0]
        elif max_id != max_ids[0]:
            raise MaxIDMismatchError(f"max_id={max_id} does not match base max_id={max_ids[0]}")

        if unique is None:
            unique = all(b.unique for b in base)

        # shape inference
        shape = ShapeError.verify(shape=combination.output_shape(input_shapes=[b.shape for b in base]), reference=shape)
        super().__init__(max_id=max_id, shape=shape, unique=unique, **kwargs)

        # assign base representations *after* super init
        self.base = nn.ModuleList(base)
        self.combination = combination

    @staticmethod
    def _combine(
        combination: nn.Module, base: Sequence[Representation], indices: LongTensor | None = None
    ) -> FloatTensor:
        """
        Combine base representations for the given indices.

        :param combination: The combination.
        :param base: The base representations.
        :param indices: The indices, as given to :meth:`Representation._plain_forward`.

        :return:
            The combined representations for the given indices.
        """
        return combination([b._plain_forward(indices=indices) for b in base])

    # docstr-coverage: inherited
    def _plain_forward(
        self,
        indices: LongTensor | None = None,
    ) -> FloatTensor:  # noqa: D102
        return self._combine(combination=self.combination, base=self.base, indices=indices)


class CachedTextRepresentation(TextRepresentation):
    """Textual representations for datasets with identifiers that can be looked up with a :class:`TextCache`."""

    cache_cls: ClassVar[type[TextCache]]

    def __init__(self, identifiers: Sequence[str], cache: TextCache | None = None, **kwargs):
        """
        Initialize the representation.

        :param identifiers:
            the IDs to be resolved by the class, e.g., wikidata IDs. for :class:`WikidataTextRepresentation`,
            biomedical entities represented as compact URIs (CURIEs) for :class:`BiomedicalCURIERepresentation`
        :param cache:
            a pre-instantiated text cache. If None, :attr:`cache_cls` is used to instantiate one.
        :param kwargs:
            additional keyword-based parameters passed to :meth:`TextRepresentation.__init__`
        """
        cache = self.cache_cls() if cache is None else cache
        labels = cache.get_texts(identifiers=identifiers)
        # delegate to super class
        super().__init__(labels=labels, **kwargs)

    # docstr-coverage: inherited
    @classmethod
    def from_triples_factory(
        cls,
        triples_factory: TriplesFactory,
        for_entities: bool = True,
        **kwargs,
    ) -> TextRepresentation:  # noqa: D102
        labeling: Labeling = triples_factory.entity_labeling if for_entities else triples_factory.relation_labeling
        return cls(identifiers=labeling.all_labels().tolist(), **kwargs)


@parse_docdata
class WikidataTextRepresentation(CachedTextRepresentation):
    """
    Textual representations for datasets grounded in Wikidata.

    The label and description for each entity are obtained from Wikidata using
    :class:`~pykeen.nn.text.cache.WikidataTextCache` and encoded with
    :class:`~pykeen.nn.representation.TextRepresentation`.

    Example usage:

    .. literalinclude:: ../examples/nn/representation/text_wikidata.py

    ---
    name: Wikidata Text Encoding
    """

    cache_cls = WikidataTextCache


class BiomedicalCURIERepresentation(CachedTextRepresentation):
    """
    Textual representations for datasets grounded with biomedical CURIEs.

    The label and description for each entity are obtained via :mod:`pyobo` using
    :class:`~pykeen.nn.text.cache.PyOBOTextCache` and encoded with
    :class:`~pykeen.nn.representation.TextRepresentation`.

    Example usage:

    .. literalinclude:: ../examples/nn/representation/text_curie.py

    ---
    name: Biomedical CURIE Text Encoding
    """

    cache_cls = PyOBOTextCache


@parse_docdata
class PartitionRepresentation(Representation):
    """
    A partition of the indices into different representation modules.

    Each index is assigned to an index in exactly one of the base representations. This representation is useful, e.g.,
    when one of the base representations cannot provide vectors for each of the indices, and another representation is
    used as back-up.

    Consider the following example: We only have textual information for two entities. We want to use textual features
    computed from them, which should not be trained. For the remaining entities we want to use directly trainable
    embeddings.

    .. literalinclude:: ../examples/nn/representation/partition.py

    .. note ::
        For this simple but often occuring case, we provide a more convenient specialized
        :class:`~pykeen.nn.representation.BackfillRepresentation`.

    ---
    name: Partition
    """

    #: the assignment from global ID to (representation, local id), shape: (max_id, 2)
    assignment: LongTensor

    @update_docstring_with_resolver_keys(ResolverKey(name="bases", resolver="pykeen.nn.representation_resolver"))
    def __init__(
        self,
        assignment: LongTensor,
        shape: OneOrSequence[int] | None = None,
        bases: OneOrSequence[HintOrType[Representation]] = None,
        bases_kwargs: OneOrSequence[OptionalKwargs] = None,
        **kwargs,
    ):
        """
        Initialize the representation.

        .. warning ::
            The base representations have to have coherent shapes.

        :param assignment: shape: (max_id, 2)
            the assignment, as tuples `(base_id, local_id)`, where `base_id` refers to the index of the base
            representation and `local_id` is an index used to lookup in the base representation
        :param shape:
            the shape of an individual representation. If provided, must match the bases' shape
        :param bases:
            the base representations, or hints thereof.
        :param bases_kwargs:
            keyword-based parameters to instantiate the base representations
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Representation.__init__`. May not contain `max_id`,
            or `shape`, which are inferred from the base representations.

        :raises ValueError:
            if any of the inputs is invalid
        """
        # import here to avoid cyclic import
        from . import representation_resolver

        # instantiate base representations if necessary
        bases = representation_resolver.make_many(bases, bases_kwargs)

        # there needs to be at least one base
        if not bases:
            raise ValueError("Must provide at least one base representation")
        # while possible, this might be unintended
        if len(bases) == 1:
            logger.warning(f"Encountered only a single base representation: {bases[0]}")

        # extract shape
        shapes = [base.shape for base in bases]
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent base shapes: {shapes}")
        shape = ShapeError.verify(shape=shapes[0], reference=shape)

        # check for invalid base ids
        unknown_base_ids = set(assignment[:, 0].tolist()).difference(range(len(bases)))
        if unknown_base_ids:
            raise ValueError(f"Invalid representation Ids in assignment: {unknown_base_ids}")

        # check for invalid local indices
        for i, base in enumerate(bases):
            base_assignment = assignment[:, 0] == i
            if not base_assignment.any():
                raise ValueError(
                    f"base {base} (index:{i}) is assigned to nothing, meaning that it should be removed from the bases"
                )
            max_index = assignment[base_assignment, 1].max().item()
            if max_index >= base.max_id:
                raise ValueError(f"base {base} (index:{i}) cannot provide indices up to {max_index}")

        super().__init__(max_id=assignment.shape[0], shape=shape, **kwargs)

        # assign modules / buffers *after* super init
        self.bases = nn.ModuleList(bases)
        self.register_buffer(name="assignment", tensor=assignment)

    # docstr-coverage: inherited
    def _plain_forward(self, indices: LongTensor | None = None) -> FloatTensor:  # noqa: D102
        assignment = self.assignment
        if indices is not None:
            assignment = assignment[indices]
        # flatten assignment to ease construction of inverse indices
        prefix_shape = assignment.shape[:-1]
        assignment = assignment.view(-1, 2)
        # we group indices by the representation which provides them
        # thus, we need an inverse to restore the correct order
        inverse = torch.empty_like(assignment[:, 0])
        xs = []
        offset = 0
        for i, base in enumerate(self.bases):
            mask = assignment[:, 0] == i
            # get representations
            local_indices = assignment[:, 1][mask]
            xs.append(base(indices=local_indices))
            # update inverse indices
            end = offset + local_indices.numel()
            inverse[mask] = torch.arange(offset, end, device=inverse.device)
            offset = end
        x = torch.cat(xs, dim=0)[inverse]
        # invert flattening
        if len(prefix_shape) != 1:
            x = x.view(*prefix_shape, *x.shape[1:])
        return x


@dataclasses.dataclass
class Partition:
    """A specification for a backfill representation."""

    #: The global identifiers for the entities, appearing in the order that
    #: corresponds to their local identifier within the base representation
    ids: Sequence[int]

    #: the base representation, or a hint to generate it
    base: HintOrType[Representation]

    #: the optional keyword arguments used to instantiate the base representation
    kwargs: OptionalKwargs | None = None

    def __post_init__(self) -> None:
        """Implement data integrity checks."""
        if len(set(self.ids)) != len(self.ids):
            raise InvalidBaseIdsError(f"Duplicate in {self.ids=}")
        if any(i < 0 for i in self.ids):
            raise InvalidBaseIdsError(f"Some of the {self.ids=} are not non-negative.")

    def get_base(self) -> Representation:
        """Instantiate the base representation."""
        # import here to avoid cyclic import
        from . import representation_resolver

        max_id = len(self.ids)

        base = representation_resolver.make(self.base, self.kwargs, max_id=max_id)
        if base.max_id != max_id:
            raise MaxIDMismatchError(
                f"When constructing the backfill specification, got a mismatch between the number of IDs "
                f"given ({max_id:,}) and the max_id assigned to the base representation ({base.max_id:,})"
            )

        return base


class InvalidBaseIdsError(ValueError):
    """Raised when the provided base ids are invalid."""


@parse_docdata
class MultiBackfillRepresentation(PartitionRepresentation):
    """Fill missing ids by backfill representation.

    ---
    name: Multi-Backfill
    """

    def __init__(
        self,
        *,
        max_id: int,
        partitions: Sequence[Partition],
        shape: OneOrSequence[int] | None = None,
        backfill: HintOrType[Representation] = None,
        backfill_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        """Initialize the backfill representation."""
        # import here to avoid cyclic import
        from . import representation_resolver

        bases: list[Representation] = []
        # format: (base_index, local_index)
        assignment = torch.zeros(size=(max_id, 2), dtype=torch.long)
        backfill_mask = torch.ones(assignment.shape[0], dtype=torch.bool)
        all_ids: set[int] = set()
        for base_index, partition in enumerate(partitions):
            if max(partition.ids) >= max_id:
                raise InvalidBaseIdsError(f"Some of the {partition.ids=} exceed {max_id=:_}")

            # check for overlap with others
            if colliding_ids := all_ids.intersection(partition.ids):
                raise ValueError(f"{colliding_ids=} for bases[{base_index}] with {partition.ids=}")
            all_ids.update(partition.ids)

            # set assignment
            ids_t = torch.as_tensor(partition.ids, dtype=torch.long)
            assignment[ids_t, 0] = base_index
            assignment[ids_t, 1] = torch.arange(len(partition.ids))
            backfill_mask[ids_t] = False

            bases.append(partition.get_base())

        # shape verification
        shapes = [base.shape for base in bases]
        if len(set(shapes)) != 1:
            raise BaseShapeError(f"Base instances had multiple different shapes: {shapes}")
        shape = ShapeError.verify(shapes[0], shape)

        # check number of backfill representations
        num_total_base_ids = len(all_ids)
        if max_id < num_total_base_ids:
            raise MaxIDMismatchError(
                f"The given {max_id=:_} was less than the number of unique IDs given in the backfill specification, "
                f"{num_total_base_ids=:_}"
            )
        elif max_id == num_total_base_ids:
            logger.warning(
                f"The given {max_id=:_} was equivalent to the number of unique IDs given in the backfill "
                f"specification, {num_total_base_ids=:_}. This means that no backfill representation is necessary, "
                f"and instead this will be a simple PartitionRepresentation."
            )
        else:
            # this block is part of the "else" to make sure that we only create a backfill
            # if there are some remaining IDs. This is necessary to make sure we don't
            # create a representation with an empty dimension.
            backfill_max_id = max_id - num_total_base_ids
            backfill = representation_resolver.make(
                backfill, merge_kwargs(backfill_kwargs, max_id=backfill_max_id), shape=shape
            )
            if backfill_max_id != backfill.max_id:
                raise MaxIDMismatchError(
                    f"Mismatch between {backfill_max_id=} and {backfill.max_id=} of "
                    f"explicitly provided backfill instance."
                )
            # set backfill assignment
            # since the backfill comes last, and it has not been added to the bases list yet:
            assignment[backfill_mask, 0] = len(bases)
            assignment[backfill_mask, 1] = torch.arange(backfill.max_id)
            bases.append(backfill)

        super().__init__(assignment=assignment, bases=bases, shape=shape, **kwargs)


@parse_docdata
class BackfillRepresentation(MultiBackfillRepresentation):
    """A variant of a partition representation that is easily applicable to a single base representation.

    .. literalinclude:: ../examples/nn/representation/backfill.py

    ---
    name: Backfill
    """

    @update_docstring_with_resolver_keys(
        ResolverKey(name="base", resolver="pykeen.nn.representation_resolver"),
        ResolverKey(name="backfill", resolver="pykeen.nn.representation_resolver"),
    )
    def __init__(
        self,
        max_id: int,
        base_ids: Iterable[int],
        base: HintOrType[Representation] = None,
        base_kwargs: OptionalKwargs = None,
        backfill: HintOrType[Representation] = None,
        backfill_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        """Initialize the representation.

        :param max_id:
            The total number of entities that need to be represented.
        :param base_ids:
            The indices (in the new, increased indexing scheme)
            which are provided through the base representation.

        :param base:
            The base representation, or a hint thereof.
        :param base_kwargs:
            Keyword-based parameters to instantiate the base representation

        :param backfill:
            The backfill representation, or hints thereof.
        :param backfill_kwargs:
            Keyword-based parameters to instantiate the backfill representation.

        :param kwargs:
            additional keyword-based parameters passed to :class:`~pykeen.nn.representation.Representation`.
            May not contain `shape`, which is inferred from the base representation.

        .. warning ::
            The base and backfill representations have to have coherent shapes.
            If the backfill representation is initialized within this constructor,
            it will receive the base representation's shape.
        """
        super().__init__(
            max_id=max_id,
            partitions=[Partition(list(base_ids), base, base_kwargs)],
            backfill=backfill,
            backfill_kwargs=backfill_kwargs,
            **kwargs,
        )


@parse_docdata
class TransformedRepresentation(Representation):
    """
    A (learnable) transformation upon base representations.

    In the following example, we create representations which are obtained from a trainable transformation of fixed
    random walk encoding features, and transform them using a 2-layer MLP.

    .. literalinclude:: ../examples/nn/representation/transformed.py

    ---
    name: Transformed
    """

    @update_docstring_with_resolver_keys(ResolverKey(name="base", resolver="pykeen.nn.representation_resolver"))
    def __init__(
        self,
        transformation: nn.Module,
        max_id: int | None = None,
        shape: OneOrSequence[int] | None = None,
        base: HintOrType[Representation] = None,
        base_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        """
        Initialize the representation.

        :param transformation:
            the transformation
        :param max_id:
            the number of representations. If provided, must match the base max id
        :param shape:
            the individual representations' shape. If provided, must match the output shape of the transformation
        :param base:
            the base representation, or a hint thereof, cf. `representation_resolver`
        :param base_kwargs:
            keyword-based parameters used to instantiate the base representation
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Representation.__init__`.

        :raises MaxIDMismatchError:
            if the ``max_id`` was given explicitly and does not match the base's ``max_id``
        """
        # import here to avoid cyclic import
        from . import representation_resolver

        base = representation_resolver.make(base, merge_kwargs(base_kwargs, max_id=max_id))

        # infer shape
        shape = ShapeError.verify(
            shape=self._help_forward(
                base=base, transformation=transformation, indices=torch.zeros(1, dtype=torch.long, device=base.device)
            ).shape[1:],
            reference=shape,
        )
        if max_id is None:
            max_id = base.max_id
        elif max_id != base.max_id:
            raise MaxIDMismatchError(f"Incompatible max_id={max_id} vs. base.max_id={base.max_id}")

        super().__init__(max_id=max_id, shape=shape, **kwargs)
        self.transformation = transformation
        self.base = base

    @staticmethod
    def _help_forward(base: Representation, transformation: nn.Module, indices: LongTensor | None) -> FloatTensor:
        """
        Obtain base representations and apply the transformation.

        :param base:
            the base representation module
        :param transformation:
            the transformation
        :param indices:
            the indices

        :return:
            the transformed base representations
        """
        return transformation(base(indices=indices))

    # docstr-coverage: inherited
    def _plain_forward(self, indices: LongTensor | None = None) -> FloatTensor:  # noqa: D102
        return self._help_forward(base=self.base, transformation=self.transformation, indices=indices)


# TODO: can be a combined representations, with appropriate tensor-train combination
@parse_docdata
class TensorTrainRepresentation(Representation):
    r"""
    A tensor train factorization of representations.

    In the simple case without provided assignment this corresponds to ``TT-emb`` described in [yin2022]_.

    where

    .. math ::

        \mathbf{A}[i_1 \cdot \ldots \cdot i_k, j_1 \cdot \ldots \cdot j_k]
            = \sum_{r_i, \ldots, r_k} \mathbf{G}_1[0, i_1, j_1, r_1]
                \cdot \mathbf{G}_2[r_1, i_2, j_2, r_2]
                \cdot \ldots
                \cdot \mathbf{G}_k[r_k, i_k, j_k, 0]

    with TT core $\mathbf{G}_i$ of shape $R_{i-1} \times m_i \times n_i \times R_i$ and $R_0 = R_d = 1$.

    Another variant in the paper used an assignment based on hierarchical topological clustering.

    .. seealso::
        - `Wikipedia: Matrix Product State <https://en.wikipedia.org/wiki/Matrix_product_state>`_
        - `TensorLy: Matrix-Product-State / Tensor-Train Decomposition
          <http://tensorly.org/stable/user_guide/tensor_decomposition.html#matrix-product-state-tensor-train-decomposition>`_

    ---
    name: Tensor-Train
    author: Yin
    year: 2022
    arxiv: 2206.10581
    link: https://arxiv.org/abs/2206.10581
    """

    #: shape: (max_id, num_cores)
    assignment: LongTensor

    #: the bases, length: num_cores, with compatible shapes
    bases: Sequence[Representation]

    @classmethod
    def factor_sizes(cls, max_id: int, shape: Sequence[int], num_cores: int) -> tuple[Sequence[int], Sequence[int]]:
        r"""Factor the representation shape into smaller shapes for the cores.

        .. note ::
            This method implements a very simple heuristic of using the same value for each $m_i$ / $n_i$.

        :param max_id:
            The number of representations, "row count", $M$.
        :param shape:
            The shape of an individual representation, "column count", $N$.
        :param num_cores:
            The number of cores, $k$.

        :return:
            A tuple ``(ms, ns)`` of positive integer sequences of length $k$ fulfilling

            .. math ::

                M \leq \prod \limits_{m_i \in \textit{ms}} m_i \quad
                N \leq \prod \limits_{n_i \in \textit{ns}} n_i
        """
        m_k = int(math.ceil(max_id ** (1 / num_cores)))
        n_k = int(math.ceil(numpy.prod(shape) ** (1 / num_cores)))
        return [m_k] * num_cores, [n_k] * num_cores

    @staticmethod
    def check_assignment(assignment: torch.Tensor, max_id: int, num_cores: int, ms: Sequence[int]) -> None:
        """
        Check that the assignment match in shape and its values are valid core "row" indices.

        :param assignment: shape: ``(max_id, num_cores)``
            The assignment.
        :param max_id:
            The number of representations.
        :param num_cores:
            The number of tensor-train cores.
        :param ms:
            The individual sizes $m_i$.

        :raises ValueError:
            If the assignment is invalid.
        """
        # check shape
        if assignment.shape != (max_id, num_cores):
            raise ValueError(
                f"Invalid assignment. Expected shape (max_id, num_cores)={(max_id, num_cores)}, "
                f"but got assignment.shape={assignment.shape}",
            )
        # check value range
        low, high = assignment.min(dim=0).values, assignment.max(dim=0).values
        if (low < 0).any() or (high >= torch.as_tensor(ms, dtype=torch.long)).any():
            raise ValueError(
                f"Invalid values inside assignment: ms={ms} vs. assignment.min(dim=0)={low} "
                f"and assignment.max(dim=0)={high}",
            )

    @staticmethod
    def get_shapes_and_einsum_eq(ranks: Sequence[int], ns: Sequence[int]) -> tuple[str, Sequence[tuple[int, ...]]]:
        """
        Determine core shapes and einsum equation.

        :param ranks:
            The core ranks.
        :param ns:
            The sizes $n_i$.
        :return:
            A pair ``(eq, shapes)``, where ``eq`` is a valid einsum equation and ``shapes`` a sequence of representation
            shapes. Notice that the shapes do not include the "``max_id`` dimension" of the resulting embedding.
        """
        shapes: list[list[int]] = []
        terms: list[list[str]] = []
        out_term: list[str] = ["..."]
        i = 0
        for n_i, (rank_in, rank_out) in zip(ns, more_itertools.pairwise([None, *ranks, None]), strict=False):
            shape = []
            term = ["..."]

            if rank_in is not None:
                shape.append(rank_in)
                term.append(string.ascii_lowercase[i])
                i += 1

            shape.append(n_i)
            term.append(string.ascii_lowercase[i])
            out_term.append(string.ascii_lowercase[i])
            i += 1

            if rank_out is not None:
                shape.append(rank_out)
                term.append(string.ascii_lowercase[i])
                # do not increase counter i, since the dimension is shared with the following term
                # i += 1

            terms.append(term)
            shapes.append(shape)
        eq = " ".join((", ".join("".join(term) for term in terms), "->", "".join(out_term)))
        return eq, [tuple(shape) for shape in shapes]

    @staticmethod
    def create_default_assignment(max_id: int, num_cores: int, ms: Sequence[int]) -> LongTensor:
        """
        Create an assignment without using structural information.

        :param max_id:
            The number of representations.
        :param num_cores:
            The number of tensor cores.
        :param ms:
            The sizes $m_i$.

        :return: shape: ``(max_id, num_cores)``
            The assignment.
        """
        assignment = torch.empty(max_id, num_cores, dtype=torch.long)
        ids = torch.arange(max_id)
        for i, m_i in enumerate(ms):
            assignment[:, i] = ids % m_i
            # ids //= m_i
            ids = torch.div(ids, m_i, rounding_mode="floor")
        return assignment

    @staticmethod
    def check_factors(
        ms: Sequence[int], ns: Sequence[int], max_id: int, shape: tuple[int, ...], num_cores: int
    ) -> None:
        r"""
        Check whether the factors match the other parts.

        Verifies that

        .. math ::
            M \leq \prod \limits_{m_i \in \textit{ms}} m_i \quad
            N \leq \prod \limits_{n_i \in \textit{ns}} n_i

        :param ms: length: ``num_cores``
            The $M$ factors $m_i$.
        :param ns: length: ``num_cores``
            The $N$ factors $n_i$.
        :param max_id:
            The maximum id, $M$.
        :param shape:
            The shape, $N=prod(shape)$.
        :param num_cores:
            The number of cores.

        :raises ValueError:
            If any of the conditions is violated.
        """
        if len(ms) != num_cores or len(ns) != num_cores:
            raise ValueError(f"Invalid length: len(ms)={len(ms)}, len(ns)={len(ns)} vs. num_cores={num_cores}")

        m_prod = numpy.prod(ms).item()
        if m_prod < max_id:
            raise ValueError(f"prod(ms)={m_prod} < max_id={max_id}")

        n_prod = numpy.prod(ns).item()
        s_prod = numpy.prod(shape).item()
        if n_prod < s_prod:
            raise ValueError(f"prod(ns)={n_prod} < prod(shape)={s_prod}")

    @update_docstring_with_resolver_keys(ResolverKey(name="bases", resolver="pykeen.nn.representation_resolver"))
    def __init__(
        self,
        assignment: LongTensor | None = None,
        num_cores: int = 3,
        ranks: OneOrSequence[int] = 2,
        bases: OneOrManyHintOrType = None,
        bases_kwargs: OneOrManyOptionalKwargs = None,
        **kwargs,
    ) -> None:
        """Initialize the representation.

        :param assignment: shape: ``(max_id, num_cores)``
            The core-assignment for each index on each level. If ``None``, :meth:`create_default_assignment` is used.
        :param num_cores:
            The number of cores to use.
        :param ranks: length: ``num_cores - 1``
            The individual ranks for each core. Note that $R_0 = R_d = 1$ should not be included.
        :param bases:
            The base representations for each level, or hints thereof.
        :param bases_kwargs:
            Keyword-based parameters for the bases.
        :param kwargs:
            Additional keyword-based parameters passed to :class:`~pykeen.nn.representation.Representation`

        :raises ValueError:
            If the input validation on ranks or assignment failed.
        """
        # import here to avoid cyclic import
        from . import representation_resolver

        super().__init__(**kwargs)

        # normalize ranks
        ranks = list(upgrade_to_sequence(ranks))
        if len(ranks) == 1:
            ranks = ranks * (num_cores - 1)
        if len(ranks) != num_cores - 1:
            raise ValueError(f"Inconsistent number of ranks {len(ranks)} for num_cores={num_cores}")

        # determine M_k, N_k
        # TODO: allow to pass them from outside?
        ms, ns = self.factor_sizes(max_id=self.max_id, shape=self.shape, num_cores=num_cores)
        self.check_factors(ms, ns, max_id=self.max_id, shape=self.shape, num_cores=num_cores)

        # normalize assignment
        if assignment is None:
            assignment = self.create_default_assignment(max_id=self.max_id, num_cores=num_cores, ms=ms)
        self.check_assignment(assignment=assignment, max_id=self.max_id, num_cores=num_cores, ms=ms)
        self.register_buffer(name="assignment", tensor=assignment)

        # determine shapes and einsum equation
        self.eq, shapes = self.get_shapes_and_einsum_eq(ranks=ranks, ns=ns)

        # create base representations
        self.bases = nn.ModuleList(
            representation_resolver.make(base, base_kwargs, max_id=m_i, shape=shape)
            for base, base_kwargs, m_i, shape in zip(
                *broadcast_upgrade_to_sequences(bases, bases_kwargs, ms, shapes), strict=False
            )
        )

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"num_cores={len(self.bases)}"
        yield f"eq='{self.eq}'"

    # docstr-coverage: inherited
    def _plain_forward(self, indices: LongTensor | None = None) -> FloatTensor:  # noqa: D102
        assignment = self.assignment
        if indices is not None:
            assignment = assignment[indices]
        return einsum(
            self.eq, *(base(indices) for indices, base in zip(assignment.unbind(dim=-1), self.bases, strict=False))
        ).view(*assignment.shape[:-1], *self.shape)


class EmbeddingBagRepresentation(Representation):
    r"""
    An embedding bag representation.

    :class:`~torch.nn.EmbeddingBag` is similar to a :class:`~pykeen.nn.TokenRepresentation`
    followed by an aggregation along the `num_tokens` dimension.

    Its main differences are:

        - It fuses the token look-up and aggregation step in a single torch call.
        - It only allows for a limited set of non-parametric aggregations:
          :func:`~torch.sum`, :func:`~torch.mean`, or :func:`~torch.max`
        - It can handle sparse/variable number of tokens per input more naturally.
        - It always uses an :class:`~torch.nn.Embedding` layer instead of permitting an arbitrary
          :class:`~pykeen.nn.Representation`

    If you have a boolean feature vector, for example, from a chemical fingerprint, you
    can construct an embedding bag with the following

    .. code-block:: python

        features: torch.BoolTensor = ...

        representation = EmbeddingBagRepresentation.from_iter(
            list(feature.nonzero())
            for feature in features
        )

    Let's denote $nnz(i)$ for the non-zero indices of the feature of molecule $i$,
    then we build the following representation $\mathbf{x}_i$

    .. math ::
        \mathbf{x}_i := \sum \limits_{j \in nnz(i)} \mathbf{y}_j

    where $\mathbf{y}_j$ is the embedding for the substructure represented by
    dimension $j$ in the signature. In a sense, it is very similar to using the
    0/1 vectors and multiplying that with a matrix; it's just implemented more
    efficiently (exploiting the sparsity).
    """

    # shape: (nnz, 2), entries: (index, comp_index)
    assignment: LongTensor

    def __init__(
        self,
        assignment: LongTensor,
        max_id: int | None = None,
        mode: Literal["sum", "mean", "max"] = "mean",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the representation.

        :param assignment: shape: (nnz, 2)
            The assignment between indices and tokens, in edge-list format.
            ``assignment[:, 0]`` denotes the indices for the representation,
            ``assignment[:, 1]`` the index of the token.
        :param max_id:
            The maximum ID (exclusively). Valid Ids reach from ``0`` to ``max_id-1``.
            Can be ``None`` to infer it from the assignment tensor.
        :param mode:
            The aggregation mode for :class:`~torch.nn.EmbeddingBag`.
        :param kwargs:
            Additional keyword-based parameters passed to :class:`~pykeen.nn.Representation`.
        """
        a_max_id, num_components = assignment.max(dim=0).values.tolist()
        # note: we use unique within _plain_forward anyway
        super().__init__(max_id=max_id or a_max_id + 1, unique=False, **kwargs)
        # sort by index
        idx = assignment[:, 0].argsort()
        assignment = assignment[idx].clone()
        # register assignment buffer *after* super init
        self.register_buffer(name="assignment", tensor=assignment)
        # flatten shape
        embedding_dim = 1
        for d in self.shape:
            embedding_dim *= d
        # set-up embedding bag
        self.embedding_bag = nn.EmbeddingBag(num_embeddings=num_components + 1, embedding_dim=embedding_dim, mode=mode)

    # docstr-coverage: inherited
    def _plain_forward(self, indices: LongTensor | None = None) -> FloatTensor:  # noqa: D102
        if indices is None:
            indices = unique_indices = inverse = torch.arange(self.max_id)
        else:
            unique_indices, inverse = indices.unique(return_inverse=True)
        # filter assignment
        mask = torch.isin(self.assignment[:, 0], test_elements=unique_indices)
        selection = self.assignment[mask]
        # set-up offsets & sub-indices
        sub_indices = selection[:, 1]
        # determine CSR-style offsets
        bag_index, bag_size = selection[:, 0].unique(return_counts=True)
        offsets = torch.zeros_like(unique_indices)
        mask = torch.isin(bag_index, test_elements=unique_indices, assume_unique=True)
        offsets[mask] = bag_size
        offsets = torch.cumsum(offsets, dim=0)[:-1]
        offsets = torch.cat([torch.zeros(1, dtype=offsets.dtype), offsets])
        return self.embedding_bag(sub_indices, offsets)[inverse].view(*indices.shape, *self.shape)

    @classmethod
    def from_iter(cls, xss: Iterable[Iterable[int]], **kwargs: Any) -> Self:
        """Instantiate from an iterable of indices.

        :param xss:
            An iterable over the indices,
            where each element is an iterable over the token indices for the given index.
        :param kwargs:
            Additional keyword-based parameters passed to :meth:`__init__`

        :return:
            A corresponding representation.
        """
        return cls(assignment=torch.as_tensor([(i, x) for i, xs in enumerate(xss) for x in xs]), **kwargs)
