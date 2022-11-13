# -*- coding: utf-8 -*-

"""Embedding weight initialization routines."""

import functools
import logging
import math
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn
import torch.nn.init
import torch_ppr.utils
from class_resolver import FunctionResolver, Hint, HintOrType, OptionalKwargs
from more_itertools import last
from torch.nn import functional

from .text import TextEncoder, text_encoder_resolver
from .utils import iter_matrix_power, safe_diagonal
from ..triples import CoreTriplesFactory, TriplesFactory
from ..typing import Initializer, MappedTriples, OneOrSequence
from ..utils import compose, get_edge_index, iter_weisfeiler_lehman, upgrade_to_sequence

__all__ = [
    "xavier_uniform_",
    "xavier_uniform_norm_",
    "xavier_normal_",
    "xavier_normal_norm_",
    "uniform_norm_",
    "uniform_norm_p1_",
    "normal_norm_",
    "init_phases",
    # Classes
    "PretrainedInitializer",
    "LabelBasedInitializer",
    "WeisfeilerLehmanInitializer",
    "RandomWalkPositionalEncodingInitializer",
    # Resolver
    "initializer_resolver",
]

logger = logging.getLogger(__name__)


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    r"""Initialize weights of the tensor similarly to Glorot/Xavier initialization.

    Proceed as if it was a linear layer with `fan_in` of zero, `fan_out` of `prod(tensor.shape[1:])` and Xavier uniform
    initialization is used, i.e. fill the weight of input `tensor` with values
    sampled from :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan_out}}}

    Example usage:

    >>> import torch, pykeen.nn.init
    >>> w = torch.empty(3, 5)
    >>> pykeen.nn.init.xavier_uniform_(w, gain=torch.nn.init.calculate_gain("relu"))

    .. seealso::
        :func:`torch.nn.init.xavier_uniform_`

    :param tensor:
        a tensor to initialize
    :param gain:
        an optional scaling factor, defaults to 1.0.

    :return:
        tensor with weights by this initializer.
    """
    fan_out = np.prod(tensor.shape[1:])
    std = gain * math.sqrt(2.0 / float(fan_out))
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    torch.nn.init.uniform_(tensor, -bound, bound)
    return tensor


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    r"""Initialize weights of the tensor similarly to Glorot/Xavier initialization.

    Proceed as if it was a linear layer with `fan_in` of zero, `fan_out` of `prod(tensor.shape[1:])` and Xavier Normal
    initialization is used, i.e. fill the weight of input `tensor` with values
    sampled from :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan_out}}}

    Example usage:

    >>> import torch, pykeen.nn.init
    >>> w = torch.empty(3, 5)
    >>> pykeen.nn.init.xavier_normal_(w, gain=torch.nn.init.calculate_gain("relu"))

    .. seealso::
        :func:`torch.nn.init.xavier_normal_`

    :param tensor:
        a tensor to initialize
    :param gain:
        an optional scaling factor, defaults to 1.0.

    :return:
        tensor with weights by this initializer.
    """
    fan_out = np.prod(tensor.shape[1:])
    std = gain * math.sqrt(2.0 / float(fan_out))
    return torch.nn.init.normal_(tensor, mean=0.0, std=std)


def init_phases(x: torch.Tensor) -> torch.Tensor:
    r"""
    Generate random phases between 0 and :math:`2\pi`.

    .. note::
        This method works on the canonical torch real representation of complex tensors, cf.
        https://pytorch.org/docs/stable/complex_numbers.html

    :param x:
        a tensor to initialize

    :return:
        tensor with weights set by this initializer

    .. seealso ::
        :func:`torch.view_as_real`
    """
    # backwards compatibility
    if x.shape[-1] != 2:
        new_shape = (*x.shape[:-1], -1, 2)
        logger.warning(
            f"The input tensor shape, {tuple(x.shape)}, does not comply with the canonical complex tensor shape, "
            f"(..., 2), cf. https://pytorch.org/docs/stable/complex_numbers.html. We'll try to reshape to {new_shape}",
        )
        x = x.view(*new_shape)
    phases = 2 * np.pi * torch.rand_like(torch.view_as_complex(x).real)
    return torch.view_as_real(torch.complex(real=phases.cos(), imag=phases.sin())).detach()


xavier_uniform_norm_ = compose(
    torch.nn.init.xavier_uniform_,
    functional.normalize,
    name="xavier_uniform_norm_",
)
xavier_normal_norm_ = compose(
    torch.nn.init.xavier_normal_,
    functional.normalize,
    name="xavier_normal_norm_",
)
uniform_norm_ = compose(
    torch.nn.init.uniform_,
    functional.normalize,
    name="uniform_norm_",
)
normal_norm_ = compose(
    torch.nn.init.normal_,
    functional.normalize,
    name="normal_norm_",
)
uniform_norm_p1_ = compose(
    torch.nn.init.uniform_,
    functools.partial(functional.normalize, p=1),
    name="uniform_norm_p1_",
)


def init_quaternions(
    x: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    Initialize quaternion.

    :param x: shape: (..., d, 4)
        the quaternions

    :raises ValueError:
        if the shape's last dimension is not 4.

    :return: shape: (..., d, 4)
        uniform quaternions
    """
    if x.ndim < 2 or x.shape[-1] != 4:
        raise ValueError(f"shape must be (..., 4) but is {x.shape}.")
    *shape, dim = x.shape[:-1]
    num_elements = math.prod(shape)
    # scaling factor
    s = 1.0 / math.sqrt(2 * num_elements)
    # modulus ~ Uniform[-s, s]
    modulus = 2 * s * torch.rand(num_elements, dim) - s
    # phase ~ Uniform[0, 2*pi]
    phase = 2 * math.pi * torch.rand(num_elements, dim)
    # real part
    real = (modulus * phase.cos()).unsqueeze(dim=-1)
    # purely imaginary quaternions unitary
    imag = torch.rand(num_elements, dim, 3)
    imag = functional.normalize(imag, p=2, dim=-1)
    imag = imag * (modulus * phase.sin()).unsqueeze(dim=-1)
    return torch.cat([real, imag], dim=-1)


class PretrainedInitializer:
    """
    Initialize tensor with pretrained weights.

    Example usage:

    .. code-block::

        import torch
        from pykeen.pipeline import pipeline
        from pykeen.nn.init import PretrainedInitializer

        # this is usually loaded from somewhere else
        # the shape must match, as well as the entity-to-id mapping
        pretrained_embedding_tensor = torch.rand(14, 128)

        result = pipeline(
            dataset="nations",
            model="transe",
            model_kwargs=dict(
                embedding_dim=pretrained_embedding_tensor.shape[-1],
                entity_initializer=PretrainedInitializer(tensor=pretrained_embedding_tensor),
            ),
        )
    """

    def __init__(self, tensor: torch.FloatTensor) -> None:
        """
        Initialize the initializer.

        :param tensor:
            the tensor of pretrained embeddings.
        """
        self.tensor = tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the tensor with the given tensor."""
        if x.shape != self.tensor.shape:
            raise ValueError(f"shape does not match: expected {self.tensor.shape} but got {x.shape}")
        return self.tensor.to(device=x.device, dtype=x.dtype)

    def as_embedding(self, **kwargs: Any):
        """Get a static embedding from this pre-trained initializer.

        :param kwargs: Keyword arguments to pass to :class:`pykeen.nn.representation.Embedding`
        :returns: An embedding
        :rtype: pykeen.nn.representation.Embedding
        """
        from .representation import Embedding

        max_id, *shape = self.tensor.shape
        return Embedding(max_id=max_id, shape=shape, initializer=self, trainable=False, **kwargs)


class LabelBasedInitializer(PretrainedInitializer):
    """
    An initializer using pretrained models from the `transformers` library to encode labels.

    Example Usage:

    Initialize entity representations as Transformer encodings of their labels. Afterwards,
    the parameters are detached from the labels, and trained on the KGE task without any
    further connection to the Transformer model.

    .. code-block :: python

        from pykeen.datasets import get_dataset
        from pykeen.nn.init import LabelBasedInitializer
        from pykeen.models import ERMLPE

        dataset = get_dataset(dataset="nations")
        model = ERMLPE(
            embedding_dim=768,  # for BERT base
            entity_initializer=LabelBasedInitializer.from_triples_factory(
                triples_factory=dataset.training,
                encoder="transformer",
            ),
        )
    """

    def __init__(
        self,
        labels: Sequence[str],
        encoder: HintOrType[TextEncoder] = None,
        encoder_kwargs: OptionalKwargs = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize the initializer.

        :param labels:
            the labels
        :param encoder:
            the text encoder to use, cf. `text_encoder_resolver`
        :param encoder_kwargs:
            additional keyword-based parameters passed to the encoder
        :param batch_size: >0
            the (maximum) batch size to use while encoding. If None, use `len(labels)`, i.e., only a single batch.
        """
        super().__init__(
            tensor=text_encoder_resolver.make(encoder, encoder_kwargs).encode_all(
                labels=labels,
                batch_size=batch_size,
            )
            # must be cloned if we want to do backprop
            .clone(),
        )

    @classmethod
    def from_triples_factory(
        cls,
        triples_factory: TriplesFactory,
        for_entities: bool = True,
        **kwargs,
    ) -> "LabelBasedInitializer":
        """
        Prepare a label-based initializer with labels from a triples factory.

        :param triples_factory:
            the triples factory
        :param for_entities:
            whether to create the initializer for entities (or relations)
        :param kwargs:
            additional keyword-based arguments passed to :func:`LabelBasedInitializer.__init__`
        :returns:
            A label-based initializer

        :raise ImportError:
            if the transformers library could not be imported
        """
        id_to_label = triples_factory.entity_id_to_label if for_entities else triples_factory.relation_id_to_label
        labels = [id_to_label[i] for i in sorted(id_to_label.keys())]
        return cls(
            labels=labels,
            **kwargs,
        )


class WeisfeilerLehmanInitializer(PretrainedInitializer):
    """An initializer based on an encoding of categorical colors from the Weisfeiler-Lehman algorithm."""

    def __init__(
        self,
        *,
        # the color initializer
        color_initializer: Hint[Initializer] = None,
        color_initializer_kwargs: OptionalKwargs = None,
        shape: OneOrSequence[int] = 32,
        # variants for the edge index
        edge_index: Optional[torch.LongTensor] = None,
        num_entities: Optional[int] = None,
        mapped_triples: Optional[torch.LongTensor] = None,
        triples_factory: Optional[CoreTriplesFactory] = None,
        # additional parameters for iter_weisfeiler_lehman
        **kwargs,
    ) -> None:
        """
        Initialize the initializer.

        :param color_initializer:
            the initializer for initialization color representations, or a hint thereof
        :param color_initializer_kwargs:
            additional keyword-based parameters for the color initializer
        :param shape:
            the shape to use for the color representations

        :param edge_index: shape: (2, m)
            the edge index
        :param num_entities:
            the number of entities. can be inferred
        :param mapped_triples: shape: (m, 3)
            the Id-based triples
        :param triples_factory:
            the triples factory

        :param kwargs:
            additional keyword-based parameters passed to :func:`pykeen.utils.iter_weisfeiler_lehman`
        """
        # normalize shape
        shape = upgrade_to_sequence(shape)
        # get coloring
        colors = last(
            iter_weisfeiler_lehman(
                edge_index=get_edge_index(
                    triples_factory=triples_factory, mapped_triples=mapped_triples, edge_index=edge_index
                ),
                num_nodes=num_entities,
                **kwargs,
            )
        )
        # make color initializer
        color_initializer = initializer_resolver.make(color_initializer, pos_kwargs=color_initializer_kwargs)
        # initialize color representations
        num_colors = colors.max().item() + 1
        # note: this could be a representation?
        color_representation = color_initializer(colors.new_empty(num_colors, *shape, dtype=torch.get_default_dtype()))
        # init entity representations according to the color
        super().__init__(tensor=color_representation[colors])


class RandomWalkPositionalEncodingInitializer(PretrainedInitializer):
    r"""
    Initialize nodes via random-walk positional encoding.

    The random walk positional encoding is given as

    .. math::
        \mathbf{x}_i = [\mathbf{R}_{i, i}, \mathbf{R}^{2}_{i, i}, \ldots, \mathbf{R}^{d}_{i, i}] \in \mathbb{R}^{d}

    where $\mathbf{R} := \mathbf{A}\mathbf{D}^{-1}$ is the random walk matrix, with
    $\mathbf{D} := \sum_i \mathbf{A}_{i, i}$.

    .. seealso::
        https://arxiv.org/abs/2110.07875
    """

    def __init__(
        self,
        *,
        triples_factory: Optional[CoreTriplesFactory] = None,
        mapped_triples: Optional[MappedTriples] = None,
        edge_index: Optional[torch.Tensor] = None,
        dim: int,
        num_entities: Optional[int] = None,
        space_dim: int = 0,
        skip_first_power: bool = True,
    ) -> None:
        """
        Initialize the positional encoding.

        One of `triples_factory`, `mapped_triples` or `edge_index` will be used.
        The preference order is:

        1. `triples_factory`
        2. `mapped_triples`
        3. `edge_index`

        :param triples_factory:
            the triples factory
        :param mapped_triples: shape: `(m, 3)`
            the mapped triples
        :param edge_index: shape: `(2, m)`
            the edge index
        :param dim:
            the dimensionality
        :param num_entities:
            the number of entities. If `None`, it will be inferred from `edge_index`
        :param space_dim:
            estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.
        :param skip_first_power:
            in most cases the adjacencies diagonal values will be zeros (since reflexive edges are not that common).
            This flag enables skipping the first matrix power.
        """
        edge_index = get_edge_index(
            triples_factory=triples_factory, mapped_triples=mapped_triples, edge_index=edge_index
        )
        # create random walk matrix
        rw = torch_ppr.utils.prepare_page_rank_adjacency(edge_index=edge_index, num_nodes=num_entities)
        # stack diagonal entries of powers of rw
        tensor = torch.stack(
            [
                (i ** (space_dim / 2.0)) * safe_diagonal(matrix=power)
                for i, power in enumerate(iter_matrix_power(matrix=rw, max_iter=dim), start=1)
                if not skip_first_power or i > 1
            ],
            dim=-1,
        )
        super().__init__(tensor=tensor)


initializer_resolver: FunctionResolver[Initializer] = FunctionResolver(
    [
        getattr(torch.nn.init, func)
        for func in dir(torch.nn.init)
        if not func.startswith("_") and func.endswith("_") and func not in {"xavier_normal_", "xavier_uniform_"}
    ],
    default=torch.nn.init.normal_,
)
for func in [
    xavier_normal_,
    xavier_uniform_,
    init_phases,
    init_quaternions,
    xavier_normal_norm_,
    xavier_uniform_norm_,
    normal_norm_,
    uniform_norm_,
]:
    initializer_resolver.register(func)
