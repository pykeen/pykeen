# -*- coding: utf-8 -*-

"""Implementation of combinations for the :class:`pykeen.models.LiteralModel`."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from class_resolver import ClassResolver, Hint, HintOrType, OptionalKwargs
from class_resolver.contrib.torch import activation_resolver, aggregation_resolver
from torch import nn

from ..utils import ExtraReprMixin, combine_complex, split_complex

__all__ = [
    "Combination",
    # Concrete classes
    "ComplexSeparatedCombination",
    "ConcatCombination",
    "ConcatAggregationCombination",
    "ConcatProjectionCombination",
    "GatedCombination",
]

logger = logging.getLogger(__name__)


class Combination(nn.Module, ExtraReprMixin, ABC):
    """Base class for combinations."""

    @abstractmethod
    def forward(self, xs: Sequence[torch.FloatTensor]) -> torch.FloatTensor:
        """
        Combine a sequence of individual representations.

        :param xs: shape: `(*batch_dims, *input_dims_i)`
            the individual representations

        :return: shape: `(*batch_dims, *output_dims)`
            a combined representation
        """
        raise NotImplementedError

    def output_shape(self, input_shapes: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
        """
        Calculate the output shape for the given input shapes.

        .. note ::
            this method runs a single forward pass if no symbolic computation is available.

        :param input_shapes:
            the input shapes without the batch dimensions

        :return:
            the output shape
        """
        logger.warning("No symbolic computation of output shape.")
        return self(xs=[torch.empty(size=shape) for shape in input_shapes]).shape


class ConcatCombination(Combination):
    """Combine representation by concatenation."""

    def __init__(self, dim: int = -1) -> None:
        """
        Initialize the combination.

        :param dim:
            the concatenation dimension
        """
        super().__init__()
        self.dim = dim

    # docstr-coverage: inherited
    def forward(self, xs: Sequence[torch.FloatTensor]) -> torch.FloatTensor:  # noqa: D102
        return torch.cat(xs, dim=self.dim)

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"dim={self.dim}"


class ConcatProjectionCombination(ConcatCombination):
    """Combine representations by concatenation follow by a linear projection and activation."""

    def __init__(
        self,
        input_dims: Sequence[int],
        output_dim: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0,
        activation: HintOrType[nn.Module] = nn.Identity,
        activation_kwargs: OptionalKwargs = None,
    ) -> None:
        """
        Initialize the combination.

        :param input_dims:
            the input dimensions
        :param output_dim:
            the output dimension. Defaults to the first input dimension
        :param bias:
            whether to add a bias term in between the linear projection and the activation
        :param dropout:
            dropout to use before the activation
        :param activation:
            the activation, or a hint thereof
        :param activation_kwargs:
            additional keyword-based parameters used to instantiate the activation

        :raises ValueError:
            if `input_dims` is empty
        """
        super().__init__()
        if not input_dims:
            raise ValueError("Cannot provide empty input dimensions")
        output_dim = output_dim or input_dims[0]
        self.projection = nn.Sequential(
            nn.Linear(sum(input_dims), output_dim, bias=bias),
            nn.Dropout(dropout),
            activation_resolver.make(activation, activation_kwargs),
        )

    # docstr-coverage: inherited
    def forward(self, xs: Sequence[torch.FloatTensor]) -> torch.FloatTensor:  # noqa: D102
        return self.projection(super().forward(xs))


class ConcatAggregationCombination(ConcatCombination):
    """Combine representation by concatenation followed by an aggregation along the same axis."""

    def __init__(
        self,
        aggregation: Hint[Callable[[torch.FloatTensor], torch.FloatTensor]] = None,
        dim: int = -1,
    ) -> None:
        """
        Initialize the combination.

        :param aggregation:
            the aggregation, or a hint thereof, cf. :data:`class_resolver.contrib.torch.aggregation_resolver`
        :param dim:
            the concatenation and reduction dimension.
        """
        super().__init__(dim=dim)
        self.dim = dim
        self.aggregation = aggregation_resolver.make(aggregation)

    # docstr-coverage: inherited
    def forward(self, xs: Sequence[torch.FloatTensor]) -> torch.FloatTensor:  # noqa: D102
        return self.aggregation(super().forward(xs=xs), dim=self.dim)

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"aggregation={self.aggregation}"


class ComplexSeparatedCombination(Combination):
    """A combination for mixed complex & real representations."""

    def __init__(
        self,
        combination: HintOrType[Combination] = None,
        combination_kwargs: OptionalKwargs = None,
        imag_combination: HintOrType[Combination] = None,
        imag_combination_kwargs: OptionalKwargs = None,
    ):
        """
        Initialize the combination.

        .. note ::
            if non-instantiated combinations are passed, separate instances will be created for real and imaginary parts

        :param combination:
            the real combination, or a hint thereof
        :param combination_kwargs:
            keyword-based parameters for the real combination
        :param imag_combination:
            the imaginary combination, or a hint thereof. If None, use combination for both.
        :param imag_combination_kwargs:
            keyword-based parameters for the imaginary combination; only used if imag_combination is not None
        """
        super().__init__()
        # input normalization
        if imag_combination is None:
            imag_combination, imag_combination_kwargs = combination, combination_kwargs
        # instantiate separate combinations
        self.real_combination = combination_resolver.make(combination, combination_kwargs)
        self.imag_combination = combination_resolver.make(imag_combination, imag_combination_kwargs)

    # docstr-coverage: inherited
    def forward(self, xs: Sequence[torch.FloatTensor]) -> torch.FloatTensor:  # noqa: D102
        if not any(x.is_complex() for x in xs):
            raise ValueError(
                f"{self.__class__} is a combination for complex representations, but none of the inputs was of "
                f"complex data type."
            )
        # split complex; repeat real
        xs_real, xs_imag = list(zip(*(split_complex(x) if x.is_complex() else (x, x) for x in xs)))
        # separately combine real and imaginary parts
        x_re = self.real_combination(xs_real)
        x_im = self.imag_combination(xs_imag)
        # combine
        return combine_complex(x_re=x_re, x_im=x_im)

    # docstr-coverage: inherited
    def output_shape(self, input_shapes: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:  # noqa: D102
        # symbolic output to avoid dtype issue
        # we only need to consider real part here
        return self.real_combination.output_shape(input_shapes=input_shapes)


class GatedCombination(Combination):
    r"""A module that implements a gated linear transformation for the combination of entities and literals.

    Compared to the other Combinations, this combination makes use of a gating mechanism commonly found in RNNs.
    The main goal of this gating mechanism is to learn which parts of the additional literal information is
    useful or not and act accordingly, by incorporating them into the new combined embedding or discarding them.

    For given entity representation $\mathbf{x}_e \in \mathbb{R}^{d_e}$ and literal representation
    $\mathbf{x}_l \in \mathbb{R}^{d_l}$, the module calculates

    .. math ::

        z = f_{gate}(\mathbf{W}_e x_e + \mathbf{W}_l x_l + \mathbf{b})
        h = f_{hidden}(\mathbf{W} [x_e; x_l])
        y = Dropout(z \odot h + (1 - z) \odot x)

    where $\mathbf{W}_e \in \mathbb{R}^{d_e \times d_e}$,$\mathbf{W}_l \in \mathbb{R}^{d_l \times d_e}$,
    $\mathbf{W} \in \mathbb{R}^{(d_e + d_l) \ times d_e}$, and $\mathbf{b} \in \mathbb{R}^{d_e}$ are trainable
    parameters, $f_{gate}$ and $f_{hidden}$ are activation functions, defaulting to sigmoid and tanh, $\odot$ denotes
    the element-wise multiplication, and $[x_e; x_l]$ the concatenation operation.

    .. note ::

        We can alternatively express the gate

        .. math ::

            z = f_{gate}(\mathbf{W}_e x_e + \mathbf{W}_l x_l + \mathbf{b})

        as

        .. math ::

            z = f_{gate}(\mathbf{W}_{el} [x_e; x_l] + \mathbf{b})

        with $\mathbf{W}_{el} \in \mathbb{R}^{(d_e + d_l) \times d_e}$.

    Implementation based on https://github.com/SmartDataAnalytics/LiteralE/blob/master/model.py Gate class.
    """

    def __init__(
        self,
        entity_dim: int = 32,
        literal_dim: Optional[int] = None,
        input_dropout: float = 0.0,
        gate_activation: HintOrType[nn.Module] = nn.Sigmoid,
        gate_activation_kwargs: Optional[Mapping[str, Any]] = None,
        hidden_activation: HintOrType[nn.Module] = nn.Tanh,
        hidden_activation_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Instantiate the module.

        :param entity_dim:
            the dimension of the entity representations.
        :param literal_dim:
            the dimension of the literals; defaults to entity_dim
        :param input_dropout:
            the dropout to use
        :param gate_activation:
            the activation to use on the gate, or a hint thereof
        :param gate_activation_kwargs:
            the keyword arguments to be used to instantiate the `gate_activation` if
            a class or name is given instead of a pre-instantiated activation module
        :param hidden_activation:
            the activation to use in the hidden layer, or a hint thereof
        :param hidden_activation_kwargs:
            the keyword arguments to be used to instantiate the hidden activation if
            a class or name is given instead of a pre-instantiated activation module
        """
        super().__init__()
        literal_dim = literal_dim or entity_dim
        self.dropout = nn.Dropout(input_dropout)
        # the gate
        self.gate = ConcatProjectionCombination(
            input_dims=[entity_dim, literal_dim],
            output_dim=entity_dim,
            bias=True,
            activation=gate_activation,
            activation_kwargs=gate_activation_kwargs,
        )
        # the combination
        self.combination = ConcatProjectionCombination(
            input_dims=[entity_dim, literal_dim],
            output_dim=entity_dim,
            bias=True,
            activation=hidden_activation,
            activation_kwargs=hidden_activation_kwargs,
        )

    # docstr-coverage: inherited
    def forward(self, xs: Sequence[torch.FloatTensor]) -> torch.FloatTensor:  # noqa: D102
        assert len(xs) == 2
        z = self.gate(xs)
        h = self.combination(xs)
        return self.dropout(z * h + (1 - z) * xs[0])


combination_resolver: ClassResolver[Combination] = ClassResolver.from_subclasses(
    base=Combination,
    default=ConcatCombination,
)
