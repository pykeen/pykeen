# -*- coding: utf-8 -*-

"""Various edge weighting implementations for R-GCN."""

from abc import abstractmethod
from typing import ClassVar, Optional, Union

import torch
from class_resolver import ClassResolver
from torch import nn

from ..utils import einsum

try:
    import torch_scatter
except ImportError:
    torch_scatter = None

__all__ = [
    "EdgeWeighting",
    "InverseInDegreeEdgeWeighting",
    "InverseOutDegreeEdgeWeighting",
    "SymmetricEdgeWeighting",
    "AttentionEdgeWeighting",
    "edge_weight_resolver",
]


def softmax(
    src: torch.Tensor,
    index: torch.LongTensor,
    num_nodes: Union[None, int, torch.Tensor] = None,
    dim: int = 0,
) -> torch.Tensor:
    r"""
    Compute a sparsely evaluated softmax.

    Given a value tensor :attr:`src`, this function first groups the values
    along the given dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    :param src:
        The source tensor.
    :param index:
        The indices of elements for applying the softmax.
    :param num_nodes:
        The number of nodes, i.e., :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    :param dim:
        The dimension along which to compute the softmax.

    :returns:
        The softmax-ed tensor.

    :raises ImportError: if :mod:`torch_scatter` is not installed
    """
    if torch_scatter is None:
        raise ImportError(
            "torch-scatter is not installed, attention aggregation won't work. "
            "Install it here: https://github.com/rusty1s/pytorch_scatter",
        )
    num_nodes = num_nodes or index.max() + 1
    out = src.transpose(dim, 0)
    out = out - torch_scatter.scatter_max(out, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / torch_scatter.scatter_add(out, index, dim=0, dim_size=num_nodes)[index].clamp_min(1.0e-16)
    return out.transpose(0, dim)


class EdgeWeighting(nn.Module):
    """Base class for edge weightings."""

    #: whether the edge weighting needs access to the message
    needs_message: ClassVar[bool] = False

    def __init__(self, **kwargs):
        """
        Initialize the module.

        :param kwargs:
            ignored keyword-based parameters.
        """
        # stub init to enable arbitrary arguments in subclasses
        super().__init__()

    @abstractmethod
    def forward(
        self,
        source: torch.LongTensor,
        target: torch.LongTensor,
        message: Optional[torch.FloatTensor] = None,
        x_e: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """Compute edge weights.

        :param source: shape: (num_edges,)
                The source indices.
        :param target: shape: (num_edges,)
            The target indices.
        :param message: shape (num_edges, dim)
            Actual messages to weight
        :param x_e: shape (num_nodes, dim)
            Node states up to the weighting point

        :return: shape: (num_edges, dim)
             Messages weighted with the edge weights.
        """
        raise NotImplementedError


def _inverse_frequency_weighting(idx: torch.LongTensor) -> torch.FloatTensor:
    """Calculate inverse relative frequency weighting."""
    # Calculate in-degree, i.e. number of incoming edges
    inv, cnt = torch.unique(idx, return_counts=True, return_inverse=True)[1:]
    return cnt[inv].float().reciprocal()


class InverseInDegreeEdgeWeighting(EdgeWeighting):
    """Normalize messages by inverse in-degree."""

    # docstr-coverage: inherited
    def forward(
        self,
        source: torch.LongTensor,
        target: torch.LongTensor,
        message: Optional[torch.FloatTensor] = None,
        x_e: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        weight = _inverse_frequency_weighting(idx=target)
        if message is not None:
            return message * weight.unsqueeze(dim=-1)
        else:
            return weight


class InverseOutDegreeEdgeWeighting(EdgeWeighting):
    """Normalize messages by inverse out-degree."""

    # docstr-coverage: inherited
    def forward(
        self,
        source: torch.LongTensor,
        target: torch.LongTensor,
        message: Optional[torch.FloatTensor] = None,
        x_e: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        weight = _inverse_frequency_weighting(idx=source)
        if message is not None:
            return message * weight.unsqueeze(dim=-1)
        else:
            return weight


class SymmetricEdgeWeighting(EdgeWeighting):
    """Normalize messages by product of inverse sqrt of in-degree and out-degree."""

    # docstr-coverage: inherited
    def forward(
        self,
        source: torch.LongTensor,
        target: torch.LongTensor,
        message: Optional[torch.FloatTensor] = None,
        x_e: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        weight = (_inverse_frequency_weighting(idx=source) * _inverse_frequency_weighting(idx=target)).sqrt()
        if message is not None:
            return message * weight.unsqueeze(dim=-1)
        else:
            # backward compatibility with RGCN
            return weight


class AttentionEdgeWeighting(EdgeWeighting):
    """Message weighting by attention."""

    needs_message = True

    def __init__(
        self,
        message_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize the module.

        :param message_dim: >0
            the message dimension. has to be divisible by num_heads
            .. todo:: change to multiplicative instead of divisive to make this easier to use
        :param num_heads: >0
            the number of attention heads
        :param dropout:
            the attention dropout
        :raises ValueError: If ``message_dim`` is not divisible by ``num_heads``
        """
        super().__init__()
        if 0 != message_dim % num_heads:
            raise ValueError(f"output_dim={message_dim} must be divisible by num_heads={num_heads}!")
        self.num_heads = num_heads
        self.weight = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(num_heads, 2 * message_dim // num_heads)))
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.attention_dim = message_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    # docstr-coverage: inherited
    def forward(
        self,
        source: torch.LongTensor,
        target: torch.LongTensor,
        message: Optional[torch.FloatTensor] = None,
        x_e: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        if message is None or x_e is None:
            raise ValueError(f"{self.__class__.__name__} requires message and x_e.")

        # view for heads
        message_ = message.view(message.shape[0], self.num_heads, -1)
        # compute attention coefficients, shape: (num_edges, num_heads)
        alpha = self.activation(
            einsum(
                "ihd,hd->ih",
                torch.cat(
                    [
                        message_,
                        x_e[target].view(target.shape[0], self.num_heads, -1),
                    ],
                    dim=-1,
                ),
                self.weight,
            )
        )
        # TODO we can use scatter_softmax from torch_scatter directly, kept this if we can rewrite it w/o scatter
        alpha = softmax(alpha, index=target, num_nodes=x_e.shape[0], dim=0)
        alpha = self.dropout(alpha)
        return (message_ * alpha.view(-1, self.num_heads, 1)).view(-1, self.num_heads * self.attention_dim)


edge_weight_resolver: ClassResolver[EdgeWeighting] = ClassResolver.from_subclasses(
    base=EdgeWeighting, default=SymmetricEdgeWeighting
)
