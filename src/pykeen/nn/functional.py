# -*- coding: utf-8 -*-

"""Functional forms of interaction methods.

The functional forms always assume the general form of the interaction function, where head, relation and tail
representations are provided in shape (batch_size, num_heads, 1, 1, ``*``), (batch_size, 1, num_relations, 1, ``*``),
and (batch_size, 1, 1, num_tails, ``*``), and return a score tensor of shape
(batch_size, num_heads, num_relations, num_tails).
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple, Union

import numpy
import torch
from torch import nn

from .compute_kernel import _complex_native_complex
from .sim import KG2E_SIMILARITIES
from ..moves import irfft, rfft
from ..typing import GaussianDistribution
from ..utils import (
    broadcast_cat, clamp_norm, estimate_cost_of_sequence, extended_einsum, is_cudnn_error, negative_norm,
    negative_norm_of_sum, project_entity, tensor_product, tensor_sum, view_complex,
)

__all__ = [
    'complex_interaction',
    'conve_interaction',
    'convkb_interaction',
    'cross_e_interaction',
    'distmult_interaction',
    'ermlp_interaction',
    'ermlpe_interaction',
    'hole_interaction',
    'kg2e_interaction',
    'mure_interaction',
    'ntn_interaction',
    'pair_re_interaction',
    'proje_interaction',
    'rescal_interaction',
    'rotate_interaction',
    'simple_interaction',
    'structured_embedding_interaction',
    'transd_interaction',
    'transe_interaction',
    'transh_interaction',
    'transr_interaction',
    'tucker_interaction',
    'unstructured_model_interaction',
]


def _extract_sizes(
    h: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[int, int, int, int, int]:
    """Extract size dimensions from head/relation/tail representations."""
    num_heads, num_relations, num_tails = [xx.shape[i] for i, xx in enumerate((h, r, t), start=1)]
    d_e = h.shape[-1]
    d_r = r.shape[-1]
    return num_heads, num_relations, num_tails, d_e, d_r


def _apply_optional_bn_to_tensor(
    x: torch.FloatTensor,
    output_dropout: nn.Dropout,
    batch_norm: Optional[nn.BatchNorm1d] = None,
) -> torch.FloatTensor:
    """Apply optional batch normalization and dropout layer. Supports multiple batch dimensions."""
    if batch_norm is not None:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x = batch_norm(x)
        x = x.view(*shape)
    return output_dropout(x)


def _add_cuda_warning(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if not is_cudnn_error(e):
                raise e
            raise RuntimeError(
                '\nThis code crash might have been caused by a CUDA bug, see '
                'https://github.com/allenai/allennlp/issues/2888, '
                'which causes the code to crash during evaluation mode.\n'
                'To avoid this error, the batch size has to be reduced.',
            ) from e

    return wrapped


def complex_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    r"""Evaluate the ComplEx interaction function.

    .. math ::
        Re(\langle h, r, conj(t) \rangle)

    :param h: shape: (batch_size, num_heads, 1, 1, `2*dim`)
        The complex head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, 2*dim)
        The complex relation representations.
    :param t: shape: (batch_size, 1, 1, num_tails, 2*dim)
        The complex tail representations.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return _complex_native_complex(h, r, t)


@_add_cuda_warning
def conve_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    t_bias: torch.FloatTensor,
    input_channels: int,
    embedding_height: int,
    embedding_width: int,
    hr2d: nn.Module,
    hr1d: nn.Module,
) -> torch.FloatTensor:
    """Evaluate the ConvE interaction function.

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, dim)
        The relation representations.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.
    :param t_bias: shape: (batch_size, 1, 1, num_tails, 1)
        The tail entity bias.
    :param input_channels:
        The number of input channels.
    :param embedding_height:
        The height of the reshaped embedding.
    :param embedding_width:
        The width of the reshaped embedding.
    :param hr2d:
        The first module, transforming the 2D stacked head-relation "image".
    :param hr1d:
        The second module, transforming the 1D flattened output of the 2D module.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # repeat if necessary, and concat head and relation, batch_size', num_input_channels, 2*height, width
    # with batch_size' = batch_size * num_heads * num_relations
    x = broadcast_cat(
        [
            h.view(*h.shape[:-1], input_channels, embedding_height, embedding_width),
            r.view(*r.shape[:-1], input_channels, embedding_height, embedding_width),
        ],
        dim=-2,
    ).view(-1, input_channels, 2 * embedding_height, embedding_width)

    # batch_size', num_input_channels, 2*height, width
    x = hr2d(x)

    # batch_size', num_output_channels * (2 * height - kernel_height + 1) * (width - kernel_width + 1)
    x = x.view(-1, numpy.prod(x.shape[-3:]))
    x = hr1d(x)

    # reshape: (batch_size', embedding_dim) -> (b, h, r, 1, d)
    x = x.view(-1, h.shape[1], r.shape[2], 1, h.shape[-1])

    # For efficient calculation, each of the convolved [h, r] rows has only to be multiplied with one t row
    # output_shape: (batch_size, num_heads, num_relations, num_tails)
    t = t.transpose(-1, -2)
    x = (x @ t).squeeze(dim=-2)

    # add bias term
    return x + t_bias.squeeze(dim=-1)


def convkb_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    conv: nn.Conv2d,
    activation: nn.Module,
    hidden_dropout: nn.Dropout,
    linear: nn.Linear,
) -> torch.FloatTensor:
    r"""Evaluate the ConvKB interaction function.

    .. math::
        W_L drop(act(W_C \ast ([h; r; t]) + b_C)) + b_L

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, dim)
        The relation representations.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.
    :param conv:
        The 3x1 convolution.
    :param activation:
        The activation function.
    :param hidden_dropout:
        The dropout layer applied to the hidden activations.
    :param linear:
        The final linear layer.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # decompose convolution for faster computation in 1-n case
    num_filters = conv.weight.shape[0]
    assert conv.weight.shape == (num_filters, 1, 1, 3)

    # compute conv(stack(h, r, t))
    # prepare input shapes for broadcasting
    # (b, h, r, t, 1, d)
    h = h.unsqueeze(dim=-2)
    r = r.unsqueeze(dim=-2)
    t = t.unsqueeze(dim=-2)

    # conv.weight.shape = (C_out, C_in, kernel_size[0], kernel_size[1])
    # here, kernel_size = (1, 3), C_in = 1, C_out = num_filters
    # -> conv_head, conv_rel, conv_tail shapes: (num_filters,)
    # reshape to (1, 1, 1, 1, f, 1)
    conv_head, conv_rel, conv_tail, conv_bias = [
        c.view(1, 1, 1, 1, num_filters, 1)
        for c in list(conv.weight[:, 0, 0, :].t()) + [conv.bias]
    ]

    # convolve -> output.shape: (*, embedding_dim, num_filters)
    h = conv_head @ h
    r = conv_rel @ r
    t = conv_tail @ t

    x = tensor_sum(conv_bias, h, r, t)
    x = activation(x)

    # Apply dropout, cf. https://github.com/daiquocnguyen/ConvKB/blob/master/model.py#L54-L56
    x = hidden_dropout(x)

    # Linear layer for final scores; use flattened representations, shape: (b, h, r, t, d * f)
    x = x.view(*x.shape[:-2], -1)
    x = linear(x)
    return x.squeeze(dim=-1)


def distmult_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Evaluate the DistMult interaction function.

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, dim)
        The relation representations.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return tensor_product(h, r, t).sum(dim=-1)


def ermlp_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    hidden: nn.Linear,
    activation: nn.Module,
    final: nn.Linear,
) -> torch.FloatTensor:
    r"""Evaluate the ER-MLP interaction function.

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, dim)
        The relation representations.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.
    :param hidden:
        The first linear layer.
    :param activation:
        The activation function of the hidden layer.
    :param final:
        The second linear layer.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # same shape
    if h.shape == r.shape and h.shape == t.shape:
        return final(activation(
            hidden(torch.cat([h, r, t], dim=-1).view(-1, 3 * h.shape[-1]))),
        ).view(*h.shape[:-1])

    hidden_dim = hidden.weight.shape[0]
    # split, shape: (embedding_dim, hidden_dim)
    head_to_hidden, rel_to_hidden, tail_to_hidden = hidden.weight.t().split(h.shape[-1])
    bias = hidden.bias.view(1, 1, 1, 1, -1)
    h = h @ head_to_hidden.view(1, 1, 1, h.shape[-1], hidden_dim)
    r = r @ rel_to_hidden.view(1, 1, 1, r.shape[-1], hidden_dim)
    t = t @ tail_to_hidden.view(1, 1, 1, t.shape[-1], hidden_dim)
    return final(activation(tensor_sum(bias, h, r, t))).squeeze(dim=-1)


def ermlpe_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    mlp: nn.Module,
) -> torch.FloatTensor:
    r"""Evaluate the ER-MLPE interaction function.

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, dim)
        The relation representations.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.
    :param mlp:
        The MLP.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # repeat if necessary, and concat head and relation, (batch_size, num_heads, num_relations, 1, 2 * embedding_dim)
    x = broadcast_cat([h, r], dim=-1)

    # Predict t embedding, shape: (b, h, r, 1, d)
    shape = x.shape
    x = mlp(x.view(-1, shape[-1])).view(*shape[:-1], -1)

    # transpose t, (b, 1, 1, d, t)
    t = t.transpose(-2, -1)

    # dot product, (b, h, r, 1, t)
    return (x @ t).squeeze(dim=-2)


def hole_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:  # noqa: D102
    """Evaluate the HolE interaction function.

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, dim)
        The relation representations.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # composite: (b, h, 1, t, d)
    composite = circular_correlation(h, t)

    # transpose composite: (b, h, 1, d, t)
    composite = composite.transpose(-2, -1)

    # inner product with relation embedding
    return (r @ composite).squeeze(dim=-2)


def circular_correlation(
    a: torch.FloatTensor,
    b: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    Compute the circular correlation between to vectors.

    .. note ::
        The implementation uses FFT.

    :param a: shape: s_1
        The tensor with the first vectors.
    :param b:
        The tensor with the second vectors.

    :return:
        The circular correlation between the vectors.
    """
    # Circular correlation of entity embeddings
    a_fft = rfft(a, dim=-1)
    b_fft = rfft(b, dim=-1)
    # complex conjugate
    a_fft = torch.conj(a_fft)
    # Hadamard product in frequency domain
    p_fft = a_fft * b_fft
    # inverse real FFT
    return irfft(p_fft, n=a.shape[-1], dim=-1)


def kg2e_interaction(
    h_mean: torch.FloatTensor,
    h_var: torch.FloatTensor,
    r_mean: torch.FloatTensor,
    r_var: torch.FloatTensor,
    t_mean: torch.FloatTensor,
    t_var: torch.FloatTensor,
    similarity: str = "KL",
    exact: bool = True,
) -> torch.FloatTensor:
    """Evaluate the KG2E interaction function.

    :param h_mean: shape: (batch_size, num_heads, 1, 1, d)
        The head entity distribution mean.
    :param h_var: shape: (batch_size, num_heads, 1, 1, d)
        The head entity distribution variance.
    :param r_mean: shape: (batch_size, 1, num_relations, 1, d)
        The relation distribution mean.
    :param r_var: shape: (batch_size, 1, num_relations, 1, d)
        The relation distribution variance.
    :param t_mean: shape: (batch_size, 1, 1, num_tails, d)
        The tail entity distribution mean.
    :param t_var: shape: (batch_size, 1, 1, num_tails, d)
        The tail entity distribution variance.
    :param similarity:
        The similarity measures for gaussian distributions. From {"KL", "EL"}.
    :param exact:
        Whether to leave out constants to accelerate similarity computation.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return KG2E_SIMILARITIES[similarity](
        h=GaussianDistribution(mean=h_mean, diagonal_covariance=h_var),
        r=GaussianDistribution(mean=r_mean, diagonal_covariance=r_var),
        t=GaussianDistribution(mean=t_mean, diagonal_covariance=t_var),
        exact=exact,
    )


def ntn_interaction(
    h: torch.FloatTensor,
    t: torch.FloatTensor,
    w: torch.FloatTensor,
    vh: torch.FloatTensor,
    vt: torch.FloatTensor,
    b: torch.FloatTensor,
    u: torch.FloatTensor,
    activation: nn.Module,
) -> torch.FloatTensor:
    r"""Evaluate the NTN interaction function.

    .. math::

        f(h,r,t) = u_r^T act(h W_r t + V_r h + V_r' t + b_r)

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param w: shape: (batch_size, 1, num_relations, 1, k, dim, dim)
        The relation specific transformation matrix W_r.
    :param vh: shape: (batch_size, 1, num_relations, 1, k, dim)
        The head transformation matrix V_h.
    :param vt: shape: (batch_size, 1, num_relations, 1, k, dim)
        The tail transformation matrix V_h.
    :param b: shape: (batch_size, 1, num_relations, 1, k)
        The relation specific offset b_r.
    :param u: shape: (batch_size, 1, num_relations, 1, k)
        The relation specific final linear transformation b_r.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.
    :param activation:
        The activation function.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    x = activation(tensor_sum(
        extended_einsum("bhrtd,bhrtkde,bhrte->bhrtk", h, w, t),
        (vh @ h.unsqueeze(dim=-1)).squeeze(dim=-1),
        (vt @ t.unsqueeze(dim=-1)).squeeze(dim=-1),
        b,
    ))
    u = u.transpose(-2, -1)
    return (x @ u).squeeze(dim=-1)


def proje_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    d_e: torch.FloatTensor,
    d_r: torch.FloatTensor,
    b_c: torch.FloatTensor,
    b_p: torch.FloatTensor,
    activation: nn.Module,
) -> torch.FloatTensor:
    r"""Evaluate the ProjE interaction function.

    .. math::

        f(h, r, t) = g(t z(D_e h + D_r r + b_c) + b_p)

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, dim)
        The relation representations.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.
    :param d_e: shape: (dim,)
        Global entity projection.
    :param d_r: shape: (dim,)
        Global relation projection.
    :param b_c: shape: (dim,)
        Global combination bias.
    :param b_p: shape: (1,)
        Final score bias
    :param activation:
        The activation function.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    num_heads, num_relations, num_tails, dim, _ = _extract_sizes(h, r, t)
    # global projections
    h = h * d_e.view(1, 1, 1, 1, dim)
    r = r * d_r.view(1, 1, 1, 1, dim)
    # combination, shape: (b, h, r, 1, d)
    x = tensor_sum(h, r, b_c)
    x = activation(x)  # shape: (b, h, r, 1, d)
    # dot product with t, shape: (b, h, r, t)
    t = t.transpose(-2, -1)  # shape: (b, 1, 1, d, t)
    return (x @ t).squeeze(dim=-2) + b_p


def rescal_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Evaluate the RESCAL interaction function.

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, dim, dim)
        The relation representations.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return extended_einsum("bhrtd,bhrtde,bhrte->bhrt", h, r, t)


def rotate_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Evaluate the RotatE interaction function.

    :param h: shape: (batch_size, num_heads, 1, 1, 2*dim)
        The head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, 2*dim)
        The relation representations.
    :param t: shape: (batch_size, 1, 1, num_tails, 2*dim)
        The tail representations.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # r expresses a rotation in complex plane.
    h, r, t = [view_complex(x) for x in (h, r, t)]
    if estimate_cost_of_sequence(h.shape, r.shape) < estimate_cost_of_sequence(r.shape, t.shape):
        # rotate head by relation (=Hadamard product in complex space)
        h = h * r
    else:
        # rotate tail by inverse of relation
        # The inverse rotation is expressed by the complex conjugate of r.
        # The score is computed as the distance of the relation-rotated head to the tail.
        # Equivalently, we can rotate the tail by the inverse relation, and measure the distance to the head, i.e.
        # |h * r - t| = |h - conj(r) * t|
        t = t * torch.conj(r)

    # Workaround until https://github.com/pytorch/pytorch/issues/30704 is fixed
    return negative_norm(h - t, p=2, power_norm=False)


def simple_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    h_inv: torch.FloatTensor,
    r_inv: torch.FloatTensor,
    t_inv: torch.FloatTensor,
    clamp: Optional[Tuple[float, float]] = None,
) -> torch.FloatTensor:
    """Evaluate the SimplE interaction function.

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, dim, dim)
        The relation representations.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.
    :param h_inv: shape: (batch_size, num_heads, 1, 1, dim)
        The inverse head representations.
    :param r_inv: shape: (batch_size, 1, num_relations, 1, dim, dim)
        The relation representations.
    :param t_inv: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.
    :param clamp:
        Clamp the scores to the given range.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    scores = 0.5 * (distmult_interaction(h=h, r=r, t=t) + distmult_interaction(h=h_inv, r=r_inv, t=t_inv))
    # Note: In the code in their repository, the score is clamped to [-20, 20].
    #       That is not mentioned in the paper, so it is made optional here.
    if clamp:
        min_, max_ = clamp
        scores = scores.clamp(min=min_, max=max_)
    return scores


def structured_embedding_interaction(
    h: torch.FloatTensor,
    r_h: torch.FloatTensor,
    r_t: torch.FloatTensor,
    t: torch.FloatTensor,
    p: int,
    power_norm: bool = False,
) -> torch.FloatTensor:
    r"""Evaluate the Structured Embedding interaction function.

    .. math ::
        f(h, r, t) = -\|R_h h - R_t t\|

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param r_h: shape: (batch_size, 1, num_relations, 1, rel_dim, dim)
        The relation-specific head projection.
    :param r_t: shape: (batch_size, 1, num_relations, 1, rel_dim, dim)
        The relation-specific tail projection.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.
    :param p:
        The p for the norm. cf. torch.norm.
    :param power_norm:
        Whether to return the powered norm.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return negative_norm(
        (r_h @ h.unsqueeze(dim=-1) - r_t @ t.unsqueeze(dim=-1)).squeeze(dim=-1),
        p=p,
        power_norm=power_norm,
    )


def transd_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    h_p: torch.FloatTensor,
    r_p: torch.FloatTensor,
    t_p: torch.FloatTensor,
    p: int,
    power_norm: bool = False,
) -> torch.FloatTensor:
    """Evaluate the TransD interaction function.

    :param h: shape: (batch_size, num_heads, 1, 1, d_e)
        The head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, d_r)
        The relation representations.
    :param t: shape: (batch_size, 1, 1, num_tails, d_e)
        The tail representations.
    :param h_p: shape: (batch_size, num_heads, 1, 1, d_e)
        The head projections.
    :param r_p: shape: (batch_size, 1, num_relations, 1, d_r)
        The relation projections.
    :param t_p: shape: (batch_size, 1, 1, num_tails, d_e)
        The tail projections.
    :param p:
        The parameter p for selecting the norm.
    :param power_norm:
        Whether to return the powered norm instead.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # Project entities
    h_bot = project_entity(
        e=h,
        e_p=h_p,
        r_p=r_p,
    )
    t_bot = project_entity(
        e=t,
        e_p=t_p,
        r_p=r_p,
    )
    return negative_norm_of_sum(h_bot, r, -t_bot, p=p, power_norm=power_norm)


def transe_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    p: Union[int, str] = 2,
    power_norm: bool = False,
) -> torch.FloatTensor:
    """Evaluate the TransE interaction function.

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, dim)
        The relation representations.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.
    :param p:
        The p for the norm.
    :param power_norm:
        Whether to return the powered norm.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return negative_norm_of_sum(h, r, -t, p=p, power_norm=power_norm)


def transh_interaction(
    h: torch.FloatTensor,
    w_r: torch.FloatTensor,
    d_r: torch.FloatTensor,
    t: torch.FloatTensor,
    p: int,
    power_norm: bool = False,
) -> torch.FloatTensor:
    """Evaluate the DistMult interaction function.

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param w_r: shape: (batch_size, 1, num_relations, 1, dim)
        The relation normal vector representations.
    :param d_r: shape: (batch_size, 1, num_relations, 1, dim)
        The relation difference vector representations.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.
    :param p:
        The p for the norm. cf. torch.norm.
    :param power_norm:
        Whether to return $|x-y|_p^p$.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return negative_norm_of_sum(
        # h projection to hyperplane
        h,
        -(h * w_r).sum(dim=-1, keepdims=True) * w_r,
        # r
        d_r,
        # -t projection to hyperplane
        -t,
        (t * w_r).sum(dim=-1, keepdims=True) * w_r,
        p=p,
        power_norm=power_norm,
    )


def transr_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    m_r: torch.FloatTensor,
    p: int,
    power_norm: bool = True,
) -> torch.FloatTensor:
    """Evaluate the TransR interaction function.

    :param h: shape: (batch_size, num_heads, 1, 1, d_e)
        Head embeddings.
    :param r: shape: (batch_size, 1, num_relations, 1, d_r)
        Relation embeddings.
    :param m_r: shape: (batch_size, 1, num_relations, 1, d_e, d_r)
        The relation specific linear transformations.
    :param t: shape: (batch_size, 1, 1, num_tails, d_e)
        Tail embeddings.
    :param p:
        The parameter p for selecting the norm.
    :param power_norm:
        Whether to return the powered norm instead.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # project to relation specific subspace and ensure constraints
    h_bot = clamp_norm((h.unsqueeze(dim=-2) @ m_r), p=2, dim=-1, maxnorm=1.).squeeze(dim=-2)
    t_bot = clamp_norm((t.unsqueeze(dim=-2) @ m_r), p=2, dim=-1, maxnorm=1.).squeeze(dim=-2)
    return negative_norm_of_sum(h_bot, r, -t_bot, p=p, power_norm=power_norm)


def tucker_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    core_tensor: torch.FloatTensor,
    do_h: nn.Dropout,
    do_r: nn.Dropout,
    do_hr: nn.Dropout,
    bn_h: Optional[nn.BatchNorm1d],
    bn_hr: Optional[nn.BatchNorm1d],
) -> torch.FloatTensor:
    r"""Evaluate the TuckEr interaction function.

    Compute scoring function W x_1 h x_2 r x_3 t as in the official implementation, i.e. as

    .. math ::

        DO_{hr}(BN_{hr}(DO_h(BN_h(h)) x_1 DO_r(W x_2 r))) x_3 t

    where BN denotes BatchNorm and DO denotes Dropout

    :param h: shape: (batch_size, num_heads, 1, 1, d_e)
        The head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, d_r)
        The relation representations.
    :param t: shape: (batch_size, 1, 1, num_tails, d_e)
        The tail representations.
    :param core_tensor: shape: (d_e, d_r, d_e)
        The core tensor.
    :param do_h:
        The dropout layer for the head representations.
    :param do_r:
        The first hidden dropout.
    :param do_hr:
        The second hidden dropout.
    :param bn_h:
        The first batch normalization layer.
    :param bn_hr:
        The second batch normalization layer.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return extended_einsum(
        # x_3 contraction
        "bhrtk,bhrtk->bhrt",
        _apply_optional_bn_to_tensor(
            x=extended_einsum(
                # x_1 contraction
                "bhrtik,bhrti->bhrtk",
                _apply_optional_bn_to_tensor(
                    x=extended_einsum(
                        # x_2 contraction
                        "ijk,bhrtj->bhrtik",
                        core_tensor,
                        r,
                    ),
                    output_dropout=do_r,
                ),
                _apply_optional_bn_to_tensor(
                    x=h,
                    batch_norm=bn_h,
                    output_dropout=do_h,
                )),
            batch_norm=bn_hr,
            output_dropout=do_hr,
        ),
        t,
    )


def mure_interaction(
    h: torch.FloatTensor,
    b_h: torch.FloatTensor,
    r_vec: torch.FloatTensor,
    r_mat: torch.FloatTensor,
    t: torch.FloatTensor,
    b_t: torch.FloatTensor,
    p: Union[int, float, str] = 2,
    power_norm: bool = False,
) -> torch.FloatTensor:
    r"""Evaluate the MuRE interaction function from [balazevic2019b]_.

    .. math ::
        -\|Rh + r - t\| + b_h + b_t

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param b_h: shape: (batch_size, num_heads, 1, 1)
        The head entity bias.
    :param r_vec: shape: (batch_size, 1, num_relations, 1, dim)
        The relation vector.
    :param r_mat: shape: (batch_size, 1, num_relations, 1, dim,)
        The diagonal relation matrix.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.
    :param b_t: shape: (batch_size, 1, 1, num_tails)
        The tail entity bias.
    :param p:
        The parameter p for selecting the norm, cf. torch.norm.
    :param power_norm:
        Whether to return the powered norm instead.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return negative_norm_of_sum(
        h * r_mat,
        r_vec,
        -t,
        p=p,
        power_norm=power_norm,
    ) + b_h + b_t


def unstructured_model_interaction(
    h: torch.FloatTensor,
    t: torch.FloatTensor,
    p: int,
    power_norm: bool = True,
) -> torch.FloatTensor:
    """Evaluate the SimplE interaction function.

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.
    :param p:
        The parameter p for selecting the norm.
    :param power_norm:
        Whether to return the powered norm instead.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return negative_norm(h - t, p=p, power_norm=power_norm)


def pair_re_interaction(
    h: torch.FloatTensor,
    t: torch.FloatTensor,
    r_h: torch.FloatTensor,
    r_t: torch.FloatTensor,
    p: Union[int, str] = 2,
    power_norm: bool = True,
) -> torch.FloatTensor:
    r"""Evaluate the PairRE interaction function.

    .. math ::
        -\|h \odot r_h - t \odot r_t \|

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.
    :param r_h: shape: (batch_size, 1, num_relations, 1, dim)
        The head part of the relation representations.
    :param r_t: shape: (batch_size, 1, num_relations, 1, dim)
        The tail part of the relation representations.
    :param p:
        The parameter p for selecting the norm.
    :param power_norm:
        Whether to return the powered norm instead.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return negative_norm_of_sum(
        h * r_h,
        -t * r_t,
        p=p,
        power_norm=power_norm,
    )


def _rotate_quaternion(qa: torch.FloatTensor, qb: torch.FloatTensor) -> torch.FloatTensor:
    # Rotate (=Hamilton product in quaternion space).
    return torch.cat(
        [
            qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3],
            qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2],
            qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1],
            qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0],
        ],
        dim=-1,
    )


def _split_quaternion(x: torch.FloatTensor) -> torch.FloatTensor:
    return torch.chunk(x, chunks=4, dim=-1)


def quat_e_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
):
    """Evaluate the interaction function of QuatE for given embeddings.

    The embeddings have to be in a broadcastable shape.

    .. note ::
        dim has to be divisible by 4.

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, dim)
        The head representations.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.

    :return: shape: (...)
        The scores.
    """
    return -(
        # Rotation in quaternion space
        _rotate_quaternion(
            _split_quaternion(h),
            _split_quaternion(r),
        ) * t
    ).sum(dim=-1)


def cross_e_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    c_r: torch.FloatTensor,
    t: torch.FloatTensor,
    bias: torch.FloatTensor,
    activation: nn.Module,
    dropout: Optional[nn.Dropout] = None,
) -> torch.FloatTensor:
    r"""
    Evaluate the interaction function of CrossE for the given representations from [zhang2019b]_.

    .. math ::
        Dropout(Activation(c_r \odot h + c_r \odot h \odot r + b))^T t)

    .. note ::
        The representations have to be in a broadcastable shape.

    .. note ::
        The CrossE paper described an additional sigmoid activation as part of the interaction function. Since using a
        log-likelihood loss can cause numerical problems (due to explicitly calling sigmoid before log), we do not
        apply this in our implementation but rather opt for the numerically stable variant. However, the model itself
        has an option ``predict_with_sigmoid``, which can be used to enforce application of sigmoid during inference.
        This can also have an impact of rank-based evaluation, since limited numerical precision can lead to exactly
        equal scores for multiple choices. The definition of a rank is not unambiguous in such case, and there exist
        multiple competing variants how to break the ties. More information on this can be found in the documentation of
        rank-based evaluation.

    :param h: shape: (batch_size, num_heads, 1, 1, dim)
        The head representations.
    :param r: shape: (batch_size, 1, num_relations, 1, dim)
        The relation representations.
    :param c_r: shape: (batch_size, 1, num_relations, 1, dim)
        The relation-specific interaction vector.
    :param t: shape: (batch_size, 1, 1, num_tails, dim)
        The tail representations.
    :param bias: shape: (1, 1, 1, 1, dim)
        The combination bias.
    :param activation:
        The combination activation. Should be :class:`torch.nn.Tanh` for consistency with the CrossE paper.
    :param dropout:
        Dropout applied after the combination.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.

    .. seealso:: https://github.com/wencolani/CrossE
    """
    # head interaction
    h = c_r * h
    # relation interaction (notice that h has been updated)
    r = h * r
    # combination
    x = activation(h + r + bias)
    if dropout is not None:
        x = dropout(x)
    # similarity
    return (x * t).sum(dim=-1)
