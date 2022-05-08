# -*- coding: utf-8 -*-

"""Functional forms of interaction methods.

These implementations allow for an arbitrary number of batch dimensions,
as well as broadcasting and thus naturally support slicing and 1:n scoring.
"""

from __future__ import annotations

import functools
from typing import Optional, Sequence, Tuple, Union

import numpy
import torch
from torch import broadcast_tensors, nn

from .compute_kernel import batched_dot
from .sim import KG2E_SIMILARITIES
from ..moves import irfft, rfft
from ..typing import GaussianDistribution, Sign
from ..utils import (
    boxe_kg_arity_position_score,
    clamp_norm,
    compute_box,
    ensure_complex,
    estimate_cost_of_sequence,
    is_cudnn_error,
    make_ones_like,
    negative_norm,
    negative_norm_of_sum,
    project_entity,
    tensor_product,
    tensor_sum,
)

__all__ = [
    "auto_sf_interaction",
    "boxe_interaction",
    "complex_interaction",
    "conve_interaction",
    "convkb_interaction",
    "cp_interaction",
    "cross_e_interaction",
    "dist_ma_interaction",
    "distmult_interaction",
    "ermlp_interaction",
    "ermlpe_interaction",
    "hole_interaction",
    "kg2e_interaction",
    "multilinear_tucker_interaction",
    "mure_interaction",
    "ntn_interaction",
    "pair_re_interaction",
    "proje_interaction",
    "rescal_interaction",
    "rotate_interaction",
    "simple_interaction",
    "se_interaction",
    "transd_interaction",
    "transe_interaction",
    "transf_interaction",
    "transh_interaction",
    "transr_interaction",
    "transformer_interaction",
    "triple_re_interaction",
    "tucker_interaction",
    "um_interaction",
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
    # docstr-coverage: excused `wrapped`
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if not is_cudnn_error(e):
                raise e
            raise RuntimeError(
                "\nThis code crash might have been caused by a CUDA bug, see "
                "https://github.com/allenai/allennlp/issues/2888, "
                "which causes the code to crash during evaluation mode.\n"
                "To avoid this error, the batch size has to be reduced.",
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

    .. note::
        this method expects all tensors to be of complex datatype, i.e., `torch.is_complex(x)` to evaluate to `True`.

    :param h: shape: (`*batch_dims`, dim)
        The complex head representations.
    :param r: shape: (`*batch_dims`, dim)
        The complex relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The complex tail representations.

    :return: shape: batch_dims
        The scores.
    """
    h, r, t = ensure_complex(h, r, t)
    # TODO: switch to einsum ?
    # return torch.real(torch.einsum("...d, ...d, ...d -> ...", h, r, torch.conj(t)))
    return torch.real(tensor_product(h, r, torch.conj(t)).sum(dim=-1))


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

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param t_bias: shape: (`*batch_dims`)
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

    :return: shape: batch_dims
        The scores.
    """
    # repeat if necessary, and concat head and relation
    # shape: -1, num_input_channels, 2*height, width
    x = torch.cat(
        torch.broadcast_tensors(
            h.view(*h.shape[:-1], input_channels, embedding_height, embedding_width),
            r.view(*r.shape[:-1], input_channels, embedding_height, embedding_width),
        ),
        dim=-2,
    )
    prefix_shape = x.shape[:-3]
    x = x.view(-1, input_channels, 2 * embedding_height, embedding_width)

    # shape: -1, num_input_channels, 2*height, width
    x = hr2d(x)

    # -1, num_output_channels * (2 * height - kernel_height + 1) * (width - kernel_width + 1)
    x = x.view(-1, numpy.prod(x.shape[-3:]))
    x = hr1d(x)

    # reshape: (-1, dim) -> (*batch_dims, dim)
    x = x.view(*prefix_shape, h.shape[-1])

    # For efficient calculation, each of the convolved [h, r] rows has only to be multiplied with one t row
    # output_shape: batch_dims
    x = torch.einsum("...d, ...d -> ...", x, t)

    # add bias term
    return x + t_bias


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

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param conv:
        The 3x1 convolution.
    :param activation:
        The activation function.
    :param hidden_dropout:
        The dropout layer applied to the hidden activations.
    :param linear:
        The final linear layer.

    :return: shape: batch_dims
        The scores.
    """
    # decompose convolution for faster computation in 1-n case
    num_filters = conv.weight.shape[0]
    assert conv.weight.shape == (num_filters, 1, 1, 3)

    # compute conv(stack(h, r, t))
    # prepare input shapes for broadcasting
    # (*batch_dims, 1, d)
    h = h.unsqueeze(dim=-2)
    r = r.unsqueeze(dim=-2)
    t = t.unsqueeze(dim=-2)

    # conv.weight.shape = (C_out, C_in, kernel_size[0], kernel_size[1])
    # here, kernel_size = (1, 3), C_in = 1, C_out = num_filters
    # -> conv_head, conv_rel, conv_tail shapes: (num_filters,)
    # reshape to (..., f, 1)
    conv_head, conv_rel, conv_tail, conv_bias = [
        c.view(*make_ones_like(h.shape[:-2]), num_filters, 1) for c in list(conv.weight[:, 0, 0, :].t()) + [conv.bias]
    ]

    # convolve -> output.shape: (*, embedding_dim, num_filters)
    h = conv_head @ h
    r = conv_rel @ r
    t = conv_tail @ t

    x = tensor_sum(conv_bias, h, r, t)
    x = activation(x)

    # Apply dropout, cf. https://github.com/daiquocnguyen/ConvKB/blob/master/model.py#L54-L56
    x = hidden_dropout(x)

    # Linear layer for final scores; use flattened representations, shape: (*batch_dims, d * f)
    x = x.view(*x.shape[:-2], -1)
    x = linear(x)
    return x.squeeze(dim=-1)


def distmult_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Evaluate the DistMult interaction function.

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.

    :return: shape: batch_dims
        The scores.
    """
    return tensor_product(h, r, t).sum(dim=-1)


def dist_ma_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    r"""Evaluate the DistMA interaction function from [shi2019]_.

    .. math ::
        \langle h, r\rangle + \langle r, t\rangle + \langle h, t\rangle

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.

    :return: shape: batch_dims
        The scores.
    """
    return batched_dot(h, r) + batched_dot(r, t) + batched_dot(h, t)


def ermlp_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    hidden: nn.Linear,
    activation: nn.Module,
    final: nn.Linear,
) -> torch.FloatTensor:
    r"""Evaluate the ER-MLP interaction function.

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param hidden:
        The first linear layer.
    :param activation:
        The activation function of the hidden layer.
    :param final:
        The second linear layer.

    :return: shape: batch_dims
        The scores.
    """
    # same shape
    *prefix, dim = h.shape
    if h.shape == r.shape and h.shape == t.shape:
        return final(activation(hidden(torch.cat([h, r, t], dim=-1).view(-1, 3 * dim)))).view(prefix)

    # split, shape: (embedding_dim, hidden_dim)
    head_to_hidden, rel_to_hidden, tail_to_hidden = hidden.weight.t().split(dim)
    bias = hidden.bias.view(*make_ones_like(prefix), -1)
    h = torch.einsum("...i,ij->...j", h, head_to_hidden)
    r = torch.einsum("...i,ij->...j", r, rel_to_hidden)
    t = torch.einsum("...i,ij->...j", t, tail_to_hidden)
    return final(activation(tensor_sum(bias, h, r, t))).squeeze(dim=-1)


def ermlpe_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    mlp: nn.Module,
) -> torch.FloatTensor:
    r"""Evaluate the ER-MLPE interaction function.

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param mlp:
        The MLP.

    :return: shape: batch_dims
        The scores.
    """
    # repeat if necessary, and concat head and relation, (batch_size, num_heads, num_relations, 1, 2 * embedding_dim)
    x = torch.cat(torch.broadcast_tensors(h, r), dim=-1)

    # Predict t embedding, shape: (*batch_dims, d)
    *batch_dims, dim = x.shape
    x = mlp(x.view(-1, dim)).view(*batch_dims, -1)

    # dot product
    return (x * t).sum(dim=-1)


def hole_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Evaluate the HolE interaction function.

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.

    :return: shape: batch_dims
        The scores.
    """
    # composite: (*batch_dims, d)
    composite = circular_correlation(h, t)

    # inner product with relation embedding
    return (r * composite).sum(dim=-1)


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

    :param h_mean: shape: (`*batch_dims`, d)
        The head entity distribution mean.
    :param h_var: shape: (`*batch_dims`, d)
        The head entity distribution variance.
    :param r_mean: shape: (`*batch_dims`, d)
        The relation distribution mean.
    :param r_var: shape: (`*batch_dims`, d)
        The relation distribution variance.
    :param t_mean: shape: (`*batch_dims`, d)
        The tail entity distribution mean.
    :param t_var: shape: (`*batch_dims`, d)
        The tail entity distribution variance.
    :param similarity:
        The similarity measures for gaussian distributions. From {"KL", "EL"}.
    :param exact:
        Whether to leave out constants to accelerate similarity computation.

    :return: shape: batch_dims
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

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param w: shape: (`*batch_dims`, k, dim, dim)
        The relation specific transformation matrix W_r.
    :param vh: shape: (`*batch_dims`, k, dim)
        The head transformation matrix V_h.
    :param vt: shape: (`*batch_dims`, k, dim)
        The tail transformation matrix V_h.
    :param b: shape: (`*batch_dims`, k)
        The relation specific offset b_r.
    :param u: shape: (`*batch_dims`, k)
        The relation specific final linear transformation b_r.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param activation:
        The activation function.

    :return: shape: batch_dims
        The scores.
    """
    return (
        u
        * activation(
            tensor_sum(
                torch.einsum("...d,...kde,...e->...k", h, w, t),  # shape: (*batch_dims, k)
                torch.einsum("...d, ...kd->...k", h, vh),
                torch.einsum("...d, ...kd->...k", t, vt),
                b,
            )
        )
    ).sum(dim=-1)


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

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param d_e: shape: (dim,)
        Global entity projection.
    :param d_r: shape: (dim,)
        Global relation projection.
    :param b_c: shape: (dim,)
        Global combination bias.
    :param b_p: shape: scalar
        Final score bias
    :param activation:
        The activation function.

    :return: shape: batch_dims
        The scores.
    """
    # global projections
    h = torch.einsum("...d, d -> ...d", h, d_e)
    r = torch.einsum("...d, d -> ...d", r, d_r)

    # combination, shape: (*batch_dims, d)
    x = activation(tensor_sum(h, r, b_c))

    # dot product with t
    return torch.einsum("...d, ...d -> ...", x, t) + b_p


def rescal_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Evaluate the RESCAL interaction function.

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.

    :return: shape: batch_dims
        The scores.
    """
    return torch.einsum("...d,...de,...e->...", h, r, t)


def rotate_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Evaluate the RotatE interaction function.

    .. note::
        this method expects all tensors to be of complex datatype, i.e., `torch.is_complex(x)` to evaluate to `True`.

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.

    :return: shape: batch_dims
        The scores.
    """
    h, r, t = ensure_complex(h, r, t)
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

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param h_inv: shape: (`*batch_dims`, dim)
        The inverse head representations.
    :param r_inv: shape: (`*batch_dims`, dim, dim)
        The relation representations.
    :param t_inv: shape: (`*batch_dims`, dim)
        The tail representations.
    :param clamp:
        Clamp the scores to the given range.

    :return: shape: batch_dims
        The scores.
    """
    scores = 0.5 * (distmult_interaction(h=h, r=r, t=t) + distmult_interaction(h=h_inv, r=r_inv, t=t_inv))
    # Note: In the code in their repository, the score is clamped to [-20, 20].
    #       That is not mentioned in the paper, so it is made optional here.
    if clamp:
        min_, max_ = clamp
        scores = scores.clamp(min=min_, max=max_)
    return scores


def se_interaction(
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

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r_h: shape: (`*batch_dims`, rel_dim, dim)
        The relation-specific head projection.
    :param r_t: shape: (`*batch_dims`, rel_dim, dim)
        The relation-specific tail projection.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param p:
        The p for the norm. cf. :func:`torch.linalg.vector_norm`.
    :param power_norm:
        Whether to return the powered norm.

    :return: shape: batch_dims
        The scores.
    """
    return negative_norm(
        (r_h @ h.unsqueeze(dim=-1) - r_t @ t.unsqueeze(dim=-1)).squeeze(dim=-1),
        p=p,
        power_norm=power_norm,
    )


def toruse_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    p: Union[int, str] = 2,
    power_norm: bool = False,
) -> torch.FloatTensor:
    """Evaluate the TorusE interaction function from [ebisu2018].

    .. note ::
        This only implements the two L_p norm based variants.

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param p:
        The p for the norm.
    :param power_norm:
        Whether to return the powered norm.

    :return: shape: batch_dims
        The scores.
    """
    d = tensor_sum(h, r, -t)
    d = d - torch.floor(d)
    d = torch.minimum(d, 1.0 - d)
    return negative_norm(d, p=p, power_norm=power_norm)


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

    :param h: shape: (`*batch_dims`, d_e)
        The head representations.
    :param r: shape: (`*batch_dims`, d_r)
        The relation representations.
    :param t: shape: (`*batch_dims`, d_e)
        The tail representations.
    :param h_p: shape: (`*batch_dims`, d_e)
        The head projections.
    :param r_p: shape: (`*batch_dims`, d_r)
        The relation projections.
    :param t_p: shape: (`*batch_dims`, d_e)
        The tail projections.
    :param p:
        The parameter p for selecting the norm.
    :param power_norm:
        Whether to return the powered norm instead.

    :return: shape: batch_dims
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

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param p:
        The p for the norm.
    :param power_norm:
        Whether to return the powered norm.

    :return: shape: batch_dims
        The scores.
    """
    return negative_norm_of_sum(h, r, -t, p=p, power_norm=power_norm)


def transf_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Evaluate the TransF interaction function.

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.

    :return: shape: batch_dims
        The scores.
    """
    return batched_dot(h + r, t) + batched_dot(h, t - r)


def transh_interaction(
    h: torch.FloatTensor,
    w_r: torch.FloatTensor,
    d_r: torch.FloatTensor,
    t: torch.FloatTensor,
    p: int,
    power_norm: bool = False,
) -> torch.FloatTensor:
    """Evaluate the DistMult interaction function.

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param w_r: shape: (`*batch_dims`, dim)
        The relation normal vector representations.
    :param d_r: shape: (`*batch_dims`, dim)
        The relation difference vector representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param p:
        The p for the norm. cf. :func:`torch.linalg.vector_norm`.
    :param power_norm:
        Whether to return $|x-y|_p^p$.

    :return: shape: batch_dims
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

    :param h: shape: (`*batch_dims`, d_e)
        Head embeddings.
    :param r: shape: (`*batch_dims`, d_r)
        Relation embeddings.
    :param m_r: shape: (`*batch_dims`, d_e, d_r)
        The relation specific linear transformations.
    :param t: shape: (`*batch_dims`, d_e)
        Tail embeddings.
    :param p:
        The parameter p for selecting the norm.
    :param power_norm:
        Whether to return the powered norm instead.

    :return: shape: batch_dims
        The scores.
    """
    # project to relation specific subspace
    h_bot = torch.einsum("...e, ...er -> ...r", h, m_r)
    t_bot = torch.einsum("...e, ...er -> ...r", t, m_r)
    # ensure constraints
    h_bot = clamp_norm(h_bot, p=2, dim=-1, maxnorm=1.0)
    t_bot = clamp_norm(t_bot, p=2, dim=-1, maxnorm=1.0)
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

    :param h: shape: (`*batch_dims`, d_e)
        The head representations.
    :param r: shape: (`*batch_dims`, d_r)
        The relation representations.
    :param t: shape: (`*batch_dims`, d_e)
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

    :return: shape: batch_dims
        The scores.
    """
    return (
        _apply_optional_bn_to_tensor(
            x=torch.einsum(
                # x_1 contraction
                "...ik,...i->...k",
                _apply_optional_bn_to_tensor(
                    x=torch.einsum(
                        # x_2 contraction
                        "ijk,...j->...ik",
                        core_tensor,
                        r,
                    ),
                    output_dropout=do_r,
                ),
                _apply_optional_bn_to_tensor(
                    x=h,
                    batch_norm=bn_h,
                    output_dropout=do_h,
                ),
            ),
            batch_norm=bn_hr,
            output_dropout=do_hr,
        )
        * t
    ).sum(dim=-1)


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

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param b_h: shape: batch_dims
        The head entity bias.
    :param r_vec: shape: (`*batch_dims`, dim)
        The relation vector.
    :param r_mat: shape: (`*batch_dims`, dim,)
        The diagonal relation matrix.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param b_t: shape: batch_dims
        The tail entity bias.
    :param p:
        The parameter p for selecting the norm, cf. :func:`torch.linalg.vector_norm`.
    :param power_norm:
        Whether to return the powered norm instead.

    :return: shape: batch_dims
        The scores.
    """
    return (
        negative_norm_of_sum(
            h * r_mat,
            r_vec,
            -t,
            p=p,
            power_norm=power_norm,
        )
        + b_h
        + b_t
    )


def um_interaction(
    h: torch.FloatTensor,
    t: torch.FloatTensor,
    p: int,
    power_norm: bool = True,
) -> torch.FloatTensor:
    """Evaluate the SimplE interaction function.

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param p:
        The parameter p for selecting the norm.
    :param power_norm:
        Whether to return the powered norm instead.

    :return: shape: batch_dims
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

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param r_h: shape: (`*batch_dims`, dim)
        The head part of the relation representations.
    :param r_t: shape: (`*batch_dims`, dim)
        The tail part of the relation representations.
    :param p:
        The parameter p for selecting the norm.
    :param power_norm:
        Whether to return the powered norm instead.

    :return: shape: batch_dims
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

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The head representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.

    :return: shape: (...)
        The scores.
    """
    return -(
        # Rotation in quaternion space
        _rotate_quaternion(
            _split_quaternion(h),
            _split_quaternion(r),
        )
        * t
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

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The relation representations.
    :param c_r: shape: (`*batch_dims`, dim)
        The relation-specific interaction vector.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param bias: shape: (dim,)
        The combination bias.
    :param activation:
        The combination activation. Should be :class:`torch.nn.Tanh` for consistency with the CrossE paper.
    :param dropout:
        Dropout applied after the combination.

    :return: shape: batch_dims
        The scores.

    .. seealso:: https://github.com/wencolani/CrossE
    """
    # head interaction
    h = c_r * h
    # relation interaction (notice that h has been updated)
    r = h * r
    # combination
    x = activation(h + r + bias.view(*make_ones_like(h.shape[:-1]), -1))
    if dropout is not None:
        x = dropout(x)
    # similarity
    return (x * t).sum(dim=-1)


def boxe_interaction(
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
) -> torch.FloatTensor:
    """
    Evalute the BoxE interaction function from [abboud2020]_.

    Entities are described via position and bump. Relations are described as a pair of boxes, where each box is
    parametrized as triple (base, delta, size), where # TODO

    .. note ::
        this interaction relies on Abboud's point-to-box distance
        :func:`pykeen.utils.point_to_box_distance`.

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
        boxe_kg_arity_position_score(
            entity_pos=entity_pos,
            other_entity_bump=other_entity_pos,
            relation_box=compute_box(base=base, delta=delta, size=size),
            tanh_map=tanh_map,
            p=p,
            power_norm=power_norm,
        )
        for entity_pos, other_entity_pos, base, delta, size in (
            (h_pos, t_bump, rh_base, rh_delta, rh_size),
            (t_pos, h_bump, rt_base, rt_delta, rt_size),
        )
    )


def cp_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Evaluate the Canonical Tensor Decomposition interaction function.

    :param h: shape: (`*batch_dims`, rank, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, rank, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, rank, dim)
        The tail representations.

    :return: shape: batch_dims
        The scores.
    """
    return (h * r * t).sum(dim=(-2, -1))


def triple_re_interaction(
    # head
    h: torch.FloatTensor,
    # relation
    r_head: torch.FloatTensor,
    r_mid: torch.FloatTensor,
    r_tail: torch.FloatTensor,
    # tail
    t: torch.FloatTensor,
    # version 2: relation factor offset
    u: Optional[float] = None,
    # extension: negative (power) norm
    p: int = 2,
    power_norm: bool = False,
) -> torch.FloatTensor:
    r"""Evaluate the TripleRE interaction function.

    .. math ::
        score(h, (r_h, r, r_t), t) = h * (r_h + u) - t * (r_t + u) + r

    .. note ::

        For equivalence to the paper version, `h` and `t` should be normalized to unit
        Euclidean length, and `p` and `power_norm` be kept at their default values.

    :param h: shape: (`*batch_dims`, rank, dim)
        The head representations.
    :param r_head: shape: (`*batch_dims`, rank, dim)
        The relation-specific head multiplicator representations.
    :param r_mid: shape: (`*batch_dims`, rank, dim)
        The relation representations.
    :param r_tail: shape: (`*batch_dims`, rank, dim)
        The relation-specific tail multiplicator representations.
    :param t: shape: (`*batch_dims`, rank, dim)
        The tail representations.
    :param u:
        the relation factor offset. If u is not None or 0, this corresponds to TripleREv2.
    :param p:
        The p for the norm. cf. :func:`torch.linalg.vector_norm`.
    :param power_norm:
        Whether to return the powered norm.

    :return: shape: batch_dims
        The scores.
    """
    # note: normalization should be done from the representations
    # cf. https://github.com/LongYu-360/TripleRE-Add-NodePiece/blob/994216dcb1d718318384368dd0135477f852c6a4/TripleRE%2BNodepiece/ogb_wikikg2/model.py#L317-L328  # noqa: E501
    # version 2
    if u is not None:
        # r_head = r_head + u * torch.ones_like(r_head)
        # r_tail = r_tail + u * torch.ones_like(r_tail)
        r_head = r_head + u
        r_tail = r_tail + u

    return negative_norm_of_sum(
        h * r_head,
        -t * r_tail,
        r_mid,
        p=p,
        power_norm=power_norm,
    )


def auto_sf_interaction(
    h: Sequence[torch.FloatTensor],
    r: Sequence[torch.FloatTensor],
    t: Sequence[torch.FloatTensor],
    coefficients: Sequence[Tuple[int, int, int, Sign]],
) -> torch.FloatTensor:
    r"""Evaluate an AutoSF-style interaction function as described by [zhang2020]_.

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

    - :class:`pykeen.models.DistMult`: one block, $\mathcal{C} = \{(0, 0, 0, 1)\}$
    - :class:`pykeen.models.ComplEx`: two blocks,
      $\mathcal{C} = \{(0, 0, 0, 1), (0, 1, 1, 1), (1, 0, 1, -1), (1, 0, 1, 1)\}$
    - :class:`pykeen.models.SimplE`: two blocks: $\mathcal{C} = \{(0, 0, 1, 1), (1, 1, 0, 1)\}$

    :param h: each shape: (`*batch_dims`, rank, dim)
        The list of head representations.
    :param r: each shape: (`*batch_dims`, rank, dim)
        The list of relation representations.
    :param t: each shape: (`*batch_dims`, rank, dim)
        The list of tail representations.
    :param coefficients:
        the coefficients, in order:

        1. head_representation_index,
        2. relation_representation_index,
        3. tail_representation_index,
        4. sign

    :return:
        The scores
    """
    return sum(sign * (h[hi] * r[ri] * t[ti]).sum(dim=-1) for hi, ri, ti, sign in coefficients)


def transformer_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    transformer: nn.TransformerEncoder,
    position_embeddings: torch.FloatTensor,
    final: nn.Module,
) -> torch.FloatTensor:
    r"""Evaluate the Transformer interaction function, as described in [galkin2020]_..

    .. math ::

        \textit{score}(h, r, t) =
            \textit{Linear}(\textit{SumPooling}(\textit{Transformer}([h + pe[0]; r + pe[1]])))^T t

    :param h: shape: (`*batch_dims`, dim)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim)
        The tail representations.
    :param transformer:
        the transformer encoder
    :param position_embeddings: shape: (2, dim)
        the positional embeddings, one for head and one for relation
    :param final:
        the final (linear) transformation

    :return:
        The scores.
    """
    # stack h & r (+ broadcast) => shape: (2, *batch_dims, dim)
    x = torch.stack(broadcast_tensors(h, r), dim=0)

    # remember shape for output, but reshape for transformer
    hr_shape = x.shape
    x = x.view(2, -1, hr_shape[-1])

    # get position embeddings, shape: (seq_len, dim)
    # Now we are position-dependent w.r.t qualifier pairs.
    x = x + position_embeddings.unsqueeze(dim=1)

    # seq_length, batch_size, dim
    x = transformer(src=x)

    # Pool output
    x = x.sum(dim=0)

    # output shape: (batch_size, dim)
    x = final(x)

    # reshape
    x = x.view(*hr_shape[1:-1], x.shape[-1])

    return (x * t).sum(dim=-1)


def multilinear_tucker_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    core_tensor: torch.FloatTensor,
) -> torch.FloatTensor:
    r"""Evaluate the (original) multi-linear TuckEr interaction function.

    .. math ::

        score(h, r, t) = \sum W_{ijk} h_i r_j t_k

    :param h: shape: (`*batch_dims`, d_e)
        The head representations.
    :param r: shape: (`*batch_dims`, d_r)
        The relation representations.
    :param t: shape: (`*batch_dims`, d_e)
        The tail representations.
    :param core_tensor: shape: (d_h, d_r, d_t)
        The core tensor.

    :return: shape: batch_dims
        The scores.
    """
    return torch.einsum("ijk,...i,...j,...k->...", core_tensor, h, r, t)
