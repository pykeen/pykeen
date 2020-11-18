# -*- coding: utf-8 -*-

"""Functional forms of interaction methods."""

from typing import Optional, Tuple, Union

import numpy
import torch
import torch.fft
from torch import nn

from .sim import KG2E_SIMILARITIES
from ..typing import GaussianDistribution
from ..utils import (
    broadcast_cat, clamp_norm, extended_einsum, is_cudnn_error, negative_norm_of_sum, project_entity,
    split_complex, view_complex,
)

__all__ = [
    "complex_interaction",
    "conve_interaction",
    "convkb_interaction",
    "distmult_interaction",
    "ermlp_interaction",
    "ermlpe_interaction",
    "hole_interaction",
    "kg2e_interaction",
    "ntn_interaction",
    "proje_interaction",
    "rescal_interaction",
    "rotate_interaction",
    "simple_interaction",
    "structured_embedding_interaction",
    "transd_interaction",
    "transe_interaction",
    "transh_interaction",
    "transr_interaction",
    "tucker_interaction",
    "unstructured_model_interaction",
]


# TODO @mberr documentation
def _extract_sizes(h, r, t) -> Tuple[int, int, int, int, int]:
    num_heads, num_relations, num_tails = [xx.shape[1] for xx in (h, r, t)]
    d_e = h.shape[-1]
    d_r = r.shape[-1]
    return num_heads, num_relations, num_tails, d_e, d_r


# TODO @mberr documentation
def _apply_optional_bn_to_tensor(
    batch_norm: Optional[nn.BatchNorm1d],
    output_dropout: nn.Dropout,
    tensor: torch.FloatTensor,
) -> torch.FloatTensor:
    if batch_norm is not None:
        shape = tensor.shape
        tensor = tensor.reshape(-1, shape[-1])
        tensor = batch_norm(tensor)
        tensor = tensor.view(*shape)
    tensor = output_dropout(tensor)
    return tensor


def _translational_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    p: Union[int, str] = 2,
    power_norm: bool = False,
) -> torch.FloatTensor:
    """
    Evaluate a translational distance interaction function on already broadcasted representations.

    :param h: shape: (batch_size, num_heads, num_relations, num_tails, dim)
        The head representations.
    :param r: shape: (batch_size, num_heads, num_relations, num_tails, dim)
        The relation representations.
    :param t: shape: (batch_size, num_heads, num_relations, num_tails, dim)
        The tail representations.
    :param p:
        The p for the norm. cf. torch.norm.
    :param power_norm:
        Whether to return $|x-y|_p^p$, cf. https://github.com/pytorch/pytorch/issues/28119

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return negative_norm_of_sum(h, r, -t, p=p, power_norm=power_norm)


def _add_cuda_warning(func):
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
    """
    Evaluate the ComplEx interaction function.

    :param h: shape: (batch_size, num_heads, `2*dim`)
        The complex head representations.
    :param r: shape: (batch_size, num_relations, 2*dim)
        The complex relation representations.
    :param t: shape: (batch_size, num_tails, 2*dim)
        The complex tail representations.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    (h_re, h_im), (r_re, r_im), (t_re, t_im) = [split_complex(x=x) for x in (h, r, t)]
    return sum(
        extended_einsum("bhd,brd,btd->bhrt", hh, rr, tt)
        for hh, rr, tt in [
            (h_re, r_re, t_re),
            (h_re, r_im, t_im),
            (h_im, r_re, t_im),
            (h_im, r_im, t_re),
        ]
    )


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
    """
    Evaluate the ConvE interaction function.

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param r: shape: (batch_size, num_relations, dim)
        The relation representations.
    :param t: shape: (batch_size, num_tails, dim)
        The tail representations.
    :param t_bias: shape: (batch_size, num_tails, dim)
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
    # bind sizes
    # batch_size = max(x.shape[0] for x in (h, r, t))
    num_heads = h.shape[1]
    num_relations = r.shape[1]
    num_tails = t.shape[1]
    embedding_dim = h.shape[-1]

    # repeat if necessary, and concat head and relation, batch_size', num_input_channels, 2*height, width
    # with batch_size' = batch_size * num_heads * num_relations
    h = h.unsqueeze(dim=2)
    h = h.view(*h.shape[:-1], input_channels, embedding_height, embedding_width)
    r = r.unsqueeze(dim=1)
    r = r.view(*r.shape[:-1], input_channels, embedding_height, embedding_width)
    x = broadcast_cat(h, r, dim=-2).view(-1, input_channels, 2 * embedding_height, embedding_width)

    # batch_size', num_input_channels, 2*height, width
    x = hr2d(x)

    # batch_size', num_output_channels * (2 * height - kernel_height + 1) * (width - kernel_width + 1)
    x = x.view(-1, numpy.prod(x.shape[-3:]))
    x = hr1d(x)

    # reshape: (batch_size', embedding_dim) -> (b, h, r, 1, d)
    x = x.view(-1, num_heads, num_relations, 1, embedding_dim)

    # For efficient calculation, each of the convolved [h, r] rows has only to be multiplied with one t row
    # output_shape: (batch_size, num_heads, num_relations, num_tails)
    t = t.view(t.shape[0], 1, 1, num_tails, embedding_dim).transpose(-1, -2)
    x = (x @ t).squeeze(dim=-2)

    # add bias term
    x = x + t_bias.view(t.shape[0], 1, 1, num_tails)

    return x


def convkb_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    conv: nn.Conv2d,
    activation: nn.Module,
    hidden_dropout: nn.Dropout,
    linear: nn.Linear,
) -> torch.FloatTensor:
    r"""
    Evaluate the ConvKB interaction function.

    .. math::
        W_L drop(act(W_C \ast ([h; r; t]) + b_C)) + b_L

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param r: shape: (batch_size, num_relations, dim)
        The relation representations.
    :param t: shape: (batch_size, num_tails, dim)
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
    # bind sizes
    num_heads, num_relations, num_tails, embedding_dim, _ = _extract_sizes(h, r, t)

    # decompose convolution for faster computation in 1-n case
    num_filters = conv.weight.shape[0]
    assert conv.weight.shape == (num_filters, 1, 1, 3)

    # compute conv(stack(h, r, t))
    conv_head, conv_rel, conv_tail = conv.weight[:, 0, 0, :].t()
    conv_bias = conv.bias.view(1, 1, 1, 1, 1, num_filters)
    # h.shape: (b, nh, d), conv_head.shape: (o), out.shape: (b, nh, d, o)
    h = (h.view(h.shape[0], h.shape[1], 1, 1, embedding_dim, 1) * conv_head.view(1, 1, 1, 1, 1, num_filters))
    r = (r.view(r.shape[0], 1, r.shape[1], 1, embedding_dim, 1) * conv_rel.view(1, 1, 1, 1, 1, num_filters))
    t = (t.view(t.shape[0], 1, 1, t.shape[1], embedding_dim, 1) * conv_tail.view(1, 1, 1, 1, 1, num_filters))
    x = activation(conv_bias + h + r + t)

    # Apply dropout, cf. https://github.com/daiquocnguyen/ConvKB/blob/master/model.py#L54-L56
    x = hidden_dropout(x)

    # Linear layer for final scores
    return linear(
        x.view(-1, embedding_dim * num_filters),
    ).view(-1, num_heads, num_relations, num_tails)


def distmult_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    Evaluate the DistMult interaction function.

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param r: shape: (batch_size, num_relations, dim)
        The relation representations.
    :param t: shape: (batch_size, num_tails, dim)
        The tail representations.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return extended_einsum("bhd,brd,btd->bhrt", h, r, t)


def ermlp_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    hidden: nn.Linear,
    activation: nn.Module,
    final: nn.Linear,
) -> torch.FloatTensor:
    r"""
    Evaluate the ER-MLP interaction function.

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param r: shape: (batch_size, num_relations, dim)
        The relation representations.
    :param t: shape: (batch_size, num_tails, dim)
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
    num_heads, num_relations, num_tails, embedding_dim, _ = _extract_sizes(h, r, t)
    hidden_dim = hidden.weight.shape[0]
    # split, shape: (embedding_dim, hidden_dim)
    head_to_hidden, rel_to_hidden, tail_to_hidden = hidden.weight.t().split(embedding_dim)
    bias = hidden.bias.view(1, 1, 1, 1, -1)
    h = h.view(-1, num_heads, 1, 1, embedding_dim) @ head_to_hidden.view(1, 1, 1, embedding_dim, hidden_dim)
    r = r.view(-1, 1, num_relations, 1, embedding_dim) @ rel_to_hidden.view(1, 1, 1, embedding_dim, hidden_dim)
    t = t.view(-1, 1, 1, num_tails, embedding_dim) @ tail_to_hidden.view(1, 1, 1, embedding_dim, hidden_dim)
    # TODO: Choosing which to combine first, h/r, h/t or r/t, depending on the shape might further improve
    #       performance in a 1:n scenario.
    return final(activation(bias + h + r + t)).squeeze(dim=-1)


def ermlpe_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    mlp: nn.Module,
) -> torch.FloatTensor:
    r"""
    Evaluate the ER-MLPE interaction function.

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param r: shape: (batch_size, num_relations, dim)
        The relation representations.
    :param t: shape: (batch_size, num_tails, dim)
        The tail representations.
    :param mlp:
        The MLP.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # repeat if necessary, and concat head and relation, (batch_size, num_heads, num_relations, 2 * embedding_dim)
    x = broadcast_cat(h.unsqueeze(dim=2), r.unsqueeze(dim=1), dim=-1)

    # Predict t embedding, shape: (b, h, r, 1, d)
    shape = x.shape
    x = mlp(x.view(-1, shape[-1])).view(*shape[:-1], -1).unsqueeze(dim=-2)

    # transpose t, (b, 1, 1, d, t)
    t = t.view(t.shape[0], 1, 1, t.shape[1], t.shape[2]).transpose(-2, -1)

    # dot product, (b, h, r, 1, t)
    x = (x @ t).squeeze(dim=-2)
    return x


def hole_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:  # noqa: D102
    """
    Evaluate the HolE interaction function.

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param r: shape: (batch_size, num_relations, dim)
        The relation representations.
    :param t: shape: (batch_size, num_tails, dim)
        The tail representations.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # Circular correlation of entity embeddings
    a_fft = torch.fft.rfft(h, dim=-1)
    b_fft = torch.fft.rfft(t, dim=-1)

    # complex conjugate, shape = (b, h, d)
    a_fft = torch.conj(a_fft)

    # Hadamard product in frequency domain, shape: (b, h, t, d)
    p_fft = a_fft.unsqueeze(dim=2) * b_fft.unsqueeze(dim=1)

    # inverse real FFT, shape: (b, h, t, d)
    composite = torch.fft.irfft(p_fft, n=h.shape[-1], dim=-1)

    # inner product with relation embedding
    return extended_einsum("bhtd,brd->bhrt", composite, r)


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
    """
    Evaluate the KG2E interaction function.

    :param h_mean: shape: (batch_size, num_heads, d)
        The head entity distribution mean.
    :param h_var: shape: (batch_size, num_heads, d)
        The head entity distribution variance.
    :param r_mean: shape: (batch_size, num_relations, d)
        The relation distribution mean.
    :param r_var: shape: (batch_size, num_relations, d)
        The relation distribution variance.
    :param t_mean: shape: (batch_size, num_tails, d)
        The tail entity distribution mean.
    :param t_var: shape: (batch_size, num_tails, d)
        The tail entity distribution variance.
    :param similarity:
        The similarity measures for gaussian distributions. From {"KL", "EL"}.
    :param exact:
        Whether to leave out constants to accelerate similarity computation.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    similarity_fn = KG2E_SIMILARITIES[similarity]
    # Compute entity distribution
    e_mean = h_mean.unsqueeze(dim=2) - t_mean.unsqueeze(dim=1)
    e_var = h_var.unsqueeze(dim=2) + t_var.unsqueeze(dim=1)
    e = GaussianDistribution(mean=e_mean, diagonal_covariance=e_var)
    r = GaussianDistribution(mean=r_mean, diagonal_covariance=r_var)
    return similarity_fn(e=e, r=r, exact=exact)


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
    r"""
    Evaluate the NTN interaction function.

    .. math::

        f(h,r,t) = u_r^T act(h W_r t + V_r h + V_r' t + b_r)

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param w: shape: (batch_size, num_relations, k, dim, dim)
        The relation specific transformation matrix W_r.
    :param vh: shape: (batch_size, num_relations, k, dim)
        The head transformation matrix V_h.
    :param vt: shape: (batch_size, num_relations, k, dim)
        The tail transformation matrix V_h.
    :param b: shape: (batch_size, num_relations, k)
        The relation specific offset b_r.
    :param u: shape: (batch_size, num_relations, k)
        The relation specific final linear transformation b_r.
    :param t: shape: (batch_size, num_tails, dim)
        The tail representations.
    :param activation:
        The activation function.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # save sizes
    num_heads, num_relations, num_tails, _, k = _extract_sizes(h, b, t)
    x = extended_einsum("bhd,brkde,bte->bhrtk", h, w, t)
    x = x + extended_einsum("brkd,bhd->bhk", vh, h).view(-1, num_heads, 1, 1, k)
    x = x + extended_einsum("brkd,btd->btk", vt, t).view(-1, 1, 1, num_tails, k)
    x = activation(x)
    x = extended_einsum("bhrtk,brk->bhrt", x, u)
    return x


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
    r"""
    Evaluate the ProjE interaction function.

    .. math::

        f(h, r, t) = g(t z(D_e h + D_r r + b_c) + b_p)

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param r: shape: (batch_size, num_relations, dim)
        The relation representations.
    :param t: shape: (batch_size, num_tails, dim)
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
    h = h * d_e.view(1, 1, dim)
    r = r * d_r.view(1, 1, dim)
    # combination, shape: (b, h, r, d)
    x = h.unsqueeze(dim=2) + r.unsqueeze(dim=1) + b_c.view(1, 1, 1, dim)
    x = activation(x)
    # dot product with t, shape: (b, h, r, t)
    return (x @ t.unsqueeze(dim=1).transpose(-2, -1)) + b_p


def rescal_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    Evaluate the RESCAL interaction function.

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param r: shape: (batch_size, num_relations, dim, dim)
        The relation representations.
    :param t: shape: (batch_size, num_tails, dim)
        The tail representations.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return extended_einsum("bhd,brde,bte->bhrt", h, r, t)


def rotate_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Evaluate the interaction function of RotatE for given embeddings.

    :param h: shape: (batch_size, num_heads, 2*dim)
        The head representations.
    :param r: shape: (batch_size, num_relations, 2*dim)
        The relation representations.
    :param t: shape: (batch_size, num_tails, 2*dim)
        The tail representations.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # # r expresses a rotation in complex plane.
    # # The inverse rotation is expressed by the complex conjugate of r.
    # # The score is computed as the distance of the relation-rotated head to the tail.
    # # Equivalently, we can rotate the tail by the inverse relation, and measure the distance to the head, i.e.
    # # |h * r - t| = |h - conj(r) * t|
    # r_inv = torch.stack([r[:, :, :, 0], -r[:, :, :, 1]], dim=-1)
    h, r, t = [view_complex(x) for x in (h, r, t)]

    # Rotate (=Hadamard product in complex space).
    hr = extended_einsum("bhd,brd->bhrd", h, r)

    # Workaround until https://github.com/pytorch/pytorch/issues/30704 is fixed
    return negative_norm_of_sum(
        hr.unsqueeze(dim=3),
        t.view(t.shape[0], 1, 1, t.shape[1], t.shape[2]),
        p=2,
        power_norm=False,
    )


def simple_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    h_inv: torch.FloatTensor,
    r_inv: torch.FloatTensor,
    t_inv: torch.FloatTensor,
    clamp: Optional[Tuple[float, float]] = None,
) -> torch.FloatTensor:
    """
    Evaluate the SimplE interaction function.

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param r: shape: (batch_size, num_relations, dim, dim)
        The relation representations.
    :param t: shape: (batch_size, num_tails, dim)
        The tail representations.
    :param h_inv: shape: (batch_size, num_heads, dim)
        The inverse head representations.
    :param r_inv: shape: (batch_size, num_relations, dim, dim)
        The relation representations.
    :param t_inv: shape: (batch_size, num_tails, dim)
        The tail representations.

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
    r"""
    Evaluate the Structured Embedding interaction function.

    .. math ::
        f(h, r, t) = \|R_h r - R_t t\|

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param r_h: shape: (batch_size, num_relations, dim, rel_dim)
        The relation-specific head projection.
    :param r_t: shape: (batch_size, num_relations, dim, rel_dim)
        The relation-specific tail projection.
    :param t: shape: (batch_size, num_tails, dim)
        The tail representations.
    :param p:
        The p for the norm. cf. torch.norm.
    :param power_norm:
        Whether to return the powered norm.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return negative_norm_of_sum(
        extended_einsum("brde,bhd->bhre", r_h, h).unsqueeze(dim=3),
        -extended_einsum("brde,btd->brte", r_t, t).unsqueeze(dim=1),
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
    """
    Evaluate the TransD interaction function.

    :param h: shape: (batch_size, num_heads, d_e)
        The head representations.
    :param r: shape: (batch_size, num_relations, d_r)
        The relation representations.
    :param t: shape: (batch_size, num_tails, d_e)
        The tail representations.
    :param h_p: shape: (batch_size, num_heads, d_e)
        The head projections.
    :param r_p: shape: (batch_size, num_relations, d_r)
        The relation projections.
    :param t_p: shape: (batch_size, num_tails, d_e)
        The tail projections.
    :param p:
        The parameter p for selecting the norm.
    :param power_norm:
        Whether to return the powered norm instead.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # Project entities
    # shape: (b, h, r, 1, d_r)
    h_bot = project_entity(
        e=h.unsqueeze(dim=2),
        e_p=h_p.unsqueeze(dim=2),
        r_p=r_p.unsqueeze(dim=1),
    ).unsqueeze(dim=-2)
    # shape: (b, 1, r, t, d_r)
    t_bot = project_entity(
        e=t.unsqueeze(dim=1),
        e_p=t_p.unsqueeze(dim=1),
        r_p=r_p.unsqueeze(dim=2),
    ).unsqueeze(dim=1)
    r = r.view(r.shape[0], 1, r.shape[1], 1, r.shape[2])
    return _translational_interaction(h=h_bot, r=r, t=t_bot, p=p, power_norm=power_norm)


def transe_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    p: Union[int, str] = 2,
    power_norm: bool = False,
) -> torch.FloatTensor:
    """
    Evaluate the TransE interaction function.

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param r: shape: (batch_size, num_relations, dim)
        The relation representations.
    :param t: shape: (batch_size, num_tails, dim)
        The tail representations.
    :param p:
        The p for the norm.
    :param power_norm:
        Whether to return the powered norm.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    num_heads, num_relations, num_tails, embedding_dim, _ = _extract_sizes(h, r, t)
    return _translational_interaction(
        h=h.view(-1, num_heads, 1, 1, embedding_dim),
        r=r.view(-1, 1, num_relations, 1, embedding_dim),
        t=t.view(-1, 1, 1, num_tails, embedding_dim),
        p=p,
        power_norm=power_norm,
    )


def transh_interaction(
    h: torch.FloatTensor,
    w_r: torch.FloatTensor,
    d_r: torch.FloatTensor,
    t: torch.FloatTensor,
    p: int,
    power_norm: bool = False,
) -> torch.FloatTensor:
    """
    Evaluate the DistMult interaction function.

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param w_r: shape: (batch_size, num_relations, dim)
        The relation normal vector representations.
    :param d_r: shape: (batch_size, num_relations, dim)
        The relation difference vector representations.
    :param t: shape: (batch_size, num_tails, dim)
        The tail representations.
    :param p:
        The p for the norm. cf. torch.norm.
    :param power_norm:
        Whether to return $|x-y|_p^p$.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # Project to hyperplane
    return _translational_interaction(
        h=(h.unsqueeze(dim=2) - extended_einsum("bhd,brd,bre->bhre", h, w_r, w_r)).unsqueeze(dim=3),
        r=d_r.view(d_r.shape[0], 1, d_r.shape[1], 1, d_r.shape[2]),
        t=(t.unsqueeze(dim=1) - extended_einsum("btd,brd,bre->brte", t, w_r, w_r)).unsqueeze(dim=1),
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
    """Evaluate the interaction function for given embeddings.

    :param h: shape: (batch_size, num_heads, d_e)
        Head embeddings.
    :param r: shape: (batch_size, num_relations, d_r)
        Relation embeddings.
    :param m_r: shape: (batch_size, num_relations, d_e, d_r)
        The relation specific linear transformations.
    :param t: shape: (batch_size, num_tails, d_e)
        Tail embeddings.
    :param p:
        The parameter p for selecting the norm.
    :param power_norm:
        Whether to return the powered norm instead.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    num_heads, num_relations, num_tails, d_e, d_r = _extract_sizes(h=h, r=r, t=t)
    # project to relation specific subspace and ensure constraints
    # head, shape: (b, h, r, 1, d_r)
    h_bot = h.view(-1, num_heads, 1, 1, d_e) @ m_r.view(-1, 1, num_relations, d_e, d_r)
    h_bot = clamp_norm(h_bot, p=2, dim=-1, maxnorm=1.)

    # head, shape: (b, 1, r, t, d_r)
    t_bot = t.view(-1, 1, 1, num_tails, d_e) @ m_r.view(-1, 1, num_relations, d_e, d_r)
    t_bot = clamp_norm(t_bot, p=2, dim=-1, maxnorm=1.)

    # evaluate score function, shape: (b, h, r, t)
    r = r.view(-1, 1, num_relations, 1, d_r)
    return _translational_interaction(h=h_bot, r=r, t=t_bot, p=p, power_norm=power_norm)


def tucker_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    core_tensor: torch.FloatTensor,
    do0: nn.Dropout,
    do1: nn.Dropout,
    do2: nn.Dropout,
    bn1: Optional[nn.BatchNorm1d],
    bn2: Optional[nn.BatchNorm1d],
) -> torch.FloatTensor:
    """
    Evaluate the TuckEr interaction function.

    Compute scoring function W x_1 h x_2 r x_3 t as in the official implementation, i.e. as

        DO(BN(DO(BN(h)) x_1 DO(W x_2 r))) x_3 t

    where BN denotes BatchNorm and DO denotes Dropout

    :param h: shape: (batch_size, num_heads, d_e)
        The head representations.
    :param r: shape: (batch_size, num_relations, d_r)
        The relation representations.
    :param t: shape: (batch_size, num_tails, d_e)
        The tail representations.
    :param core_tensor: shape: (d_e, d_r, d_e)
        The core tensor.
    :param do1:
        The dropout layer for the head representations.
    :param do0:
        The first hidden dropout.
    :param do2:
        The second hidden dropout.
    :param bn1:
        The first batch normalization layer.
    :param bn2:
        The second batch normalization layer.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # Compute wr = DO(W x_2 r)
    x = do0(extended_einsum("idj,brd->brij", core_tensor, r))

    # Compute h_n = DO(BN(h))
    h = _apply_optional_bn_to_tensor(batch_norm=bn1, output_dropout=do1, tensor=h)

    # compute whr = DO(BN(h_n x_1 wr))
    x = extended_einsum("brid,bhd->bhri", x, h)
    x = _apply_optional_bn_to_tensor(batch_norm=bn2, tensor=x, output_dropout=do2)

    # Compute whr x_3 t
    return extended_einsum("bhrd,btd->bhrt", x, t)


def unstructured_model_interaction(
    h: torch.FloatTensor,
    t: torch.FloatTensor,
    p: int,
    power_norm: bool = True,
) -> torch.FloatTensor:
    """
    Evaluate the SimplE interaction function.

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param t: shape: (batch_size, num_tails, dim)
        The tail representations.
    :param p:
        The parameter p for selecting the norm.
    :param power_norm:
        Whether to return the powered norm instead.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    h = h.unsqueeze(dim=2).unsqueeze(dim=3)
    t = t.unsqueeze(dim=1).unsqueeze(dim=2)
    return negative_norm_of_sum(h, -t, p=p, power_norm=power_norm)
