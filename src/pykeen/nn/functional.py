# -*- coding: utf-8 -*-

"""Functional forms of interaction methods."""
import math
from typing import NamedTuple, Optional, SupportsFloat, Tuple, Union

import torch
import torch.fft
from torch import nn

from ..utils import broadcast_cat, clamp_norm, is_cudnn_error, split_complex

__all__ = [
    "complex_interaction",
    "conve_interaction",
    "convkb_interaction",
    "distmult_interaction",
    "ermlp_interaction",
    "ermlpe_interaction",
    "hole_interaction",
    "rotate_interaction",
    "translational_interaction",
    "transr_interaction",
]


def _extract_sizes(h, r, t) -> Tuple[int, int, int, int, int]:
    num_heads, num_relations, num_tails = [xx.shape[1] for xx in (h, r, t)]
    d_e = h.shape[-1]
    d_r = r.shape[-1]
    return num_heads, num_relations, num_tails, d_e, d_r


def _extended_einsum(
    eq: str,
    *tensors,
) -> torch.FloatTensor:
    """Drop dimensions of size 1 to allow broadcasting."""
    # TODO: check if einsum is still very slow.
    lhs, rhs = eq.split("->")
    mod_ops, mod_t = [], []
    for op, t in zip(lhs.split(","), tensors):
        mod_op = ""
        assert len(op) == len(t.shape)
        for i, c in reversed(list(enumerate(op))):
            if t.shape[i] == 1:
                t = t.squeeze(dim=i)
            else:
                mod_op = c + mod_op
        mod_ops.append(mod_op)
        mod_t.append(t)
    m_lhs = ",".join(mod_ops)
    r_keep_dims = set("".join(mod_ops))
    m_rhs = "".join(c for c in rhs if c in r_keep_dims)
    m_eq = f"{m_lhs}->{m_rhs}"
    mod_r = torch.einsum(m_eq, *mod_t)
    # unsqueeze
    for i, c in enumerate(rhs):
        if c not in r_keep_dims:
            mod_r = mod_r.unsqueeze(dim=i)
    return mod_r


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


@_add_cuda_warning
def conve_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    t_bias: torch.FloatTensor,
    input_channels: int,
    embedding_height: int,
    embedding_width: int,
    num_in_features: int,
    bn0: Optional[nn.BatchNorm1d],
    bn1: Optional[nn.BatchNorm1d],
    bn2: Optional[nn.BatchNorm1d],
    inp_drop: nn.Dropout,
    feature_map_drop: nn.Dropout2d,
    hidden_drop: nn.Dropout,
    conv1: nn.Conv2d,
    activation: nn.Module,
    fc: nn.Linear,
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
    :param num_in_features:
        The number of output features of the final layer (calculated with kernel and embedding dimensions).
    :param bn0:
        The first batch normalization layer.
    :param bn1:
        The second batch normalization layer.
    :param bn2:
        The third batch normalization layer.
    :param inp_drop:
        The input dropout layer.
    :param feature_map_drop:
        The feature map dropout layer.
    :param hidden_drop:
        The hidden dropout layer.
    :param conv1:
        The convolution layer.
    :param activation:
        The activation function.
    :param fc:
        The final fully connected layer.

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
    x = broadcast_cat(h, r, dim=2).view(-1, input_channels, 2 * embedding_height, embedding_width)

    # batch_size, num_input_channels, 2*height, width
    if bn0 is not None:
        x = bn0(x)

    # batch_size, num_input_channels, 2*height, width
    x = inp_drop(x)

    # (N,C_out,H_out,W_out)
    x = conv1(x)

    if bn1 is not None:
        x = bn1(x)

    x = activation(x)
    x = feature_map_drop(x)

    # batch_size', num_output_channels * (2 * height - kernel_height + 1) * (width - kernel_width + 1)
    x = x.view(-1, num_in_features)
    x = fc(x)
    x = hidden_drop(x)

    if bn2 is not None:
        x = bn2(x)
    x = activation(x)

    # reshape: (batch_size', embedding_dim)
    x = x.view(-1, num_heads, num_relations, 1, embedding_dim)

    # For efficient calculation, each of the convolved [h, r] rows has only to be multiplied with one t row
    # output_shape: (batch_size, num_heads, num_relations, num_tails)
    t = t.view(t.shape[0], 1, 1, num_tails, embedding_dim).transpose(-1, -2)
    x = (x @ t).squeeze(dim=-2)

    # add bias term
    x = x + t_bias.view(t.shape[0], 1, 1, num_tails)

    return x


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
    return _extended_einsum("bhd,brd,btd->bhrt", h, r, t)


def complex_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    Evaluate the ComplEx interaction function.

    :param h: shape: (batch_size, num_heads, 2*dim)
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
        _extended_einsum("bhd,brd,btd->bhrt", hh, rr, tt)
        for hh, rr, tt in [
            (h_re, r_re, t_re),
            (h_re, r_im, t_im),
            (h_im, r_re, t_im),
            (h_im, r_im, t_re),
        ]
    )


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
    assert embedding_dim % 3 == 0
    embedding_dim = embedding_dim // 3
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

    # Predict t embedding, shape: (batch_size, num_heads, num_relations, embedding_dim)
    x = mlp(x)

    return (x.unsqueeze(dim=-2) @ t.view(t.shape[0], 1, 1, t.shape[1], t.shape[2]).transpose(-2, -1)).squeeze(dim=-1)


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
    return _extended_einsum("bhtd,brd->bhrt", composite, r)


def _view_complex(
    x: torch.FloatTensor,
) -> torch.Tensor:
    real, imag = split_complex(x=x)
    return torch.complex(real=real, imag=imag)


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
    h, r, t = [_view_complex(x) for x in (h, r, t)]

    # Rotate (=Hadamard product in complex space).
    hr = _extended_einsum("bhd,brd->bhrd", h, r)

    # Workaround until https://github.com/pytorch/pytorch/issues/30704 is fixed
    return negative_norm_of_sum(
        hr.unsqueeze(dim=3),
        t.view(t.shape[0], 1, 1, t.shape[1], t.shape[2]),
        p=2,
        power_norm=False,
    )


def negative_norm_of_sum(
    *x: torch.FloatTensor,
    p: Union[int, str] = 2,
    power_norm: bool = False,
) -> torch.FloatTensor:
    """
    Evaluate negative norm of a sum of vectors on already broadcasted representations.

    :param x: shape: (batch_size, num_heads, num_relations, num_tails, dim)
        The representations.
    :param p:
        The p for the norm. cf. torch.norm.
    :param power_norm:
        Whether to return |x-y|_p^p, cf. https://github.com/pytorch/pytorch/issues/28119

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    d = sum(x)
    if power_norm:
        assert isinstance(p, SupportsFloat)
        return -(d.abs() ** p).sum(dim=-1)
    else:
        if torch.is_complex(d):
            # workaround for complex numbers: manually compute norm
            return -(d.abs() ** p).sum(dim=-1) ** (1 / p)
        else:
            return -d.norm(p=p, dim=-1)


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
        Whether to return |x-y|_p^p, cf. https://github.com/pytorch/pytorch/issues/28119

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return negative_norm_of_sum(h, r, -t, p=p, power_norm=power_norm)


def translational_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    p: Union[int, str] = 2,
    power_norm: bool = False,
) -> torch.FloatTensor:
    """
    Evaluate a translational distance interaction function.

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param r: shape: (batch_size, num_relations, dim)
        The relation representations.
    :param t: shape: (batch_size, num_tails, dim)
        The tail representations.
    :param p:
        The p for the norm. cf. torch.norm.
    :param power_norm:
        Whether to return |x-y|_p^p, cf. https://github.com/pytorch/pytorch/issues/28119

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    num_heads, num_relations, num_tails, d_e, _ = _extract_sizes(h, r, t)
    h = h.view(-1, num_heads, 1, 1, d_e)
    r = r.view(-1, 1, num_relations, 1, d_e)
    t = t.view(-1, 1, 1, num_tails, d_e)
    return _translational_interaction(h=h, r=r, t=t, p=p, power_norm=power_norm)


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


class GaussianDistribution(NamedTuple):
    """A gaussian distribution with diagonal covariance matrix."""

    mean: torch.FloatTensor
    diagonal_covariance: torch.FloatTensor


def _expected_likelihood(
    e: GaussianDistribution,
    r: GaussianDistribution,
    epsilon: float = 1.0e-10,
    exact: bool = True,
) -> torch.FloatTensor:
    r"""Compute the similarity based on expected likelihood.

    .. math::

        D((\mu_e, \Sigma_e), (\mu_r, \Sigma_r)))
        = \frac{1}{2} \left(
            (\mu_e - \mu_r)^T(\Sigma_e + \Sigma_r)^{-1}(\mu_e - \mu_r)
            + \log \det (\Sigma_e + \Sigma_r) + d \log (2 \pi)
        \right)
        = \frac{1}{2} \left(
            \mu^T\Sigma^{-1}\mu
            + \log \det \Sigma + d \log (2 \pi)
        \right)

    :param e: shape: (batch_size, num_heads, num_tails, d)
        The entity Gaussian distribution.
    :param r: shape: (batch_size, num_relations, d)
        The relation Gaussian distribution.
    :param epsilon: float (default=1.0)
        Small constant used to avoid numerical issues when dividing.
    :param exact:
        Whether to return the exact similarity, or leave out constant offsets.

    :return: torch.Tensor, shape: (s_1, ..., s_k)
        The similarity.
    """
    # subtract, shape: (batch_size, num_heads, num_relations, num_tails, dim)
    r_shape = r.mean.shape
    r_shape = (r_shape[0], 1, r_shape[1], 1, r_shape[2])
    var = r.diagonal_covariance.view(*r_shape) + e.diagonal_covariance.unsqueeze(dim=2)
    mean = e.mean.unsqueeze(dim=2) - r.mean.view(*r_shape)

    #: a = \mu^T\Sigma^{-1}\mu
    safe_sigma = torch.clamp_min(var, min=epsilon)
    sigma_inv = torch.reciprocal(safe_sigma)
    sim = torch.sum(sigma_inv * mean ** 2, dim=-1)

    #: b = \log \det \Sigma
    sim = sim + safe_sigma.log().sum(dim=-1)
    if exact:
        sim = sim + sim.shape[-1] * math.log(2. * math.pi)
    return sim


def _kullback_leibler_similarity(
    e: GaussianDistribution,
    r: GaussianDistribution,
    epsilon: float = 1.0e-10,
    exact: bool = True,
) -> torch.FloatTensor:
    r"""Compute the similarity based on KL divergence.

    This is done between two Gaussian distributions given by mean mu_* and diagonal covariance matrix sigma_*.

    .. math::

        D((\mu_e, \Sigma_e), (\mu_r, \Sigma_r)))
        = \frac{1}{2} \left(
            tr(\Sigma_r^{-1}\Sigma_e)
            + (\mu_r - \mu_e)^T\Sigma_r^{-1}(\mu_r - \mu_e)
            - \log \frac{det(\Sigma_e)}{det(\Sigma_r)} - k_e
        \right)

    Note: The sign of the function has been flipped as opposed to the description in the paper, as the
          Kullback Leibler divergence is large if the distributions are dissimilar.

    :param e: shape: (batch_size, num_heads, num_tails, d)
        The entity Gaussian distributions, as mean/diagonal covariance pairs.
    :param r: shape: (batch_size, num_relations, d)
        The relation Gaussian distributions, as mean/diagonal covariance pairs.
    :param epsilon: float (default=1.0)
        Small constant used to avoid numerical issues when dividing.
    :param exact:
        Whether to return the exact similarity, or leave out constant offsets.

    :return: torch.Tensor, shape: (s_1, ..., s_k)
        The similarity.
    """
    # invert covariance, shape: (batch_size, num_relations, d)
    safe_sigma_r = torch.clamp_min(r.diagonal_covariance, min=epsilon)
    sigma_r_inv = torch.reciprocal(safe_sigma_r)

    #: a = tr(\Sigma_r^{-1}\Sigma_e), (batch_size, num_heads, num_relations, num_tails)
    # [(b, h, t, d), (b, r, d) -> (b, 1, r, d) -> (b, 1, d, r)] -> (b, h, t, r) -> (b, h, r, t)
    sim = (e.diagonal_covariance @ sigma_r_inv.unsqueeze(dim=1).transpose(-2, -1)).transpose(-2, -1)

    #: b = (\mu_r - \mu_e)^T\Sigma_r^{-1}(\mu_r - \mu_e)
    r_shape = r.mean.shape
    # mu.shape: (b, h, r, t, d)
    mu = r.mean.view(r_shape[0], 1, r_shape[1], 1, r_shape[2]) - e.mean.unsqueeze(dim=2)
    sim = sim + (mu ** 2 @ sigma_r_inv.view(r_shape[0], 1, r_shape[1], r_shape[2], 1)).squeeze(dim=-1)

    #: c = \log \frac{det(\Sigma_e)}{det(\Sigma_r)}
    # = sum log (sigma_e)_i - sum log (sigma_r)_i
    # ce.shape: (b, h, t)
    ce = e.diagonal_covariance.clamp_min(min=epsilon).log().sum(dim=-1)
    # cr.shape: (b, r)
    cr = safe_sigma_r.log().sum(dim=-1)
    sim = sim + ce.unsqueeze(dim=2) - cr.view(r_shape[0], 1, r_shape[1], 1)

    if exact:
        sim = sim - e.mean.shape[-1]
        sim = 0.5 * sim

    return sim


_KG2E_SIMILARITIES = dict(
    KL=_kullback_leibler_similarity,
    EL=_expected_likelihood,
)
KG2E_SIMILARITIES = set(_KG2E_SIMILARITIES.keys())


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
    if similarity not in KG2E_SIMILARITIES:
        raise KeyError(similarity)
    similarity = _KG2E_SIMILARITIES[similarity]
    # Compute entity distribution
    e_mean = h_mean.unsqueeze(dim=2) - t_mean.unsqueeze(dim=1)
    e_var = h_var.unsqueeze(dim=2) + t_var.unsqueeze(dim=1)
    e = GaussianDistribution(mean=e_mean, diagonal_covariance=e_var)
    r = GaussianDistribution(mean=r_mean, diagonal_covariance=r_var)
    return similarity(e=e, r=r, exact=exact)


def ntn_interaction(
    h: torch.FloatTensor,
    t: torch.FloatTensor,
    w: torch.FloatTensor,
    b: torch.FloatTensor,
    u: torch.FloatTensor,
    vh: torch.FloatTensor,
    vt: torch.FloatTensor,
    activation: nn.Module,
) -> torch.FloatTensor:
    r"""
    Evaluate the NTN interaction function.

    .. math::

        f(h,r,t) = u_r^T act(h W_r t + V_r h + V_r' t + b_r)

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param vh: shape: (batch_size, num_relations, k, dim)
        The head transformation matrix V_h.
    :param vt: shape: (batch_size, num_relations, k, dim)
        The tail transformation matrix V_h.
    :param w: shape: (batch_size, num_relations, k, dim, dim)
        The relation specific transformation matrix W_r.
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
    x = _extended_einsum("bhd,brkde,bte->bhrtk", h, w, t)
    x = x + _extended_einsum("brkd,bhd->bhk", vh, h).view(-1, num_heads, 1, 1, k)
    x = x + _extended_einsum("brkd,btd->btk", vt, t).view(-1, 1, 1, num_tails, k)
    x = activation(x)
    x = _extended_einsum("bhrtk,brk->bhrt", x, u)
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
    return _extended_einsum("bhd,brde,bte->bhrt", h, r, t)


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
        _extended_einsum("brde,bhd->bhre", r_h, h).unsqueeze(dim=3),
        -_extended_einsum("brde,btd->brte", r_t, t).unsqueeze(dim=1),
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
        Whether to return |x-y|_p^p.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # Project to hyperplane
    return _translational_interaction(
        h=(h.unsqueeze(dim=2) - _extended_einsum("bhd,brd,bre->bhre", h, w_r, w_r)).unsqueeze(dim=3),
        r=d_r.view(d_r.shape[0], 1, d_r.shape[1], 1, d_r.shape[2]),
        t=(t.unsqueeze(dim=1) - _extended_einsum("btd,brd,bre->brte", t, w_r, w_r)).unsqueeze(dim=1),
        p=p,
        power_norm=power_norm,
    )


def _apply_optional_bn_to_tensor(
    batch_norm: Optional[nn.BatchNorm1d],
    output_dropout: nn.Dropout,
    tensor: torch.FloatTensor,
) -> torch.FloatTensor:
    if batch_norm is not None:
        shape = tensor.shape
        tensor = tensor.view(-1, shape[-1])
        tensor = batch_norm(tensor)
        tensor = tensor.view(*shape)
    tensor = output_dropout(tensor)
    return tensor


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
    x = do0(_extended_einsum("ijd,brd->brij", core_tensor, r))

    # Compute h_n = DO(BN(h))
    h = _apply_optional_bn_to_tensor(batch_norm=bn1, output_dropout=do1, tensor=h)

    # compute whr = DO(BN(h_n x_1 wr))
    x = _extended_einsum("brid,bhd->bhri", h, x)
    x = _apply_optional_bn_to_tensor(batch_norm=bn2, tensor=x, output_dropout=do2)

    # Compute whr x_3 t
    return _extended_einsum("bhrd,btd->bhrt", x, t)
