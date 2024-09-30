"""Functional forms of interaction methods.

These implementations allow for an arbitrary number of batch dimensions,
as well as broadcasting and thus naturally support slicing and 1:n scoring.
"""

from __future__ import annotations

import torch
from torch import broadcast_tensors, nn

from .compute_kernel import batched_dot
from .sim import KG2E_SIMILARITIES
from ..typing import FloatTensor, GaussianDistribution
from ..utils import (
    clamp_norm,
    einsum,
    make_ones_like,
    negative_norm,
    negative_norm_of_sum,
    project_entity,
    tensor_product,
    tensor_sum,
)

__all__ = [
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
    "linea_re_interaction",
]


def _apply_optional_bn_to_tensor(
    x: FloatTensor,
    output_dropout: nn.Dropout,
    batch_norm: nn.BatchNorm1d | None = None,
) -> FloatTensor:
    """Apply optional batch normalization and dropout layer. Supports multiple batch dimensions."""
    if batch_norm is not None:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x = batch_norm(x)
        x = x.view(*shape)
    return output_dropout(x)


def ermlp_interaction(
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
    hidden: nn.Linear,
    activation: nn.Module,
    final: nn.Linear,
) -> FloatTensor:
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
    # shortcut for same shape
    if h.shape == r.shape and h.shape == t.shape:
        x = hidden(torch.cat([h, r, t], dim=-1))
    else:
        # split weight into head-/relation-/tail-specific sub-matrices
        *prefix, dim = h.shape
        x = tensor_sum(
            hidden.bias.view(*make_ones_like(prefix), -1),
            *(
                einsum("...i, ji -> ...j", xx, weight)
                for xx, weight in zip([h, r, t], hidden.weight.split(split_size=dim, dim=-1))
            ),
        )
    return final(activation(x)).squeeze(dim=-1)


def ermlpe_interaction(
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
    mlp: nn.Module,
) -> FloatTensor:
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
    return einsum("...d,...d->...", x, t)


def hole_interaction(
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
) -> FloatTensor:
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
    a: FloatTensor,
    b: FloatTensor,
) -> FloatTensor:
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
    a_fft = torch.fft.rfft(a, dim=-1)
    b_fft = torch.fft.rfft(b, dim=-1)
    # complex conjugate
    a_fft = torch.conj(a_fft)
    # Hadamard product in frequency domain
    p_fft = a_fft * b_fft
    # inverse real FFT
    return torch.fft.irfft(p_fft, n=a.shape[-1], dim=-1)


def kg2e_interaction(
    h_mean: FloatTensor,
    h_var: FloatTensor,
    r_mean: FloatTensor,
    r_var: FloatTensor,
    t_mean: FloatTensor,
    t_var: FloatTensor,
    similarity: str = "KL",
    exact: bool = True,
) -> FloatTensor:
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
    h: FloatTensor,
    t: FloatTensor,
    w: FloatTensor,
    vh: FloatTensor,
    vt: FloatTensor,
    b: FloatTensor,
    u: FloatTensor,
    activation: nn.Module,
) -> FloatTensor:
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
                einsum("...d,...kde,...e->...k", h, w, t),  # shape: (*batch_dims, k)
                einsum("...d, ...kd->...k", h, vh),
                einsum("...d, ...kd->...k", t, vt),
                b,
            )
        )
    ).sum(dim=-1)


def proje_interaction(
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
    d_e: FloatTensor,
    d_r: FloatTensor,
    b_c: FloatTensor,
    b_p: FloatTensor,
    activation: nn.Module,
) -> FloatTensor:
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
    h = einsum("...d, d -> ...d", h, d_e)
    r = einsum("...d, d -> ...d", r, d_r)

    # combination, shape: (*batch_dims, d)
    x = activation(tensor_sum(h, r, b_c))

    # dot product with t
    return einsum("...d, ...d -> ...", x, t) + b_p


def rescal_interaction(
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
) -> FloatTensor:
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
    return einsum("...d,...de,...e->...", h, r, t)


def simple_interaction(
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
    h_inv: FloatTensor,
    r_inv: FloatTensor,
    t_inv: FloatTensor,
    clamp: tuple[float, float] | None = None,
) -> FloatTensor:
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
    scores = 0.5 * (tensor_product(h, r, t).sum(dim=-1) + tensor_product(h_inv, r_inv, t_inv).sum(dim=-1))
    # Note: In the code in their repository, the score is clamped to [-20, 20].
    #       That is not mentioned in the paper, so it is made optional here.
    if clamp:
        min_, max_ = clamp
        scores = scores.clamp(min=min_, max=max_)
    return scores


def se_interaction(
    h: FloatTensor,
    r_h: FloatTensor,
    r_t: FloatTensor,
    t: FloatTensor,
    p: int,
    power_norm: bool = False,
) -> FloatTensor:
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
        einsum("...rd,...d->...r", r_h, h) - einsum("...rd,...d->...r", r_t, t),
        p=p,
        power_norm=power_norm,
    )


def toruse_interaction(
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
    p: int | str = 2,
    power_norm: bool = False,
) -> FloatTensor:
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
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
    h_p: FloatTensor,
    r_p: FloatTensor,
    t_p: FloatTensor,
    p: int,
    power_norm: bool = False,
) -> FloatTensor:
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
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
    p: int | str = 2,
    power_norm: bool = False,
) -> FloatTensor:
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
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
) -> FloatTensor:
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
    h: FloatTensor,
    w_r: FloatTensor,
    d_r: FloatTensor,
    t: FloatTensor,
    p: int,
    power_norm: bool = False,
) -> FloatTensor:
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
        -einsum("...i, ...i, ...j -> ...j", h, w_r, w_r),
        # r
        d_r,
        # -t projection to hyperplane
        -t,
        einsum("...i, ...i, ...j -> ...j", t, w_r, w_r),
        p=p,
        power_norm=power_norm,
    )


def transr_interaction(
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
    m_r: FloatTensor,
    p: int,
    power_norm: bool = True,
) -> FloatTensor:
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
    h_bot = einsum("...e, ...er -> ...r", h, m_r)
    t_bot = einsum("...e, ...er -> ...r", t, m_r)
    # ensure constraints
    h_bot = clamp_norm(h_bot, p=2, dim=-1, maxnorm=1.0)
    t_bot = clamp_norm(t_bot, p=2, dim=-1, maxnorm=1.0)
    return negative_norm_of_sum(h_bot, r, -t_bot, p=p, power_norm=power_norm)


def tucker_interaction(
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
    core_tensor: FloatTensor,
    do_h: nn.Dropout,
    do_r: nn.Dropout,
    do_hr: nn.Dropout,
    bn_h: nn.BatchNorm1d | None,
    bn_hr: nn.BatchNorm1d | None,
) -> FloatTensor:
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
            x=einsum(
                # x_1 contraction
                "...ik,...i->...k",
                _apply_optional_bn_to_tensor(
                    x=einsum(
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
    h: FloatTensor,
    b_h: FloatTensor,
    r_vec: FloatTensor,
    r_mat: FloatTensor,
    t: FloatTensor,
    b_t: FloatTensor,
    p: int | float | str = 2,
    power_norm: bool = False,
) -> FloatTensor:
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
    h: FloatTensor,
    t: FloatTensor,
    p: int,
    power_norm: bool = True,
) -> FloatTensor:
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
    h: FloatTensor,
    t: FloatTensor,
    r_h: FloatTensor,
    r_t: FloatTensor,
    p: int | str = 2,
    power_norm: bool = True,
) -> FloatTensor:
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


def quat_e_interaction(
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
    table: FloatTensor,
):
    """Evaluate the interaction function of QuatE for given embeddings.

    The embeddings have to be in a broadcastable shape.

    :param h: shape: (`*batch_dims`, dim, 4)
        The head representations.
    :param r: shape: (`*batch_dims`, dim, 4)
        The head representations.
    :param t: shape: (`*batch_dims`, dim, 4)
        The tail representations.
    :param table:
        the quaternion multiplication table.

    :return: shape: (...)
        The scores.
    """
    # TODO: this sign is in the official code, too, but why do we need it?
    return -einsum("...di, ...dj, ...dk, ijk -> ...", h, r, t, table)


def triple_re_interaction(
    # head
    h: FloatTensor,
    # relation
    r_head: FloatTensor,
    r_mid: FloatTensor,
    r_tail: FloatTensor,
    # tail
    t: FloatTensor,
    # version 2: relation factor offset
    u: float | None = None,
    # extension: negative (power) norm
    p: int = 2,
    power_norm: bool = False,
) -> FloatTensor:
    r"""Evaluate the TripleRE interaction function.

    .. seealso ::
        :class:`pykeen.nn.modules.TripleREInteraction` for the stateful interaction module

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


def transformer_interaction(
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
    transformer: nn.TransformerEncoder,
    position_embeddings: FloatTensor,
    final: nn.Module,
) -> FloatTensor:
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
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
    core_tensor: FloatTensor,
) -> FloatTensor:
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
    return einsum("ijk,...i,...j,...k->...", core_tensor, h, r, t)


def linea_re_interaction(
    # head
    h: FloatTensor,
    # relation
    r_head: FloatTensor,
    r_mid: FloatTensor,
    r_tail: FloatTensor,
    # tail
    t: FloatTensor,
    # extension: negative (power) norm
    p: int = 2,
    power_norm: bool = False,
) -> FloatTensor:
    """Evaluate the LineaRE interaction function.

    .. note ::
        the interaction is equivalent to TripleRE interaction without the `u` term.

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
    :param p:
        The p for the norm. cf. :func:`negative_norm_of_sum`.
    :param power_norm:
        Whether to return the powered norm. cf. :func:`negative_norm_of_sum`.

    :return: shape: batch_dims
        The scores.
    """
    return triple_re_interaction(
        h=h, r_head=r_head, r_mid=r_mid, r_tail=r_tail, t=t, u=None, p=p, power_norm=power_norm
    )
