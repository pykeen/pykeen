"""Functional forms of interaction methods.

These implementations allow for an arbitrary number of batch dimensions,
as well as broadcasting and thus naturally support slicing and 1:n scoring.
"""

from __future__ import annotations

import torch
from torch import broadcast_tensors, nn

from ..typing import FloatTensor
from ..utils import einsum

__all__ = [
    "transformer_interaction",
]


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
