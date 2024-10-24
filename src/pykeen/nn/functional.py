"""Functional forms of interaction methods.

These implementations allow for an arbitrary number of batch dimensions,
as well as broadcasting and thus naturally support slicing and 1:n scoring.
"""

from __future__ import annotations

import torch

from ..typing import FloatTensor
from ..utils import einsum


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
