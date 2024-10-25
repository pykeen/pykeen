"""Utilities for quaternions."""

from functools import lru_cache

import torch

from ..typing import FloatTensor

__all__ = [
    "normalize",
    "hamiltonian_product",
    "multiplication_table",
]


def normalize(x: FloatTensor) -> FloatTensor:
    r"""
    Normalize the length of relation vectors, if the forward constraint has not been applied yet.

    Absolute value of a quaternion

    .. math::

        |a + bi + cj + dk| = \sqrt{a^2 + b^2 + c^2 + d^2}

    L2 norm of quaternion vector:

    .. math::
        \|x\|^2 = \sum_{i=1}^d |x_i|^2
                 = \sum_{i=1}^d (x_i.re^2 + x_i.im_1^2 + x_i.im_2^2 + x_i.im_3^2)

    :param x: shape: ``(*batch_dims, 4 \cdot d)``
        The vector in flat form.

    :return: shape: ``(*batch_dims, 4 \cdot d)``
        The normalized vector.
    """
    # Normalize relation embeddings
    shape = x.shape
    x = x.view(*shape[:-1], -1, 4)
    x = torch.nn.functional.normalize(x, p=2, dim=-1)
    return x.view(*shape)


def hamiltonian_product(qa: FloatTensor, qb: FloatTensor) -> FloatTensor:
    """Compute the hamiltonian product of two quaternions (which enables rotation)."""
    return torch.stack(
        [
            qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3],
            qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2],
            qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1],
            qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0],
        ],
        dim=-1,
    )


@lru_cache(1)
def multiplication_table() -> FloatTensor:
    """
    Create the quaternion basis multiplication table.

    :return: shape: (4, 4, 4)
        the table of products of basis elements.

    ..seealso:: https://en.wikipedia.org/wiki/Quaternion#Multiplication_of_basis_elements
    """
    _1, _i, _j, _k = 0, 1, 2, 3
    table = torch.zeros(4, 4, 4)
    for i, j, k, v in [
        # 1 * ? = ?; ? * 1 = ?
        (_1, _1, _1, 1),
        (_1, _i, _i, 1),
        (_1, _j, _j, 1),
        (_1, _k, _k, 1),
        (_i, _1, _i, 1),
        (_j, _1, _j, 1),
        (_k, _1, _k, 1),
        # i**2 = j**2 = k**2 = -1
        (_i, _i, _1, -1),
        (_j, _j, _1, -1),
        (_k, _k, _1, -1),
        # i * j = k; i * k = -j
        (_i, _j, _k, 1),
        (_i, _k, _j, -1),
        # j * i = -k, j * k = i
        (_j, _i, _k, -1),
        (_j, _k, _i, 1),
        # k * i = j; k * j = -i
        (_k, _i, _j, 1),
        (_k, _j, _i, -1),
    ]:
        table[i, j, k] = v
    return table
