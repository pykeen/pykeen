"""Utilities for handling exoctic algebras such as quaternions."""
from functools import lru_cache

import torch

__all__ = [
    "quaterion_multiplication_table",
]


@lru_cache(1)
def quaterion_multiplication_table() -> torch.Tensor:
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
