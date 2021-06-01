# -*- coding: utf-8 -*-

"""Embedding weight initialization routines."""

import math

import numpy as np
import torch
import torch.nn
import torch.nn.init
from torch.nn import functional

from ..utils import compose

__all__ = [
    'xavier_uniform_',
    'xavier_uniform_norm_',
    'xavier_normal_',
    'xavier_normal_norm_',
    'uniform_norm_',
    'normal_norm_',
    'init_phases',
]


def xavier_uniform_(tensor, gain: float = 1.):
    r"""Initialize weights of the tensor similarly to Glorot/Xavier initialization.

    Proceed as if it was a linear layer with fan_in of zero and Xavier uniform
    initialization is used, i.e. fill the weight of input `embedding` with values values
    sampled from :math:`\mathcal{U}(-a, a)` where

    .. math::

        a = \text{gain} \times \sqrt{\frac{6}{\text{embedding_dim}}}

    :param tensor: A tensor
    :param gain: An optional scaling factor, defaults to 1.0.
    :return: Embedding with weights by the Xavier uniform initializer.
    """
    bound = gain * 6 / math.sqrt(tensor.shape[-1])
    torch.nn.init.uniform_(tensor, -bound, bound)
    return tensor


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    r"""Initialize weights of the tensor similarly to Glorot/Xavier initialization.

    Proceed as if it was a linear layer with fan_in of zero and Xavier normal
    initialization is used. Fill the weight of input `embedding` with values values
    sampled from :math:`\mathcal{N}(0, a^2)` where

    .. math::

        a = \text{gain} \times \sqrt{\frac{2}{\text{embedding_dim}}}

    :param tensor: A tensor
    :param gain: An optional scaling factor, defaults to 1.0.
    :return: Embedding with weights by the Xavier normal initializer.
    """
    std = gain * 2 / math.sqrt(tensor.shape[-1])
    torch.nn.init.normal_(tensor, mean=0., std=std)
    return tensor


def init_phases(x: torch.Tensor) -> torch.Tensor:
    r"""Generate random phases between 0 and :math:`2\pi`."""
    phases = 2 * np.pi * torch.rand_like(x[..., :x.shape[-1] // 2])
    return torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1).detach()


xavier_uniform_norm_ = compose(
    torch.nn.init.xavier_uniform_,
    functional.normalize,
)
xavier_normal_norm_ = compose(
    torch.nn.init.xavier_normal_,
    functional.normalize,
)
uniform_norm_ = compose(
    torch.nn.init.uniform_,
    functional.normalize,
)
normal_norm_ = compose(
    torch.nn.init.normal_,
    functional.normalize,
)


def init_quaternions(
    x: torch.FloatTensor,
) -> torch.FloatTensor:
    """Initialize quaternion."""
    num_elements, dim = x.shape
    if dim % 4 != 0:
        raise ValueError("Quaternions have four components, but dimension {dim} is not divisible by four.")
    dim //= 4
    # scaling factor
    s = 1. / math.sqrt(2 * num_elements)
    # modulus ~ Uniform[-s, s]
    modulus = 2 * s * torch.rand(num_elements, dim) - s
    # phase ~ Uniform[0, 2*pi]
    phase = 2 * math.pi * torch.rand(num_elements, dim)
    # real part
    real = (modulus * phase.cos()).unsqueeze(dim=-1)
    # purely imaginary quaternions unitary
    imag = torch.rand(num_elements, dim, 3)
    imag = functional.normalize(imag, p=2, dim=-1)
    imag = imag * (modulus * phase.sin()).unsqueeze(dim=-1)
    x = torch.cat([real, imag], dim=-1)
    return x.view(num_elements, 4 * dim)
