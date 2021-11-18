# -*- coding: utf-8 -*-

"""Compute kernels for common sub-tasks."""

import numpy
import torch

from ..utils import extended_einsum, split_complex, tensor_product, view_complex, view_complex_native

__all__ = [
    "batched_dot",
    "batched_complex",
]


def _batched_dot_manual(
    a: torch.FloatTensor,
    b: torch.FloatTensor,
) -> torch.FloatTensor:
    return (a * b).sum(dim=-1)


# TODO benchmark
def _batched_dot_matmul(
    a: torch.FloatTensor,
    b: torch.FloatTensor,
) -> torch.FloatTensor:
    return (a.unsqueeze(dim=-2) @ b.unsqueeze(dim=-1)).view(a.shape[:-1])


# TODO benchmark
def _batched_dot_einsum(
    a: torch.FloatTensor,
    b: torch.FloatTensor,
) -> torch.FloatTensor:
    return torch.einsum("...i,...i->...", a, b)


def batched_dot(
    a: torch.FloatTensor,
    b: torch.FloatTensor,
) -> torch.FloatTensor:
    """Compute "element-wise" dot-product between batched vectors."""
    return _batched_dot_manual(a, b)


# TODO benchmark
def _complex_broadcast_optimized(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Manually split into real/imag, and used optimized broadcasted combination."""
    (h_re, h_im), (r_re, r_im), (t_re, t_im) = [split_complex(x=x) for x in (h, r, t)]
    return sum(
        factor * tensor_product(hh, rr, tt).sum(dim=-1)
        for factor, hh, rr, tt in [
            (+1, h_re, r_re, t_re),
            (+1, h_re, r_im, t_im),
            (+1, h_im, r_re, t_im),
            (-1, h_im, r_im, t_re),
        ]
    )


# TODO benchmark
def _complex_direct(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Manually split into real/imag, and directly evaluate interaction."""
    (h_re, h_im), (r_re, r_im), (t_re, t_im) = [split_complex(x=x) for x in (h, r, t)]
    return (
        (h_re * r_re * t_re).sum(dim=-1)
        + (h_re * r_im * t_im).sum(dim=-1)
        + (h_im * r_re * t_im).sum(dim=-1)
        - (h_im * r_im * t_re).sum(dim=-1)
    )


# TODO benchmark
def _complex_einsum(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Use einsum."""
    x = h.new_zeros(2, 2, 2)
    x[0, 0, 0] = 1
    x[0, 1, 1] = 1
    x[1, 0, 1] = 1
    x[1, 1, 0] = -1
    return extended_einsum(
        "ijk,bhrtdi,bhrtdj,bhrtdk->bhrt",
        x,
        h.view(*h.shape[:-1], -1, 2),
        r.view(*r.shape[:-1], -1, 2),
        t.view(*t.shape[:-1], -1, 2),
    )


def _complex_native_complex(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Use torch built-ins for computation with complex numbers."""
    h, r, t = [view_complex_native(x=x) for x in (h, r, t)]
    return torch.real(tensor_product(h, r, torch.conj(t)).sum(dim=-1))


# TODO benchmark
def _complex_native_complex_select(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Use torch built-ins for computation with complex numbers and select whether to combine hr or ht first."""
    h, r, t = [view_complex(x=x) for x in (h, r, t)]
    hr_cost = numpy.prod([max(hs, rs) for hs, rs in zip(h.shape, r.shape)])
    rt_cost = numpy.prod([max(ts, rs) for ts, rs in zip(t.shape, r.shape)])
    t = torch.conj(t)
    if hr_cost < rt_cost:
        h = h * r
    else:
        t = r * t
    return torch.real((h * t).sum(dim=-1))


# TODO benchmark
def _complex_select(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Decide based on result shape whether to combine hr or ht first."""
    hr_cost = numpy.prod([max(hs, rs) for hs, rs in zip(h.shape, r.shape)])
    rt_cost = numpy.prod([max(ts, rs) for ts, rs in zip(t.shape, r.shape)])
    (h_re, h_im), (r_re, r_im), (t_re, t_im) = [split_complex(x=x) for x in (h, r, t)]
    if hr_cost < rt_cost:
        h_re, h_im = (h_re * r_re - h_im * r_im), (h_re * r_im + h_im * r_re)
    else:
        t_re, t_im = (t_re * r_re - t_im * r_im), (t_re * r_im + t_im * r_re)
    return h_re @ t_re.transpose(-2, -1) - h_im @ t_im.transpose(-2, -1)


def _complex_to_stacked(h, r, t):
    (r_re, r_im), (t_re, t_im) = [split_complex(x=x) for x in (r, t)]
    h = torch.cat([h, h], dim=-1)  # re im re im
    r = torch.cat([r_re, r_re, r_im, r_im], dim=-1)  # re re im im
    t = torch.cat([t_re, -t_im, -t_im, t_re], dim=-1)  # re im im re
    return h, r, t


# TODO benchmark
def _complex_stacked(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Stack vectors."""
    h, r, t = _complex_to_stacked(h, r, t)
    return (h * r * t).sum(dim=-1)


# TODO benchmark
def _complex_stacked_select(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Stack vectors and select order."""
    h, r, t = _complex_to_stacked(h, r, t)
    hr_cost = numpy.prod([max(hs, rs) for hs, rs in zip(h.shape, r.shape)])
    rt_cost = numpy.prod([max(ts, rs) for ts, rs in zip(t.shape, r.shape)])
    if hr_cost < rt_cost:
        # h = h_re, -h_im
        h = h * r
    else:
        t = r * t
    return h @ t.transpose(-2, -1)


def batched_complex(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """Compute real part of tri-linear complex dot product."""
    return _complex_native_complex(h=h, r=r, t=t)
