"""Functional forms of interaction methods."""
from typing import Tuple

import torch

from ..utils import normalize_for_einsum, split_complex

__all__ = [
    "complex_interaction",
    "distmult_interaction",
]


def _normalize_terms_for_einsum(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, str, torch.FloatTensor, str, torch.FloatTensor, str]:
    batch_size = max(h.shape[0], r.shape[0], t.shape[0])
    h_term, h = normalize_for_einsum(x=h, batch_size=batch_size, symbol='h')
    r_term, r = normalize_for_einsum(x=r, batch_size=batch_size, symbol='r')
    t_term, t = normalize_for_einsum(x=t, batch_size=batch_size, symbol='t')
    return h, h_term, r, r_term, t, t_term


def distmult_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    h, h_term, r, r_term, t, t_term = _normalize_terms_for_einsum(h, r, t)
    return torch.einsum(f'{h_term},{r_term},{t_term}->bhrt', h, r, t)


def complex_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    h, h_term, r, r_term, t, t_term = _normalize_terms_for_einsum(h, r, t)
    (h_re, h_im), (r_re, r_im), (t_re, t_im) = [split_complex(x=x) for x in (h, r, t)]
    return sum(
        torch.einsum(f'{h_term},{r_term},{t_term}->bhrt', hh, rr, tt)
        for hh, rr, tt in [
            (h_re, r_re, t_re),
            (h_re, r_im, t_im),
            (h_im, r_re, t_im),
            (h_im, r_im, t_re),
        ]
    )
