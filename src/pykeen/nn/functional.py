"""Functional forms of interaction methods."""
import torch

from ..utils import normalize_for_einsum


def distmult_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    batch_size = max(h.shape[0], r.shape[0], t.shape[0])
    h_term, h = normalize_for_einsum(x=h, batch_size=batch_size, symbol='h')
    r_term, r = normalize_for_einsum(x=r, batch_size=batch_size, symbol='r')
    t_term, t = normalize_for_einsum(x=t, batch_size=batch_size, symbol='t')
    return torch.einsum(f'{h_term},{r_term},{t_term}->bhrt', h, r, t)
