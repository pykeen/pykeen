# -*- coding: utf-8 -*-

"""A compatibility layer for different versions of PyTorch."""

from typing import Optional

import torch

try:
    from torch.fft import rfft, irfft  # works on pytorch >= 1.7

except ImportError:
    from torch import rfft as old_rfft, irfft as old_irfft  # works on pytorch < 1.7

    def _resolve_normalized_option(norm: Optional[str]) -> bool:
        """Convert PyTorch >= 1.7 "norm" option to <1.7 "normalized" option."""
        if norm is None or norm == "backward":
            normalized = False
        elif norm == "ortho":
            normalized = True
        else:
            raise NotImplementedError("In PyTorch < 1.7, there is no \"forward\" option.")
        return normalized

    def rfft(
        input: torch.Tensor,
        n: Optional[int] = None,
        dim: int = -1,
        norm: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute 1D real-to-complex FFT."""
        if dim != -1:
            raise ValueError("In PyTorch < 1.7, there is no dim argument.")
        normalized = _resolve_normalized_option(norm)
        if n is not None:
            m = input.shape[-1]
            if m < n:
                # pad with zeros
                input = torch.cat([input, input.new_zeros(*input.shape[:-1], n - m)], dim=-1)
            elif m > n:
                # trim
                input = input[..., :n]
        return old_rfft(input, signal_ndim=1, normalized=normalized, onesided=True)

    def irfft(
        input: torch.Tensor,
        n: Optional[int] = None,
        dim: int = -1,
        norm: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute inverse FFT."""
        if dim != -1:
            raise ValueError("In PyTorch < 1.7, there is no dim argument.")
        normalized = _resolve_normalized_option(norm)
        return old_irfft(input, signal_ndim=1, normalized=normalized, onesided=True, signal_sizes=(n,))

__all__ = [
    'rfft',
    'irfft',
]
