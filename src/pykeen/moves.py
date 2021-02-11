# -*- coding: utf-8 -*-

"""A compatibility layer for different versions of PyTorch."""

try:
    from torch.fft import rfft, irfft  # works on pytorch >= 1.7
except ModuleNotFoundError:
    from torch import rfft, irfft  # works on pytorch < 1.7

__all__ = [
    'rfft',
    'irfft',
]
