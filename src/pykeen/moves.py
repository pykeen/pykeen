# -*- coding: utf-8 -*-

"""A compatibility layer for different versions of PyTorch."""
import torch

try:
    from torch.fft import rfft, irfft  # works on pytorch >= 1.7

except ModuleNotFoundError:
    from torch import rfft as old_rfft, irfft as old_irfft  # works on pytorch < 1.7


    def rfft(x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute 1D real-to-complex FFT."""
        # new parameters
        assert kwargs.pop("dim", -1) == -1
        norm = kwargs.pop("norm", None) or "backward"
        assert norm == "backward"
        return old_rfft(x, signal_ndim=1, onesided=True, normalized=False, **kwargs)


    def irfft(x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute inverse FFT."""
        raise NotImplementedError

__all__ = [
    'rfft',
    'irfft',
]
