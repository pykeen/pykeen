# -*- coding: utf-8 -*-

"""Test that CUDA exceptions are processed properly."""

import unittest

from pykeen.utils import _CUDA_OOM_ERROR, _CUDNN_ERROR, is_cuda_oom_error, is_cudnn_error


class TestCudaExceptionsHandling(unittest.TestCase):
    """Test handling of CUDA exceptions."""

    not_cuda_error = Exception("Something else.")

    def test_is_cuda_oom_error(self):
        """Test handling of a CUDA out of memory exception."""
        error = RuntimeError(_CUDA_OOM_ERROR)
        self.assertTrue(is_cuda_oom_error(runtime_error=error))
        self.assertFalse(is_cudnn_error(runtime_error=error))

        self.assertFalse(is_cuda_oom_error(runtime_error=self.not_cuda_error))

    def test_is_cudnn_error(self):
        """Test handling of a cuDNN error."""
        error = RuntimeError(_CUDNN_ERROR)
        self.assertTrue(is_cudnn_error(runtime_error=error))
        self.assertFalse(is_cuda_oom_error(runtime_error=error))

        error = Exception("Something else.")
        self.assertFalse(is_cudnn_error(runtime_error=self.not_cuda_error))
