"""Tests for algebraic utilities."""

import torch

import pykeen.nn.algebra


def _test_multiplication_table(t: torch.Tensor):
    """Test properties of multiplication tables."""
    # check type
    assert torch.is_tensor(t)
    # check size
    assert t.ndim == 3
    n = t.shape[0]
    assert t.shape == (n, n, n)
    # check value range
    cond = torch.zeros_like(t, dtype=torch.bool)
    for v in {-1, 0, 1}:
        cond |= torch.isclose(t, torch.full_like(t, fill_value=v))
    assert cond.all()


def test_quaternion_multiplication_table():
    """Test quaternion multiplication table."""
    _test_multiplication_table(pykeen.nn.algebra.quaterion_multiplication_table())
