"""Test for ranks."""
from typing import Sequence

import pytest
import torch

import pykeen.evaluation.ranks


@pytest.mark.parametrize(
    ("batch_shape", "num_batches", "partial_num_choices"), [(tuple(), 4, 3), ((2,), 3, 7), ((2, 3), 2, 12)]
)
def test_rank_builder(batch_shape: Sequence[int], num_batches: int, partial_num_choices: int):
    """Test for rank builder."""
    generator = torch.manual_seed(seed=42)
    y_true = torch.rand(size=batch_shape, generator=generator)
    # initialize
    builder = pykeen.evaluation.ranks.RankBuilder(y_true=y_true)
    # update with batches
    total = 0
    for _ in range(num_batches):
        y_pred = torch.rand(size=(*batch_shape, partial_num_choices), generator=generator)
        builder = builder.update(y_pred=y_pred)
        total += partial_num_choices
        assert (builder.total == total).all()
        assert builder.larger.shape == batch_shape
    ranks = builder.compute()
    assert isinstance(ranks, pykeen.evaluation.ranks.Ranks)
