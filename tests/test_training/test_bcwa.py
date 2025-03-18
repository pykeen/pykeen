"""Tests for training with batch-local closed world assumption."""

import pytest
import torch

from pykeen.training import bcwa
from pykeen.typing import LongTensor


@pytest.fixture()
def generator() -> torch.Generator:
    """Build a generator with fixed seed for reproducible tests."""
    return torch.manual_seed(seed=42)


def _make_unique(max_id: int, num: int, generator: torch.Generator) -> LongTensor:
    ind = torch.randperm(max_id, generator=generator)[:num]
    ind = ind.unique()
    assert len(ind) == num
    return ind


@pytest.mark.parametrize(
    ("num_unique_heads", "num_unique_relations", "num_unique_tails", "num_triples", "num_entities", "num_relations"),
    [(1, 1, 1, 1, 13, 7), (3, 2, 2, 3, 13, 7)],
)
def test_convert_to_batch_local(
    num_unique_heads: int,
    num_unique_relations: int,
    num_unique_tails: int,
    num_triples: int,
    num_entities: int,
    num_relations: int,
    generator: torch.Generator,
) -> None:
    """Test conversion of batch to local indices."""
    # verify valid test input
    assert num_unique_heads <= num_entities
    assert num_unique_relations <= num_relations
    assert num_unique_tails <= num_entities
    assert num_triples >= max(num_unique_heads, num_unique_relations, num_unique_tails)
    hs = _make_unique(max_id=num_entities, num=num_unique_heads, generator=generator)
    rs = _make_unique(max_id=num_relations, num=num_unique_relations, generator=generator)
    ts = _make_unique(max_id=num_entities, num=num_unique_tails, generator=generator)
    indices_l = []
    for num_unique in (num_unique_heads, num_unique_relations, num_unique_tails):
        index = torch.randint(0, num_unique, size=(num_triples,), generator=generator)
        # ensure that each index occurs at least once
        index[torch.randperm(num_triples)[:num_unique]] = torch.arange(num_unique)
        indices_l.append(index)
    indices_t = torch.stack(indices_l, dim=-1)
    mapped_triples = torch.stack([source[index] for source, index in zip((hs, rs, ts), indices_l, strict=True)], dim=-1)
    (hs_u, rs_u, ts_u), local_triples = bcwa._convert_to_batch_local(mapped_triples)
    assert torch.equal(indices_t, local_triples)
    for i, i_unique in zip((hs, rs, ts), (hs_u, rs_u, ts_u), strict=True):
        assert torch.equal(i.unique(), i_unique)
