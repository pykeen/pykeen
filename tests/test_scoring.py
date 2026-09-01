"""Smoke-tests for fast scoring."""

from __future__ import annotations

import pytest
import torch

from pykeen.models import DistMult, ERModel
from pykeen.models.scoring import Batch, Scorer
from pykeen.triples import KGInfo
from pykeen.typing import LongTensor


@pytest.fixture()
def generator() -> torch.Generator:
    """Build a random number generator with fixed seed."""
    return torch.manual_seed(42)


def make_tensor(size: tuple[int, ...] | None, num: int, generator: torch.Generator) -> LongTensor | None:
    """Create a tensor with the given size."""
    if size is None:
        return None
    return torch.randint(num, size=size, generator=generator)


NUM_ENTITIES = 7
NUM_RELATIONS = 5


@pytest.fixture()
def model() -> ERModel:
    """Build a test model."""
    return DistMult(
        triples_factory=KGInfo(num_entities=NUM_ENTITIES, num_relations=NUM_RELATIONS, create_inverse_triples=False),
        embedding_dim=8,
    )


B = 2
K = 3


@pytest.mark.parametrize(
    "hd, rd, td, shape",
    [
        # hrt
        ((B,), (B,), (B,), (B,)),
        # score_t
        ((B,), (B,), None, (B, NUM_ENTITIES)),
        # score_r
        ((B,), None, (B,), (B, NUM_RELATIONS)),
        # score_h
        (None, (B,), (B,), (B, NUM_ENTITIES)),
        # score_ts
        ((B,), (B,), (B, K), (B, K)),
        # score_ts_all?
        ((B,), (B, K), None, (B, K, NUM_ENTITIES)),
    ],
    ids=str,
)
def test_scoring(
    hd: tuple[int, ...] | None,
    rd: tuple[int, ...] | None,
    td: tuple[int, ...] | None,
    shape: tuple[int, ...],
    model: ERModel,
    generator: torch.Generator,
) -> None:
    """Test scorer."""
    scorer = Scorer()
    scores = scorer.score(
        model=model,
        batch=Batch(
            head=make_tensor(size=hd, num=model.num_entities, generator=generator),
            relation=make_tensor(size=rd, num=model.num_relations, generator=generator),
            tail=make_tensor(size=td, num=model.num_entities, generator=generator),
        ),
    )
    assert scores.shape == shape
