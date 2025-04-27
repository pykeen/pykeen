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


@pytest.fixture()
def model() -> ERModel:
    """Build a test model."""
    return DistMult(
        triples_factory=KGInfo(num_entities=20, num_relations=3, create_inverse_triples=False), embedding_dim=8
    )


@pytest.mark.parametrize(
    "hd, rd, td",
    [
        # hrt
        ((2,), (2,), (2,)),
        # score_t
        ((2,), (2,), None),
        # score_r
        ((2,), None, (2,)),
        # score_h
        (None, (2,), (2,)),
        # score_ts
        ((2,), (2,), (2, 5)),
        # score_ts_all?
        ((2,), (2, 3), None),
    ],
)
def test_scoring(
    model: ERModel,
    hd: tuple[int, ...] | None,
    rd: tuple[int, ...] | None,
    td: tuple[int, ...] | None,
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
    assert scores.shape
