"""Contract tests for the model scoring API.

These tests pin the shape and value semantics shared by :meth:`~pykeen.models.Model.score_hrt`,
:meth:`~pykeen.models.Model.score_h`, :meth:`~pykeen.models.Model.score_r`, :meth:`~pykeen.models.Model.score_t`,
and the :meth:`~pykeen.models.Model.score` dispatcher, in particular the ``heads`` / ``relations`` / ``tails``
restriction, which may be given either shared across the batch (shape: ``(num,)``) or per batch element
(shape: ``(batch_size, num)``). The latter is what grouped sLCWA training relies on, cf.
:class:`~pykeen.triples.instances.GroupedSLCWABatch`.
"""

from __future__ import annotations

import pytest
import torch

from pykeen.constants import TARGET_TO_KEYS
from pykeen.models import UM, DistMult, ERModel, ScoringBatch
from pykeen.triples import KGInfo
from pykeen.typing import LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, LongTensor, Target

NUM_ENTITIES = 7
NUM_RELATIONS = 5
BATCH_SIZE = 3
NUM_IDS = 4

#: for each target: the number of scored candidates, and the keyword to restrict them
TARGETS: dict[Target, tuple[int, str]] = {
    LABEL_HEAD: (NUM_ENTITIES, "heads"),
    LABEL_RELATION: (NUM_RELATIONS, "relations"),
    LABEL_TAIL: (NUM_ENTITIES, "tails"),
}

#: ``UM`` has no relation representations, and thus exercises the score-repetition path
MODEL_CLASSES = [DistMult, UM]


@pytest.fixture(params=MODEL_CLASSES, ids=[cls.__name__ for cls in MODEL_CLASSES])
def model(request) -> ERModel:
    """Return a small model with deterministic parameters."""
    return request.param(
        triples_factory=KGInfo(num_entities=NUM_ENTITIES, num_relations=NUM_RELATIONS, create_inverse_triples=False),
        embedding_dim=8,
        random_seed=42,
    ).eval()


@pytest.fixture
def hrt_batch() -> LongTensor:
    """Return a batch of triples."""
    generator = torch.manual_seed(7)
    return torch.stack(
        [
            torch.randint(NUM_ENTITIES, size=(BATCH_SIZE,), generator=generator),
            torch.randint(NUM_RELATIONS, size=(BATCH_SIZE,), generator=generator),
            torch.randint(NUM_ENTITIES, size=(BATCH_SIZE,), generator=generator),
        ],
        dim=-1,
    )


def _ids(target: Target, per_batch: bool) -> LongTensor:
    """Return candidate IDs for the given target, either shared or per batch element."""
    num, _ = TARGETS[target]
    generator = torch.manual_seed(13)
    size = (BATCH_SIZE, NUM_IDS) if per_batch else (NUM_IDS,)
    return torch.randint(num, size=size, generator=generator)


def test_score_hrt(model: ERModel, hrt_batch: LongTensor) -> None:
    """Test that ``score_hrt`` returns one score per triple."""
    assert model.score_hrt(hrt_batch).shape == (BATCH_SIZE, 1)


@pytest.mark.parametrize("target", list(TARGETS))
def test_score_target_all(model: ERModel, hrt_batch: LongTensor, target: Target) -> None:
    """Test that unrestricted 1:n scoring returns scores for all candidates."""
    num, _ = TARGETS[target]
    assert model.score(hrt_batch, target=target).shape == (BATCH_SIZE, num)


@pytest.mark.parametrize("target", list(TARGETS))
@pytest.mark.parametrize("per_batch", [False, True])
def test_score_target_restricted(model: ERModel, hrt_batch: LongTensor, target: Target, per_batch: bool) -> None:
    """Test that restricted 1:n scoring returns scores for the requested candidates only."""
    ids = _ids(target=target, per_batch=per_batch)
    assert model.score(hrt_batch, target=target, ids=ids).shape == (BATCH_SIZE, NUM_IDS)


@pytest.mark.parametrize("target", list(TARGETS))
@pytest.mark.parametrize("per_batch", [False, True])
def test_restriction_matches_full_scoring(
    model: ERModel, hrt_batch: LongTensor, target: Target, per_batch: bool
) -> None:
    """Test that restricting the candidates selects a sub-set of the unrestricted scores."""
    ids = _ids(target=target, per_batch=per_batch)
    full = model.score(hrt_batch, target=target)
    restricted = model.score(hrt_batch, target=target, ids=ids)
    expected = full.gather(dim=-1, index=ids.expand(BATCH_SIZE, NUM_IDS))
    torch.testing.assert_close(restricted, expected)


@pytest.mark.parametrize("target", list(TARGETS))
def test_restriction_matches_score_hrt(model: ERModel, hrt_batch: LongTensor, target: Target) -> None:
    """Test that per-batch restricted 1:n scoring agrees with scoring the explicit triples."""
    ids = _ids(target=target, per_batch=True)
    restricted = model.score(hrt_batch, target=target, ids=ids)

    # build the explicit triples corresponding to the restricted scoring
    triples = hrt_batch.unsqueeze(dim=1).repeat(1, NUM_IDS, 1)
    triples[..., list(TARGETS).index(target)] = ids
    expected = model.score_hrt(triples.view(-1, 3)).view(BATCH_SIZE, NUM_IDS)

    torch.testing.assert_close(restricted, expected)


@pytest.mark.parametrize("target", list(TARGETS))
@pytest.mark.parametrize("ids_kind", ["none", "shared", "per_batch"])
def test_slicing(model: ERModel, hrt_batch: LongTensor, target: Target, ids_kind: str) -> None:
    """Test that slicing does not change the result."""
    ids = None if ids_kind == "none" else _ids(target=target, per_batch=ids_kind == "per_batch")
    unsliced = model.score(hrt_batch, target=target, ids=ids)
    sliced = model.score(hrt_batch, target=target, ids=ids, slice_size=2)
    torch.testing.assert_close(sliced, unsliced)


@pytest.mark.parametrize("target", list(TARGETS))
@pytest.mark.parametrize("per_batch", [False, True])
def test_score_dispatch(model: ERModel, hrt_batch: LongTensor, target: Target, per_batch: bool) -> None:
    """Test that the ``score`` dispatcher agrees with the target-specific method."""
    _, ids_name = TARGETS[target]
    ids = _ids(target=target, per_batch=per_batch)
    method = getattr(model, f"score_{target[0]}")

    expected = method(hrt_batch[:, TARGET_TO_KEYS[target]], **{ids_name: ids})
    torch.testing.assert_close(model.score(hrt_batch, target=target, ids=ids), expected)
    # ... and that ``full_batch=False`` skips the column selection
    torch.testing.assert_close(
        model.score(hrt_batch[:, TARGET_TO_KEYS[target]], target=target, ids=ids, full_batch=False), expected
    )


def test_score_invalid_target(model: ERModel, hrt_batch: LongTensor) -> None:
    """Test that an invalid target is rejected."""
    with pytest.raises(ValueError, match="nope"):
        model.score(hrt_batch, target="nope")


class TestScoringBatch:
    """Tests for the scoring request representation."""

    @staticmethod
    def _index(*shape: int) -> LongTensor:
        """Return an index tensor of the given shape."""
        return torch.zeros(shape, dtype=torch.long)

    @pytest.mark.parametrize("target", list(TARGETS))
    def test_batch_shape(self, target: Target) -> None:
        """Test that the batch shape is inferred from the non-target index tensors."""
        indices = {label: self._index(BATCH_SIZE) for label in TARGETS if label != target}
        batch = ScoringBatch(**indices, **{target: None}, target=target)
        assert batch.batch_ndim == 1
        assert batch.batch_shape == (BATCH_SIZE,)
        assert batch.shared_target
        assert batch.target_ids is None

    @pytest.mark.parametrize("target", list(TARGETS))
    @pytest.mark.parametrize("per_batch", [False, True])
    def test_shared_target(self, target: Target, per_batch: bool) -> None:
        """Test that shared and per-batch candidates are distinguished."""
        indices = {label: self._index(BATCH_SIZE) for label in TARGETS if label != target}
        ids = self._index(BATCH_SIZE, NUM_IDS) if per_batch else self._index(NUM_IDS)
        batch = ScoringBatch(**indices, **{target: ids}, target=target)
        assert batch.shared_target is not per_batch

    def test_no_target(self) -> None:
        """Test that the index tensors are left-aligned when there is no target."""
        batch = ScoringBatch(
            head=self._index(BATCH_SIZE), relation=self._index(BATCH_SIZE), tail=self._index(BATCH_SIZE, NUM_IDS)
        )
        assert batch.batch_ndim == 2
        assert batch.batch_shape == (BATCH_SIZE, NUM_IDS)
        # the (batch_size,) index tensors gained a trailing singleton dimension
        assert batch.head.shape == (BATCH_SIZE, 1)
        # ... and there is no target axis to insert
        assert batch.lookup_indices == batch.indices

    @pytest.mark.parametrize("target", list(TARGETS))
    def test_lookup_indices(self, target: Target) -> None:
        """Test that the non-target index tensors gain the target axis."""
        indices = {label: self._index(BATCH_SIZE) for label in TARGETS if label != target}
        batch = ScoringBatch(**indices, **{target: None}, target=target)
        for label, index in zip(TARGETS, batch.lookup_indices, strict=True):
            assert index is None if label == target else index.shape == (BATCH_SIZE, 1)

    def test_with_target_ids(self) -> None:
        """Test replacing the target's index tensor."""
        batch = ScoringBatch(
            head=self._index(BATCH_SIZE), relation=self._index(BATCH_SIZE), tail=None, target=LABEL_TAIL
        )
        sliced = batch.with_target_ids(self._index(NUM_IDS))
        assert sliced.target == LABEL_TAIL
        assert sliced.batch_shape == batch.batch_shape
        assert sliced.target_ids is not None
        # the original is untouched
        assert batch.target_ids is None

    def test_missing_index(self) -> None:
        """Test that only the target may be None."""
        with pytest.raises(ValueError, match="Missing index tensors"):
            ScoringBatch(head=self._index(BATCH_SIZE), relation=None, tail=None, target=LABEL_TAIL)

    def test_unknown_target(self) -> None:
        """Test that an invalid target is rejected."""
        with pytest.raises(ValueError, match="nope"):
            ScoringBatch(
                head=self._index(BATCH_SIZE),
                relation=self._index(BATCH_SIZE),
                tail=self._index(BATCH_SIZE),
                target="nope",
            )

    def test_non_broadcastable(self) -> None:
        """Test that non-broadcastable index shapes are rejected."""
        with pytest.raises(ValueError, match="Cannot broadcast"):
            ScoringBatch(head=self._index(2), relation=self._index(3), tail=None, target=LABEL_TAIL)

    def test_invalid_target_shape(self) -> None:
        """Test that target IDs with an unusable number of dimensions are rejected."""
        with pytest.raises(ValueError, match="must have shape"):
            ScoringBatch(
                head=self._index(BATCH_SIZE),
                relation=self._index(BATCH_SIZE),
                tail=self._index(1, BATCH_SIZE, NUM_IDS),
                target=LABEL_TAIL,
            )

    def test_no_target_slicing(self, model: ERModel, hrt_batch: LongTensor) -> None:
        """Test that slicing a batch without a target is rejected."""
        batch = ScoringBatch(head=hrt_batch[:, 0], relation=hrt_batch[:, 1], tail=hrt_batch[:, 2])
        with pytest.raises(ValueError, match="requires a target"):
            model._score(batch, slice_size=2)

    def test_no_target_scoring(self, model: ERModel, hrt_batch: LongTensor) -> None:
        """Test that scoring without a target agrees with ``score_hrt``."""
        batch = ScoringBatch(head=hrt_batch[:, 0], relation=hrt_batch[:, 1], tail=hrt_batch[:, 2])
        torch.testing.assert_close(model._score(batch), model.score_hrt(hrt_batch).squeeze(dim=-1))

    def test_no_target_broadcasting(self, model: ERModel, hrt_batch: LongTensor) -> None:
        """Test that scoring without a target broadcasts a block of triples."""
        tails = _ids(target=LABEL_TAIL, per_batch=True)
        batch = ScoringBatch(head=hrt_batch[:, 0], relation=hrt_batch[:, 1], tail=tails)
        assert batch.batch_shape == (BATCH_SIZE, NUM_IDS)
        torch.testing.assert_close(
            model._score(batch), model.score_t(hrt_batch[:, TARGET_TO_KEYS[LABEL_TAIL]], tails=tails)
        )
