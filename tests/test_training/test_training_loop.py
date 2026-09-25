"""Tests for training loops."""

import pytest

from pykeen.datasets import Nations
from pykeen.losses import BCEWithLogitsLoss, CrossEntropyLoss, MarginRankingLoss, NSSALoss, SoftplusLoss
from pykeen.losses import CrossEntropyLoss, MarginRankingLoss, NSSALoss, SoftplusLoss
from pykeen.models import TransE
from pykeen.sampling.filtering import BloomFilterer, PythonSetFilterer
from pykeen.training import BatchCWATrainingLoop, LCWATrainingLoop, SLCWATrainingLoop, SymmetricLCWATrainingLoop
from tests.test_training import cases


class MRUnfilteredSLCWATrainingLoopTestCase(cases.SLCWATrainingLoopTestCase):
    """Test sLCWA with unfiltered negative sampling with margin ranking loss."""

    cls = SLCWATrainingLoop
    filterer_cls = None
    loss_cls = MarginRankingLoss


class NSSAUnfilteredSLCWATrainingLoopTestCase(cases.SLCWATrainingLoopTestCase):
    """Test sLCWA with unfiltered negative sampling with NSSA loss."""

    cls = SLCWATrainingLoop
    filterer_cls = None
    loss_cls = NSSALoss


class SoftplusUnfilteredSLCWATrainingLoopTestCase(cases.SLCWATrainingLoopTestCase):
    """Test sLCWA with unfiltered negative sampling with softplus loss."""

    cls = SLCWATrainingLoop
    filterer_cls = None
    loss_cls = SoftplusLoss


class MRSetFilteredSLCWATrainingLoopTestCase(cases.SLCWATrainingLoopTestCase):
    """Test sLCWA with set filtered negative sampling with margin ranking loss."""

    cls = SLCWATrainingLoop
    filterer_cls = PythonSetFilterer
    loss_cls = MarginRankingLoss


class NSSASetFilteredSLCWATrainingLoopTestCase(cases.SLCWATrainingLoopTestCase):
    """Test sLCWA with set filtered negative sampling with NSSA loss."""

    cls = SLCWATrainingLoop
    filterer_cls = PythonSetFilterer
    loss_cls = NSSALoss


class SoftplusSetFilteredSLCWATrainingLoopTestCase(cases.SLCWATrainingLoopTestCase):
    """Test sLCWA with set filtered negative sampling with softplus loss."""

    cls = SLCWATrainingLoop
    filterer_cls = PythonSetFilterer
    loss_cls = SoftplusLoss


# Multiple permutations of loss not necessary for bloom filter since it's more of a
# filter vs. no filter thing.
class BloomFilteredSLCWATrainingLoopTestCase(cases.SLCWATrainingLoopTestCase):
    """Test sLCWA with bloom filtered negative sampling."""

    cls = SLCWATrainingLoop
    filterer_cls = BloomFilterer
    loss_cls = MarginRankingLoss


class GroupedSLCWATrainingLoopTestCase(cases.SLCWATrainingLoopTestCase):
    """Test sLCWA with grouped negative sampling."""

    cls = SLCWATrainingLoop
    filterer_cls = None
    loss_cls = MarginRankingLoss
    kwargs = {"grouped": True}


class MRLossLCWATrainingLoopTestCase(cases.TrainingLoopTestCase):
    """Test LCWA with margin ranking loss."""

    cls = LCWATrainingLoop
    loss_cls = MarginRankingLoss


class NSSALossLCWATrainingLoopTestCase(cases.TrainingLoopTestCase):
    """Test LCWA with NSSA loss."""

    cls = LCWATrainingLoop
    loss_cls = NSSALoss


class SoftPlusLCWATrainingLoopTestCase(cases.TrainingLoopTestCase):
    """Test LCWA with softplus loss."""

    cls = LCWATrainingLoop
    loss_cls = SoftplusLoss


class SymmetricLCWATrainingLoopTestCase(cases.TrainingLoopTestCase):
    """Test for symmetric LCWA with cross-entropy."""

    cls = SymmetricLCWATrainingLoop
    loss_cls = CrossEntropyLoss


class BCELossBatchCWATrainingLoopTestCase(cases.TrainingLoopTestCase):
    """Test batch-local CWA with binary cross entropy loss."""

    cls = BatchCWATrainingLoop
    loss_cls = BCEWithLogitsLoss


class CrossEntropyLossBatchCWATrainingLoopTestCase(cases.TrainingLoopTestCase):
    """Test batch-local CWA with cross entropy loss."""

    cls = BatchCWATrainingLoop
    loss_cls = CrossEntropyLoss


class MRLossBatchCWATrainingLoopTestCase(cases.TrainingLoopTestCase):
    """Test batch-local CWA with margin ranking loss."""

    cls = BatchCWATrainingLoop
    loss_cls = MarginRankingLoss


class NSSALossBatchCWATrainingLoopTestCase(cases.TrainingLoopTestCase):
    """Test batch-local CWA with NSSA loss."""

    cls = BatchCWATrainingLoop
    loss_cls = NSSALoss


class CrossEntropyLossHeadBatchCWATrainingLoopTestCase(cases.TrainingLoopTestCase):
    """Test batch-local CWA with cross entropy loss and head prediction."""

    cls = BatchCWATrainingLoop
    loss_cls = CrossEntropyLoss
    kwargs = {"target": "head"}


class CrossEntropyLossRelationBatchCWATrainingLoopTestCase(cases.TrainingLoopTestCase):
    """Test batch-local CWA with cross entropy loss and relation prediction."""

    cls = BatchCWATrainingLoop
    loss_cls = CrossEntropyLoss
    kwargs = {"target": "relation"}


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        # Nations has 14 entities and 55 relations
        ("head", 7),
        ("relation", 28),
        ("tail", 7),
    ],
)
def test_lcwa_initial_slice_size(target: str, expected: int) -> None:
    """Test that the slice size search starts at half the number of candidates for the target."""
    triples_factory = Nations().training
    model = TransE(triples_factory=triples_factory)
    loop = LCWATrainingLoop(model=model, triples_factory=triples_factory, target=target)
    assert loop._get_initial_slice_size(batch_size=32) == expected


@pytest.mark.parametrize(
    ("target", "batch_size", "expected"),
    [
        # Nations has 14 entities and 55 relations
        ("tail", 32, 7),
        ("relation", 32, 16),
        ("relation", 100, 28),
    ],
)
def test_bcwa_initial_slice_size(target: str, batch_size: int, expected: int) -> None:
    """Test that the slice size search starts at half the number of the batch's candidates for the target."""
    triples_factory = Nations().training
    model = TransE(triples_factory=triples_factory)
    loop = BatchCWATrainingLoop(model=model, triples_factory=triples_factory, target=target)
    assert loop._get_initial_slice_size(batch_size=batch_size) == expected
