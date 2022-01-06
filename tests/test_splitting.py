"""Tests for splitting of triples."""
import torch

from pykeen.triples.splitting import (
    CleanupSplitter,
    CoverageSplitter,
    DeterministicCleaner,
    RandomizedCleaner,
    _get_cover_deterministic,
)
from pykeen.triples.utils import get_entities, get_relations
from tests.cases import CleanerTestCase, SplitterTestCase


class DeterministicCleanerTests(CleanerTestCase):
    """Tests for deterministic cleaner."""

    cls = DeterministicCleaner

    def test_manual(self):
        """Test that triples in a test set can get moved properly to the training set."""
        training = torch.as_tensor(
            data=[
                [1, 1000, 2],
                [1, 1000, 3],
                [1, 1001, 3],
            ],
            dtype=torch.long,
        )
        testing = torch.as_tensor(
            data=[
                [2, 1001, 3],
                [1, 1002, 4],
            ],
            dtype=torch.long,
        )
        expected_training = torch.as_tensor(
            data=[
                [1, 1000, 2],
                [1, 1000, 3],
                [1, 1001, 3],
                [1, 1002, 4],
            ],
            dtype=torch.long,
        )
        expected_testing = torch.as_tensor(
            data=[
                [2, 1001, 3],
            ],
            dtype=torch.long,
        )

        new_training, new_testing = self.instance.cleanup_pair(training, testing, random_state=...)
        assert (expected_training == new_training).all()
        assert (expected_testing == new_testing).all()


class RandomizedCleanerTests(CleanerTestCase):
    """Tests for deterministic cleaner."""

    cls = RandomizedCleaner

    def test_manual(self):
        """Test that triples in a test set can get moved properly to the training set."""
        training = torch.as_tensor(
            data=[
                [1, 1000, 2],
                [1, 1000, 3],
            ],
            dtype=torch.long,
        )
        testing = torch.as_tensor(
            data=[
                [2, 1000, 3],
                [1, 1000, 4],
                [2, 1000, 4],
                [1, 1001, 3],
            ],
            dtype=torch.long,
        )
        expected_training_1 = {
            (1, 1000, 2),
            (1, 1000, 3),
            (1, 1000, 4),
            (1, 1001, 3),
        }
        expected_testing_1 = {
            (2, 1000, 3),
            (2, 1000, 4),
        }

        expected_training_2 = {
            (1, 1000, 2),
            (1, 1000, 3),
            (2, 1000, 4),
            (1, 1001, 3),
        }
        expected_testing_2 = {
            (2, 1000, 3),
            (1, 1000, 4),
        }

        new_training, new_testing = [
            set(tuple(row) for row in arr.tolist())
            for arr in self.instance.cleanup_pair(training, testing, random_state=None)
        ]

        if expected_training_1 == new_training:
            self.assertEqual(expected_testing_1, new_testing)
        elif expected_training_2 == new_training:
            self.assertEqual(expected_testing_2, new_testing)
        else:
            self.fail("training was not correct")


class CleanupSplitterTest(SplitterTestCase):
    """Tests for cleanup splitter."""

    cls = CleanupSplitter


class CoverageSplitterTest(SplitterTestCase):
    """Tests for coverage splitter."""

    cls = CoverageSplitter

    def test_get_cover_deterministic(self):
        """Test _get_cover_deterministic."""
        # generated_triples = generate_triples()
        cover = _get_cover_deterministic(triples=self.mapped_triples)

        # check type
        assert torch.is_tensor(cover)
        assert cover.dtype == torch.bool
        # check format
        assert cover.shape == (self.mapped_triples.shape[0],)

        # check coverage
        self.assertEqual(
            get_entities(self.mapped_triples),
            get_entities(self.mapped_triples[cover]),
            msg="entity coverage is not full",
        )
        self.assertEqual(
            get_relations(self.mapped_triples),
            get_relations(self.mapped_triples[cover]),
            msg="relation coverage is not full",
        )
