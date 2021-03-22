"""Tests for message passing blocks."""
import unittest

import pykeen.nn.message_passing
from tests.cases import DecompositionTestCase


class BlockDecompositionTests(DecompositionTestCase, unittest.TestCase):
    """Tests for block Decomposition."""

    cls = pykeen.nn.message_passing.BlockDecomposition


class _BasesDecompositionTests(DecompositionTestCase):
    """Tests for bases Decomposition."""

    cls = pykeen.nn.message_passing.BasesDecomposition


class LowMemoryBasesDecompositionTests(_BasesDecompositionTests, unittest.TestCase):
    """Tests for BasesDecomposition with low memory requirement."""

    kwargs = dict(
        num_bases=4,
        memory_intense=False,
    )


class HighMemoryBasesDecompositionTests(_BasesDecompositionTests, unittest.TestCase):
    """Tests for BasesDecomposition with high memory requirement."""

    kwargs = dict(
        num_bases=4,
        memory_intense=True,
    )
