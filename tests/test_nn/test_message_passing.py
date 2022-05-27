# -*- coding: utf-8 -*-

"""Tests for message passing blocks."""

import pykeen.nn.message_passing
from tests import cases
import unittest_templates


class BlockDecompositionTests(cases.DecompositionTestCase):
    """Tests for block Decomposition."""

    cls = pykeen.nn.message_passing.BlockDecomposition


class EfficientBasesDecompositionTestCase(cases.BasesDecompositionTestCase):
    """Tests for efficient bases decomposition."""

    cls = pykeen.nn.message_passing.EfficientBasesDecomposition


class EfficientBlockDecompositionTestCase(cases.DecompositionTestCase):
    """Tests for efficient block decomposition."""

    cls = pykeen.nn.message_passing.EfficientBlockDecomposition


class LowMemoryBasesDecompositionTestCase(cases.BasesDecompositionTestCase):
    """Tests for BasesDecomposition with low memory requirement."""

    kwargs = dict(
        num_bases=4,
        memory_intense=False,
    )


class HighMemoryBasesDecompositionTestCase(cases.BasesDecompositionTestCase):
    """Tests for BasesDecomposition with high memory requirement."""

    kwargs = dict(
        num_bases=4,
        memory_intense=True,
    )


class DecompositionMetaTestCase(unittest_templates.MetaTestCase):
    """A test for tests of all decompositions."""

    base_cls = pykeen.nn.message_passing.Decomposition
    base_test = cases.DecompositionTestCase
    skip_cls = {
        # mixin
        pykeen.nn.message_passing.EfficientDecomposition,
    }


class EfficientBasesDecompositionUtilTestCase(cases.EfficientDecompositionUtilTestCase):
    """Test case for efficient bases decomposition utils."""

    cls = pykeen.nn.message_passing.EfficientBasesDecomposition


class EfficientBlockDecompositionUtilTestCase(cases.EfficientDecompositionUtilTestCase):
    """Test case for efficient block decomposition utils."""

    cls = pykeen.nn.message_passing.EfficientBlockDecomposition


class EfficientDecompositionUtilMetaTestCase(unittest_templates.MetaTestCase):
    """Test for tests for efficient decomposition mixin."""

    base_test = cases.EfficientDecompositionUtilTestCase
    base_cls = pykeen.nn.message_passing.EfficientDecomposition
