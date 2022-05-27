# -*- coding: utf-8 -*-

"""Tests for message passing blocks."""

import unittest_templates

import pykeen.nn.message_passing
from tests import cases


class BlockDecompositionTests(cases.DecompositionTestCase):
    """Tests for block decomposition."""

    cls = pykeen.nn.message_passing.BlockDecomposition


class BasesDecompositionTests(cases.DecompositionTestCase):
    """Tests for bases decomposition."""

    cls = pykeen.nn.message_passing.BasesDecomposition


class DecompositionMetaTestCase(unittest_templates.MetaTestCase):
    """A test for tests of all decompositions."""

    base_cls = pykeen.nn.message_passing.Decomposition
    base_test = cases.DecompositionTestCase
